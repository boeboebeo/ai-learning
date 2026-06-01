"""
==============================================
DAY 10: Complete Synthesizer Chain Integration
==============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.io import wavfile

SAMPLE_RATE = 44100

def polyblep_sawtooth(freq, duration, sample_rate):
    def polyblep_residual(t, dt):
        if t < dt:  #점프 지점 직후 구간
            t = t / dt
                # t = 0.0 점프 바로 직수, t = 1.0 dt 범위 끝
                # t = t/dt => t를 0~1사이로 정규화 함
            return t + t - t * t - 1.0
                # 점프 
        elif t > 1.0 - dt:
            t = (t - 1.0) / dt
            return t * t + t + t + 1.0
        else:
            # 
            return 0.0
        
    """
    1. t < dt
    점프가 일어나는 구간 : phase = 1.0 -> 0.0 으로 리셋되는 구간
                      phase = 0.001, 0.002 <- 점프 바로 직후 샘플들 
        => 보정 값이 음수 : naive_saw 에서 빼면 살짝 올라감
    
    2. t > 1.0 - dt
    직전 구간         : phase = 0.98, 0.99 <- 점프 바로 직전 샘플들

    3. 나머지 (else:)
    직후 구간         : phase = 0.001 <- 보정없음. 그대로

    sample 0 -> t = 0 < dt 이므로 
    
    """
        
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # 전체 구간 그냥 각 샘플로 나눈 것 t 
    dt = freq / sample_rate
        # dt 는 한 샘플이 위상에서 차지하는 크기
    phase = 0
    output = np.zeros_like(t)

    for i in range(len(t)):
        naive_saw = 2 * phase - 1
        correction = polyblep_residual(phase, dt)
        output[i] = naive_saw - correction
            # 내가 

        phase += dt
            # 한 샘플마다 dt값이 더해져서 phase 로 출력되고 그 값이 1이상이면 항상 -1.0 을 하여 
            # 0 ~ 1 사이로 머물게 함
        if phase >= 1.0:
            phase -= 1.0

    return output, t



def polyblep_square(freq, duration, sample_rate):
    def polyblep_residual(t, dt):
        if t < dt:
            t = t / dt
            return t + t - t * t - 1.0
        elif t > 1.0 - dt:
            t = (t - 1.0) / dt
            return t * t + t + t + 1.0
        else:
            return 0.0 
            # 점프 구간에서 먼 샘플들은 보정값 0.0으로 출력 : 아무 변화 안만듦
        
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dt = freq / sample_rate
    phase = 0
    output = np.zeros_like(t)

    for i in range(len(t)):
        naive_square = 1.0 if phase<0.5 else -1.0
            #phase 값이 0.5보다 작드면 다 1.0, 그것보다 크면 -1.0
        correction = polyblep_residual(phase, dt)
        correction += polyblep_residual(phase - 0.5, dt)
        output[i] = naive_square - correction

        phase += dt
        if phase >= 1.0:
            phase -= 1.0

    """ square 보정값
index  phase    naive_sq  corr1(0근처)  corr2(0.5근처)  correction  output
────────────────────────────────────────────────────────────────────────────
0      0.000    +1.0      -1.0          0.0             -1.0        +2.0  ← 점프 직후
1      0.010    +1.0       0.0          0.0              0.0        +1.0  ← 일반
2      0.020    +1.0       0.0          0.0              0.0        +1.0
...
49     0.489    +1.0       0.0          0.0              0.0        +1.0
50     0.499    +1.0       0.0         +0.59             0.59       +0.41 ← 0.5 점프 직전
51     0.508    -1.0       0.0         -0.06            -0.06       -0.94 ← 0.5 점프 직후
52     0.518    -1.0       0.0          0.0              0.0        -1.0  ← 일반
...
99     0.988    -1.0       0.0          0.0              0.0        -1.0
100    0.998    -1.0      +0.59         0.0             +0.59       -1.59 ← 1.0 점프 직전
101    0.008    +1.0      -0.06         0.0             -0.06       +1.06 ← 1.0 점프 직후
102    0.018    +1.0       0.0          0.0              0.0        +1.0  ← 일반
    """

    return output, t


def biquad_filter(signal_input, filter_type, cutoff_freq, resonance, sample_rate):
    omega = 2 * np.pi * cutoff_freq / sample_rate
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    Q = resonance
    alpha = sin_omega / (2 * Q)

    if filter_type == 'lowpass':  #📍각 계수에 대한 부분들 아직 모호
        b0 = (1 - cos_omega) / 2
        b1 = 1 - cos_omega
        b2 = (1 - cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha 
       
    
    elif filter_type == 'highpass':
        b0 = (1 + cos_omega) / 2
        b1 = -(1 + cos_omega)
        b2 = (1 + cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1/a0, a2/a0]) #📍여기 이 배열은 뭘까?

    filtered = signal.lfilter(b, a, signal_input)

    return filtered

def adsr_envelope(attack, decay, sustain, release, duration, sample_rate, gate_time=None):
    """
    ADSR Envelope

    Attack: 0 → 1 (상승 시간)
    Decay: 1 → sustain level (감쇠 시간)
    Sustain: constant level (지속 레벨)
    Release: sustain → 0 (릴리스 시간)
    
    Parameters in seconds (초 단위)
    gate_time: when note is released (노트 릴리스 시간)

    """
    if gate_time is None:
        gate_time = duration * 0.7
        # Default: 사용자 미지정 시 그냥 70%의 시간임 (전체 duration의)
        # Gate : gate on/off
        # gate time : 노트가 눌려져 있는 시간
        # gate_time is None : 사용자가 따로 지정을 안함 => 자동으로 70%로 설정함

    num_samples = int(duration * sample_rate)
    envelope = np.zeros(num_samples)

    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    gate_sample = int(gate_time * sample_rate)
        #gate_sample = 사용자가 노트를 누르고 있는동안의 샘플 개수

    current_sample = 0

    # Attack phase (attack 구간)
    if current_sample < attack_samples:
        end_sample = min(attack_samples, num_samples)
            # attack구간의 샘플수와 전체 샘플의 합중의 작은 값이 end_sample 지점
        envelope[current_sample:end_sample] = np.linspace(0, 1, end_sample - current_sample)
            # current_sample 인 0 지점부터 end_sample 까지에 
            # 0, 1 까지를 end_sample - 0 한 만큼의 샘플수로 나누어서 넣기
            # 아 0부터 1(최대값)까지 점점 커져야 하니까
        current_sample = end_sample
            # attack 만큼의 샘플 이후가 current_sample 이 됨
    
    # Decay phase
    if current_sample < attack_samples + decay_samples and current_sample < num_samples:
        # 현재의 샘플이 attack+decay 한것보다 작고, 전체 샘플 개수보다 작다면!
        start_sample = current_sample
        end_sample = min(attack_samples + decay_samples, num_samples)
            # 둘 중 작은것을 decay 마지막 지점으로 선택 (전체 샘플개수 범위 넘어가면 안되니까)
        envelope[start_sample:end_sample] = np.linspace(1, sustain, end_sample - start_sample)
            # 1부터 sustain 레벨까지를 decay 구간 샘플개수 만큼 나눠서 천천히 줄어듦
        current_sample = end_sample
    
    # Sustain phase
    if current_sample < gate_sample and current_sample < num_samples:
        envelope[current_sample:min(gate_sample, num_samples)] = sustain
            # attack+decay 구간 이후의 레벨은 그냥 다 같은 magnitude 
            # sustain 레벨로 집어넣음
        current_sample = min(gate_sample, num_samples)

    # Release phase
    if current_sample < num_samples:
        sustain_level = envelope[current_sample - 1] if current_sample > 0 else sustain 
            # if current_sample 이 0보다 크다면 (노트가 눌렸다면?. 당연히 큰거아닌가?)
            # 아니면 sustain 레벨 => 근데 걍 같은거 아닌가....#📍
        end_sample = min(current_sample + release_samples, num_samples)
        envelope[current_sample:end_sample] = np.linspace(sustain_level, 0, end_sample - current_sample)

        return envelope #envelope 이란 전체 샘플개수 만큼 들어있는 magnitude 배열 출력

class Synthesizer:
    """
    Complete Synthesizer
    
    Signal chain : OSC -> MIXER -> FILTER -> ENVELOPE -> OUTPUT

    Features:
        - 2 oscillators with detune (디튠)
        - Oscillator sync option (싱크 옵션)
        - Filter with envelope modulation (엔벨로프 변조)
        - ADSR envelope
        - Master volume
    """

    def __init__(self, sample_rate = 44100):  #📍이 __init__ 확실히 이해
        self.sample_rate = sample_rate

        # Oscillator parameters
        self.osc1_waveform = 'sawtooth'
        self.osc1_level = 0.5 # 📍 이것도 범위가 어떻게 되는거지

        self.osc2_waveform = 'sawtooth' # 우선 기본값이 둘다 saw 인건가?
        self.osc2_detune = 0.0  # cents
        self.osc2_level = 0.5

        self.sync_enabled = False

        # Filter parameters
        self.filter_cutoff = 2000 #Hz
        self.filter_resonance = 1.0
        self.filter_envelope_amount = 0.0 # 0 to 1

        # Envelope parameters (s 단위)
        self.env_attack = 0.01
        self.env_decay = 0.1
        self.env_sustain = 0.7
        self.env_release = 0.3

        # Master 
        self.master_volume = 0.8

    def generate_note(self, frequency, duration, gate_time=None):
        # Generate a single note
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # OSC 1
        if self.osc1_waveform == 'sawtooth':
            osc1, _ = polyblep_sawtooth(frequency, duration, self.sample_rate)
        elif self.osc1_waveform == 'sqaure':
            osc1, _ = polyblep_square(frequency, duration, self.sample_rate)
        elif self.osc1_waveform == 'sine':
            osc1- _ = np.sin(2 * np.pi * t * frequency)

        # OSC 2
        detune_ratio = 2 ** (self.osc2_detune / 1200) 
            #cents to ratio ? 📍어떤 비율인건지?
        freq2 = frequency * detune_ratio

        if self.osc2_waveform == 'sawtooth':
            osc2, _ = polyblep_sawtooth(freq2, duration, self.sample_rate)
        elif self.osc2_waveform == 'square':
            osc2, _ = polyblep_square(freq2, duration, self.sample_rate)
        elif self.osc2_waveform == 'sine':
            osc2 = np.sin(2 * np.pi * freq2 * t)

        # Mix oscillators 
        mixed = self.osc1_level * osc1 + self.osc2_level * osc2

        # Generate envelope 
        envelope = adsr_envelope(self.env_attack, self.env_decay,
                                 self.env_sustain, self.env_release,
                                 duration, self.sample_rate, gate_time)
        
        # Filter with envelope modulation
        if self.filter_envelope_amount > 0:
            # Time-varing filter (시간 가변 필터?📍)
            filtered = np.zeros_like(mixed)
            # osc 1+osc 2 의 샘플개수만큼의 0으로 찬 배열을 만들어서 filtered 라고 만들고
            chunk_size = 512
            num_chunks = len(mixed) // chunk_size
            # 📍왜 나눠서 계산하는가 : 하나하나의 샘플을 다 계산하면 오래걸린다 한것 때문에 그런가?

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size

                # Modulate cutoff with envelope
                env_value = np.mean(envelope[start:end])
                    # 📍왜 평균내지? 엔벨롭 magnitude 를 512 개의 지점내에서 평균내는 이유는?
                modulated_cutoff = self.filter_cutoff * (1 + self.filter_envelope_amount * env_value * 4)
                    # filter env_amount = 0 to 1 
                    # 📍 +1 * 4 이런거 왜 하는거야
                chunk_filtered = biquad_filter(mixed[start:end], 'lowpass',
                                               modulated_cutoff, self.filter_resonance,
                                               self.sample_rate)
                filtered[start:end] = chunk_filtered

        else:
            # filter env amount 없으면
            # static filter (고정필터)
            filtered = biquad_filter(mixed, 'lowpass', self.filter_cutoff,
                                     self.filter_resonance, self.sample_rate)
            
        # Apply envelope
        output = filtered * envelope

        # Master volume
        output *= self.master_volume

        # Prevent clipping 
        output = np.clip(output, -1.0, 1.0) # 📍np.clip?

        return output, t




    
