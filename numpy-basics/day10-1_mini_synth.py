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
            #두 보정값이 겹치는 샘플은 없음 -> 서로 방해 안함
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

    if filter_type == 'lowpass':  #📍각 계수에 대한 부분들 아직 모호 => 걍 recipe
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

    b = np.array([b0, b1, b2]) / a0 # =[b0/a0, b1/a0, b2/a0] 
        # x에 곱하는 계수들 : b배열
    a = np.array([1, a1/a0, a2/a0]) 
        # 분자 계수들 
        # y에 곱하는 계수들 : a배열

    filtered = signal.lfilter(b, a, signal_input)
    # signal.lfilter(b, a, signal_input) 이 함수가 내부적으로 아래의 차분방적식 실행함
    # y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2]- a1·y[n-1] - a2·y[n-2]
    # 이걸 실행하려면 b계수랑 a계수들을 따로 묶어줘야 함
    


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
            # attack구간의 샘플수 vs 전체 샘플 개수중 의 작은 값이 end_sample 지점
            # duration 보다 전체 샘플 수가 길면 안되니까 
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
            # if current_sample > 0 : 배열 인덱스가 -1이 되는것을 방지
            # 만약 current_sample = 0 이면, envelope[current_sample - 1] = envelope[-1]이 됨
            # 파이썬에서의 -1인덱스는 배열 맨 마지막을 의미 
            # 위의 상황은 attack, decay 가 0인 상황일때를 대비하는 것 
            # 위에서 attack, decay = 0 & gate_time = 0 이면 아예 실행이 안됨

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

    def __init__(self, sample_rate = 44100):  
        #📍 __init__ : 신스를 여러개 만들어도, 각자 독립적인 설정을 가질 수 있는게 self.xxx 덕분
        self.sample_rate = sample_rate

        # Oscillator parameters
        self.osc1_waveform = 'sawtooth'
        self.osc1_level = 0.5 #범위 : 0 ~ 1 사이

        self.osc2_waveform = 'sawtooth' 
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
        self.master_volume = 0.6

    def generate_note(self, frequency, duration, gate_time=None):
        # Generate a single note
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # OSC 1
        if self.osc1_waveform == 'sawtooth':
            osc1, _ = polyblep_sawtooth(frequency, duration, self.sample_rate)
        elif self.osc1_waveform == 'square':
            osc1, _ = polyblep_square(frequency, duration, self.sample_rate)
        elif self.osc1_waveform == 'sine':
            osc1 = np.sin(2 * np.pi * t * frequency)
            # 얘는 뭐 반환안하고 여기서 바로 만드는거기때문에 걍 osc, _ = 이거 없어도 됨

        # OSC 2
        detune_ratio = 2 ** (self.osc2_detune / 1200) 
            # cents to ratio : 전체 한 옥타브중에서 몇 cents 올리는지 
            # ex. detune = 100 이면 반음위  
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
            # Time-varing filter : cutoff 가 시간에 따라 변하는 필터
            filtered = np.zeros_like(mixed)
            # osc 1+osc 2 의 샘플개수만큼의 0으로 찬 배열을 만들어서 filtered 라고 만들고
            chunk_size = 512
            num_chunks = len(mixed) // chunk_size
            # 왜 512 씩 나눠서 계산하는가
            # 컷오프를 샘플 하나씩 다 바꾸면 1초에 44100번 biquad_filter 를 호출해야함 -> 너무 느려짐
            # 512 개 안에서는 거의 컷오프 안바뀌니까 한번에 묶어서 처리!
            # 그냥 평균값으로 대표값을 사용함 

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size

                # Modulate cutoff with envelope
                env_value = np.mean(envelope[start:end])
                    # 📍왜 평균내지? 엔벨롭 magnitude 를 512 개의 지점내에서 평균내는 이유는?
                    # 그냥 청크묶음의 엔벨롭의 평균값을 걍 대표값으로 사용하기 위함 
                modulated_cutoff = self.filter_cutoff * (1 + self.filter_envelope_amount * env_value * 4)
                    # filter env_amount = 0 to 1 
                    # 1 + : 일 더하는 이유는 원래 컷오프에 0을 곱해버리면 원래 있던 컷오프도 없어지니까
                    # * env_value 
                    # * 4 : 4를 곱하는 이유는 엔벨롭이 컷오프를 최대 4배까지 올릴 수 있도록 스케일을 키운것. 4가 없으면 변화가 거의 안느껴짐  
                    # 근데 이러면 가청주파수 이상으로도 필터가 올라가 질 수 도 있음
                    # 안전 장치 필요! (아래가 안전장치임)
                modulated_cutoff = np.clip(modulated_cutoff, 20, self.sample_rate / 2 - 100)
                    # 적어도 샘플레이트의 절반보다 -100Hz 한 주파수 까지만 컷오프가 올라가게끔 
                    # np.clip 은 np.clip(값, 최솟값, 최댓값) : 이렇게 어떤 값이던(진폭 or 주파수 or index or Q 제한 Etc..) 제한하는 기능
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
        output = np.clip(output, -1.0, 1.0) 
            # 📍np.clip : -1.0 ~ +1.0 사이로 강제로 잘라냄 (clipping)

        return output, t

def demo_synthesizer_presets():
    duration = 2.0
    frequency = 220 

    # Preset 1: Classic analog lead
    synth1 = Synthesizer()
    synth1.osc1_waveform = 'sawtooth'
    synth1.osc2_waveform = 'sawtooth'
    synth1.osc2_detune = 7
    synth1.osc1_level = 0.5 
    synth1.osc2_level = 0.5
    synth1.filter_cutoff = 1500
    synth1.filter_resonance = 7.0
    synth1.filter_envelope_amount = 0.8
    synth1.env_attack = 0.01
    synth1.env_decay = 0.4 
    synth1.env_sustain = 0.0
    synth1.env_release = 0.5

    lead_sound, t = synth1.generate_note(frequency, duration, gate_time = 1.5)
    #직접 연주를 하는게 아니니까 게이트 타임을 넣어주는거구나?

    # Preset 2: Bass (베이스)
    synth2 = Synthesizer()
    synth2.osc1_waveform = 'square'
    synth2.osc2_waveform = 'sawtooth'
    synth2.osc2_detune = -1200  # one octave down (한 옥타브 아래)
    synth2.osc1_level = 0.6
    synth2.osc2_level = 0.4
    synth2.filter_cutoff = 500
    synth2.filter_resonance = 1.0
    synth2.filter_envelope_amount = 0.5
    synth2.env_attack = 0.001
    synth2.env_decay = 0.1
    synth2.env_sustain = 0.1
    synth2.env_release = 0.3
    
    bass_sound, _ = synth2.generate_note(frequency/2, duration, gate_time=0.5)
    
    # Preset 3: Pad (패드)
    synth3 = Synthesizer()
    synth3.osc1_waveform = 'sawtooth'
    synth3.osc2_waveform = 'sawtooth'
    synth3.osc2_detune = 3
    synth3.filter_cutoff = 2500
    synth3.filter_resonance = 0.7
    synth3.filter_envelope_amount = 0.2
    synth3.env_attack = 0.8
    synth3.env_decay = 0.5
    synth3.env_sustain = 0.7
    synth3.env_release = 1.5
    synth3.master_volume = 0.6
    
    pad_sound, _ = synth3.generate_note(frequency, duration)

    # Visualize 
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    
    sounds = [
        ("Analog Lead", lead_sound),
        ("Bass", bass_sound),
        ("Pad", pad_sound)
    ]

    for idx, (name, sound) in enumerate(sounds):
        # Time domain
        plot_samples = int(0.1 * SAMPLE_RATE)
        t_plot = t[:plot_samples] * 1000

        axes[idx, 0].plot(t_plot, sound[:plot_samples], linewidth=1)
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].set_title(f'{name} - Waveform')
        axes[idx, 0].grid(True, alpha = 0.3)

        # Spectrum
        N = len(sound)
        fft_result = fft(sound)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N

        axes[idx, 1].plot(positive_freqs, magnitude, linewidth=0.5)
        axes[idx, 1].set_xlim(20, 8000)
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_title(f'{name} - Spectrum')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xscale('log')
        axes[idx, 1].set_yscale('log')

    axes[-1, 0].set_xlabel('Time(ms)')

    # Play sounds. 소리 들어보는 코드
    import sounddevice as sd

    print("Playing Lead...") 
    sd.play(lead_sound.astype(np.float32), SAMPLE_RATE)
    sd.wait()

    print("Playing Bass...")
    sd.play(bass_sound.astype(np.float32), SAMPLE_RATE)
    sd.wait()

    print("Playing Pad...")
    sd.play(pad_sound.astype(np.float32), SAMPLE_RATE)
    sd.wait()

    print("Done.")

    plt.tight_layout()
    plt.show()


def demo_melody():
    """
    간단한 멜로디 생성
    
    Note: C major scale (C 메이저 스케일)
    """
    # Note frequencies (노트 주파수)
    notes = {
        'C4': 261.63,
        'D4': 293.66,
        'E4': 329.63,
        'F4': 349.23,
        'G4': 392.00,
        'A4': 440.00,
        'B4': 493.88,
        'C5': 523.25
    }

    # Simple melody (간단한 멜로디)
    # (Note_name, duration)
    melody = [
        ('C4', 0.5), ('E4', 0.5), ('G4', 0.5), ('C5', 0.5),
        ('B4', 0.5), ('G4', 0.5), ('E4', 0.5), ('C4', 0.5),
        ('F4', 0.5), ('A4', 0.5), ('C5', 0.5), ('A4', 0.5),
        ('G4', 1.0), ('C4', 1.0)
    ]

    # Create synth (신스 생성)
    synth = Synthesizer()
    synth.osc1_waveform = 'sawtooth'
    synth.osc2_waveform = 'square'
    synth.osc2_detune = 5
    synth.filter_cutoff = 2000
    synth.filter_resonance = 2.0
    synth.filter_envelope_amount = 0.6
    synth.env_attack = 0.01
    synth.env_decay = 0.15
    synth.env_sustain = 0.4
    synth.env_release = 0.2
    
    # Generate melody (멜로디 생성)
    melody_audio = []

    for note_name, note_duration in melody:
        freq = notes[note_name]
        sound, _ = synth.generate_note(freq, note_duration, gate_time=note_duration * 0.9)
        melody_audio.append(sound)

    # Concatenate (연결)
    full_melody = np.concatenate(melody_audio)

    # save 자리!
    import os
    filename = os.path.join(os.path.dirname(__file__), "day10_melody.wav")
        # 간단하게 지금 파일 위치에 저장하는 법
    melody_normalized = full_melody / np.max(np.abs(full_melody)) * 0.9
    melody_int16 = (melody_normalized * 32767).astype(np.int16)
    wavfile.write(filename, SAMPLE_RATE, melody_int16)
    print(f"✓ Melody saved: {filename}")

    # Visualize (시각화)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    t_full = np.arange(len(full_melody)) / SAMPLE_RATE

    # Waveform
    axes[0].plot(t_full, full_melody, linewidth=0.5, color='blue')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Generated Melody - Waveform')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    from scipy import signal as sp_signal
    f, t_spec, Sxx = sp_signal.spectrogram(full_melody, SAMPLE_RATE,
                                           nperseg=2048, noverlap=1536)

    im = axes[1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                           shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Generated Melody - Spectrogram')
    axes[1].set_ylim(0, 2000)
    plt.colorbar(im, ax=axes[1], label='Power (dB)')

    plt.tight_layout()
    plt.show()

def complete_synthesis_pipeline():
    """
    전체 synthesis 파이프라인 시각화
    """
    synth = Synthesizer()
    synth.osc1_waveform = 'sawtooth'
    synth.osc2_waveform = 'sawtooth'
    synth.osc2_detune = 7
    synth.filter_cutoff = 1200
    synth.filter_resonance = 4.0
    synth.filter_envelope_amount = 0.7
    synth.env_attack = 0.02
    synth.env_decay = 0.3
    synth.env_sustain = 0.4
    synth.env_release = 0.5

    frequency = 220
    duration = 2.0
    gate_time = 1.5

    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Generate intermediate stages (중간 단계 생성)
    osc1, _ = polyblep_sawtooth(frequency, duration, SAMPLE_RATE)
    osc2, _ = polyblep_sawtooth(frequency * 2**(7/1200), duration, SAMPLE_RATE)
    
    mixed = 0.5 * osc1 + 0.5 * osc2

    envelope = adsr_envelope(synth.env_attack, synth.env_decay,
                             synth.env_sustain, synth.env_release,
                             duration, SAMPLE_RATE, gate_time)
    
    # Filter (static for visualization)
    filtered = biquad_filter(mixed, 'lowpass', synth.filter_cutoff,
                             synth.filter_resonance, SAMPLE_RATE)

    final = filtered * envelope

    # Visualize pipeline 

    fig, axes = plt.subplots(5, 2, figsize=(12, 8))
    
    stages = [
        ("OSC 1 (Sawtooth)", osc1),
        ("OSC 2 (Sawtooth +7 cents)", osc2),
        ("Mixed Oscillators", mixed),
        ("After Filter", filtered),
        ("Final Output (with Envelope)", final)
    ]
    
    plot_samples = int(0.05 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000

    for idx, (name, stage_signal) in enumerate(stages):
        # Time domain
        axes[idx, 0].plot(t_plot, stage_signal[:plot_samples], linewidth=1)
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].set_title(f'{name} - Time Domain')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Spectrum
        N = len(stage_signal)
        fft_result = fft(stage_signal)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N
        
        axes[idx, 1].plot(positive_freqs, magnitude, linewidth=0.5)
        axes[idx, 1].set_xlim(20, 5000)
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_title(f'{name} - Spectrum')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xscale('log')
        axes[idx, 1].set_yscale('log')
        
        # Add arrow for flow (흐름 화살표)
        if idx < len(stages) - 1:
            axes[idx, 0].text(0.5, -0.3, '↓', transform=axes[idx, 0].transAxes,
                            fontsize=20, ha='center', color='red')
    
    axes[-1, 0].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()



demo_synthesizer_presets()
# demo_melody()
# complete_synthesis_pipeline()
