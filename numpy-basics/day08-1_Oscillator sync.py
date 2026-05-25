"""
==============================================
DAY 8: Oscillator Sync (Hard Sync & Soft Sync)
==============================================
Goal: Hard sync와 soft sync의 원리를 이해하고 spectrum 변화를 분석한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100
DURATION = 1.0

def hard_sync(master_freq, slave_freq, duration, sample_rate):
    """
    Hard Sync

    Concept : 
    - Master oscillator (마스터 오실레이터) sets the fundamental frequency (기본 주파수 설정)
    - Slave oscillator (슬레이브 오실레이터) is reset when master resets (마스터가 리셋될때 같이 리셋)
    - Creates harmonically rich sound (조화적으로 풍부한 소리)

    Mathematical description:
    - Master phase: φ_m(t) = 2π * f_m * t
    - Slave phase: φ s(t) = 2π * f_s * t
    - When φ_m crosses 0 → φ_s resets to 0 (위상 강제 리셋)
    
    Result:
    - Pitch = master frequency (피치는 마스터 주파수)
    - Timbre (음색) controlled by slave frequency
    - Slave > Master → formant-like peaks (포먼트 같은 피크)    
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    master_phase = 0
    slave_phase = 0
        #초기화 : Phase 는 0으로 시작

    master_increment = master_freq / sample_rate
    slave_increment = slave_freq / sample_rate
        # increment : 증가
        # ex. master_freq = 441Hz 면 441/44100 = 1/100 이고, 
        # 그러면 이게 의미하는 바는 한 샘플당 Phase 증가량이 0.01ms 라는 뜻
        # 100샘플에 거쳐 0 -> 1 로 가야하므로 매 샘플마다의 phase 증가량 임

        # sample 0   : phase = 0.000
        # sample 10  : phase = 0.100
        # sample 50  : phase = 0.500
        # sample 99  : phase = 0.990 .. 
        # sample 100 : phase = 1.000 (wraps to 0)

    output = np.zeros_like(t)
    master_out = np.zeros_like(t)
        # t만큼의 0 이 들어있는 배열 우선 만듦

    for i in range(len(t)):
        # Generate master (for visualization)
        master_out[i] = 2 * master_phase - 1 #sawtooth
            # 0번째 값에 첫번째 샘플값 들어가고 

        # Generate slave (sawtooth)
        output[i] = 2 * slave_phase - 1


        # Advance phases
        master_phase += master_increment
        slave_phase += slave_increment
            # phase increment만큼 증가 한 값이 master_phase 값이 됨
            # 그럼 한번 턴 돌면 441Hz 라면 master_phase 값은 현재 0.01

        # Master reset
        if master_phase >= 1.0:
            master_phase -= 1.0
            #master phase 값이 1.0과 같거나 넘었다면 거기서 다시 1.0 빼서 0~1사이로 만들고, slave는 0점으로 리셋시킴
            #Sync : reset slave when master resets
            slave_phase = 0.0

        # Slave wraparound (but will be reset by master)
        if slave_phase >= 1.0:
            slave_phase -= 1.0
            # master 의 Sync가 없었다면 일어날 Wrap

    return output, master_out, t

def soft_sync(master_freq, slave_freq, duration, sample_rate):
    """
    soft sync 
    
    Concept:
    - Instead of hard reset, reverse slave direction (리셋 대신 방향 반전)
    - Smoother than hard sync (하드 싱크보다 부드러움)
    - Different harmonic structure (다른 배음 구조)
    
    Implementation:
    - When master resets → slave direction inverts (마스터 리셋시 슬레이브 방향 반전)
    - Creates triangle-like modulation (삼각파 같은 변조)
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint = False)

    #여기는 위상처리 하는 것
    master_phase = 0
    slave_phase = 0
    slave_direction = 1 # +1 or -1

    master_increment = master_freq / sample_rate
    slave_increment = slave_freq / sample_rate

    output = np.zeros_like(t)
    master_out = np.zeros_like(t)

    #여기는 진폭처리 하는 것 
    for i in range(len(t)): 
        master_out[i] = 2 * master_phase - 1
        output[i] = 2 * slave_phase - 1
        #master, slave 진폭 설정
        #amplitude = 2 * phase - 1 (phase -> amplitude 변환)
        # phase 는 0 ~ 1 까지의 경계가 있음
        # amplitude 는 -1 ~ +1 까지의 경계

        #Advance master
        master_phase += master_increment

        #Advance slave(with direction)
        slave_phase += slave_direction * slave_increment
            # slave direction이 바뀌면 원래 증가하던게 감소하게 되는 반대의 진폭으로 커지니까

        # Master reset
        if master_phase >= 1.0:
            master_phase -= 1.0
            #master 의 한 주기가 끝나면 거기서 1.0을 빼고, slave의 방향성을 반대로 바꿈
            #soft sync : reverse slave direction (방향 반전)
            slave_direction *= -1

        # Keep slave in bounds (slave 파형의 위쪽 경계를 만난다면)
        # Slave 의 경계처리 (0 ~ 1 범위 유지하게 함)
        if slave_phase >= 1.0:
            #slave 의 한 주기가 끝나면 1.0에서 slave phase 에서 1.0을 뺀걸 또 빼서 반대로 움직이기 시작하게 함
            slave_phase = 1.0 - (slave_phase - 1.0)
            slave_direction = -1

        elif slave_phase < 0: #아래쪽 경계
            #음수를 양수로 바꿔서 다시 증가하기 시작하게 해야함
            slave_phase = -slave_phase
            slave_direction = 1

    return output, master_out, t

"""soft sync example

ex.1 (slave 정상 증가중)
    slave_phase = 0.7
    slave_direction = 1 (증가중. 음수라면 감소하고 있는것)
    slave_increment = 0.1

    # Advance:
        slave_phase += slave_direction * slave_increment 
        slave_phase = 0.7 + 1 * 0.1 = 0.8
        => 계속 증가

ex.2 (master reset. soft sync 발동)
    master_phase = 1.05   (한주기 넘음)
    slave_phase = 0.6
    slave_direction = 1   (증가중)

    #master reset :
        if master_phase >= 1.0:
        master_phase -= 1.0 # 0.05
        slave_direction *= -1 이니까 => 이제 -1 (방향 반전)

    #다음 스텝에서:
        slave_phase += slave_direction * slave_increment
        slave_phase = 0.6 + (-1) * 0.1 = 0.5  (이제 감소하기 시작함)

ex.3 (slave 가 1.0 넘어감 -> 위쪽 경계)
    slave_phase = 1.05
    slave_direction = 1  (증가중이었음)

    #bounce back : 다시 진폭감소하기 시작해야함 (맨 위 꼭대기 쳤으니까)
    if slave_phase >= 1.0:
        overshoot = slave_phase - 1.0 # 0.05

        #튕겨서 되돌림
        slave_phase = 1.0 - overshoot # 0.95
            => 여기 부분이 위 식에서 
                'slave_phase = 1.0 - (slave_phase - 1.0)'  
                => 이렇게 한 부분!
        
        #방향 반전
        slave_direction = -1

ex.4 (slave 가 0 아래로 감. 아래쪽 경계)
    slave_phase = -0.03 (음수)
    slave_direction = -1 (감소중이었음)

    #Bounce back:
    elif slave_phase < 0:
        
        #음수를 양수로
        slave_phase = -slave_phase #0.03
        
        #방향 반전
        slave_direction = 1

"""

def compare_hard_soft_sync():
    """
    Hard sync vs Soft sync 
    """
    master_freq = 110
    slave_freq = 280 # 3x master 

    hard_out, master_hard, t = hard_sync(master_freq, slave_freq, DURATION, SAMPLE_RATE)
    soft_out, master_soft, t = soft_sync(master_freq, slave_freq, DURATION, SAMPLE_RATE)

    # No sync (for reference)
    slave_nosync = np.sin(2 * np.pi * slave_freq * t)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    # Time domain
    plot_samples = int(0.05 * SAMPLE_RATE) #50ms
    t_plot = t[:plot_samples] * 1000

    # Master
    axes[0, 0].plot(t_plot, master_hard[:plot_samples], linewidth=1.5, color='blue')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Master Oscillator: {master_freq} Hz (sets pitch)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Slave no sync
    axes[0, 1].plot(t_plot, slave_nosync[:plot_samples], linewidth=1.5, color='gray')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'Slave No Sync: {slave_freq} Hz')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hard sync
    axes[1, 0].plot(t_plot, hard_out[:plot_samples], linewidth=1.5, color='red')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Hard Sync Output (abrupt resets)')
    axes[1, 0].grid(True, alpha=0.3)

    # Mark sync points
    master_period = SAMPLE_RATE / master_freq
    for n in range(int(len(t_plot) / (master_period / 1000)) +1):
        sync_time = n * (1000/master_freq)
        if sync_time < t_plot[-1]:
            axes[1, 0].axvline(sync_time, color='blue', linestyle='--', alpha=0.3)

    # Soft sync
    axes[1, 1].plot(t_plot, soft_out[:plot_samples], linewidth=1.5, color='green')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Soft Sync Output (direction reversal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Spectra
    N = len(hard_out)
    fft_hard = fft(hard_out)
    fft_soft = fft(soft_out)
    fft_nosync = fft(slave_nosync)
    
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_hard = np.abs(fft_hard[:N//2]) * 2 / N
    mag_soft = np.abs(fft_soft[:N//2]) * 2 / N
    mag_nosync = np.abs(fft_nosync[:N//2]) * 2 / N
    
    # Hard sync spectrum
    axes[2, 0].plot(positive_freqs, mag_hard, linewidth=0.5, color='red')
    axes[2, 0].set_xlim(0, 3000)
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Magnitude')
    axes[2, 0].set_title('Hard Sync Spectrum (rich harmonics at master freq)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_yscale('log')
    
    # Mark master harmonics
    for n in range(1, 30):
        harmonic = master_freq * n
        if harmonic < 3000:
            axes[2, 0].axvline(harmonic, color='blue', alpha=0.2, linestyle=':')
    
    # Soft sync spectrum
    axes[2, 1].plot(positive_freqs, mag_soft, linewidth=0.5, color='green')
    axes[2, 1].set_xlim(0, 3000)
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].set_title('Soft Sync Spectrum (smoother harmonics)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def sync_sweep_classic_sound():
    """
    Classic sync sweep (클래식 싱크 스윕)
    
    Technique:
    - Master freq: fixed (고정)
    - Slave freq: swept from low to high (낮은 주파수에서 높은 주파수로 스윕)
    - Creates formant-like resonances (포먼트 같은 공명)
    
    Famous in:
    - 1980s synth leads (80년대 신스 리드)
    - Trance supersaws (트랜스 슈퍼쏘)
    """
    master_freq = 220  # Fixed master

    # sweep slave from 1x to 8x master freq
    sweep_duration = 2.0
    t = np.linspace(0, sweep_duration, int(SAMPLE_RATE*sweep_duration), endpoint=False)

    # Exponential sweep (linear in log space)
    slave_freq_start = master_freq
    slave_freq_end = master_freq * 8

    slave_freq_curve = slave_freq_start * (slave_freq_end / slave_freq_start) ** (t / sweep_duration)
    # Why Exponential?
    # 주파수는 음악적으로 배수관계 -> 옥타브마다 주파수가 2배씩 증가하고, 그것이 지수적인 형태임(exponential)
    """
    step 1. 비율계산
        ratio = slave_freq_end / slave_freq_start
              = 1760 / 220 = 8
              (옥타브로 몇 옥타브 위인지 계산)

    step 2. 진행도 계산 (0~1)
        progress = t / sweep_duration
        # t = 0초 : 0
        # t = 1초 : 0.5
        # t = 2초 : 1.0

    step 3. Exponential 
        multiplier = ratio ** progress
                   = 8 ** progress

    step 4. 최종 주파수
        freq = slave_freq_start * multiplier
             = 220 * (8 ** progress)

        
        1)Linear sweep : 수학적으로 간단하지만, 주파수가 균일하게 증가한다는건 귀에는 비균일하게 증가하게 들림
        2)Exponential sweep : 수학적으로 복잡하지만, 옥타브가 균일하게 증가하므로, 귀에 균일하게 증가하게 들림

    """
    
    # Generate hard sync with time-varying slave freq
    master_phase = 0
    slave_phase = 0

    master_increment = master_freq / SAMPLE_RATE

    output = np.zeros_like(t)

    for i in range(len(t)):
        slave_increment = slave_freq_curve[i] / SAMPLE_RATE

        output[i] = 2 * slave_phase - 1

        master_phase += master_increment
        slave_phase += slave_increment

        if master_phase >= 1.0:
            master_phase -= 1.0
            slave_phase = 0.0  #hard sync

        if slave_phase >= 1.0:
            slave_phase -= 1.0

    #Create spectrogram 
    from scipy import signal as sp_signal
    
    f, t_spec, Sxx = sp_signal.spectrogram(output, SAMPLE_RATE, 
                                           nperseg=2048, 
                                           noverlap=1536)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Slave frequency curve
    axes[0].plot(t, slave_freq_curve, linewidth=2, color='blue')
    axes[0].set_ylabel('Slave Frequency (Hz)')
    axes[0].set_title('Sync Sweep: Slave Frequency Modulation')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(master_freq, color='red', linestyle='--', label=f'Master: {master_freq} Hz')
    axes[0].legend()
    
    # Waveform
    plot_start = int(0.5 * SAMPLE_RATE)  # Start at 0.5s
    plot_end = int(0.6 * SAMPLE_RATE)   # 100ms window
    t_plot = t[plot_start:plot_end] * 1000
    
    axes[1].plot(t_plot, output[plot_start:plot_end], linewidth=1, color='purple')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Waveform Snapshot (at 0.5s)')
    axes[1].grid(True, alpha=0.3)
    
    # Spectrogram
    im = axes[2].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                           shading='gouraud', cmap='viridis')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Spectrogram: Sync Sweep (formant-like resonances)')
    axes[2].set_ylim(0, 3000)
    plt.colorbar(im, ax=axes[2], label='Power (dB)')
    
    # Mark master harmonics
    for n in range(1, 15):
        axes[2].axhline(master_freq * n, color='red', alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def sync_ratio_exploration():
    #exploration : 탐험
    """
    Different master:slave ratios (다양한 마스터:슬레이브 비율)
    
    Ratios (비율):
    - 1:1 → no sync effect (싱크 효과 없음)
    - 1:2 → octave relationship (옥타브 관계)
    - 1:3, 1:5 → musical intervals (음정 간격)
    - 1:2.5 → inharmonic (비조화적)
    """
    master_freq = 100

    ratios = [
        (1, 1, "1:1 (no effect)"),
        (1, 2, "1:2 (octave)"),
        (1, 3, "1:3 (fifth + octave)"),
        (1, 5, "1:5 (major third + 2 octaves)"),
        (1, 2.5, "1:2.5 (inharmonic)"),
    ]

    fig, axes = plt.subplots(len(ratios), 2, figsize=(12, 8))

    for idx, (master_ratio, slave_ratio, label) in enumerate(ratios):
        slave_freq = master_freq * (slave_ratio / master_ratio)

        hard_out, _, t = hard_sync(master_freq, slave_freq, DURATION, SAMPLE_RATE)

        # Time domain
        plot_samples = int(0.03 * SAMPLE_RATE)
        t_plot = t[:plot_samples] * 1000

        axes[idx, 0].plot(t_plot, hard_out[:plot_samples], linewidth=1.5)
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].set_title(f'{label} - Time Domain')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Spectrum
        N = len(hard_out)
        fft_result = fft(hard_out)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N
        
        axes[idx, 1].plot(positive_freqs, magnitude, linewidth=0.5)
        axes[idx, 1].set_xlim(0, 2000)
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_title(f'{label} - Spectrum')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_yscale('log')
        
        # Mark master harmonics
        for n in range(1, 20):
            harmonic = master_freq * n
            if harmonic < 2000:
                axes[idx, 1].axvline(harmonic, color='red', alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    plt.show()

def reversible_sync():
    """
    Reversible Sync (가역 싱크)
    
    Variant:
    - Slave can sync master OR master can sync slave
    - Creates asymmetric timbres (비대칭 음색)
    - Experimental technique (실험적 기법)
    """
    freq_a = 110
    freq_b = 165  # 1.5x ratio
    
    # A syncs B (normal)
    output_ab, _, t = hard_sync(freq_a, freq_b, DURATION, SAMPLE_RATE)
    
    # B syncs A (reversed)
    output_ba, _, _ = hard_sync(freq_b, freq_a, DURATION, SAMPLE_RATE)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    plot_samples = int(0.03 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000
    
    # A syncs B
    axes[0, 0].plot(t_plot, output_ab[:plot_samples], linewidth=1.5, color='blue')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'A ({freq_a}Hz) syncs B ({freq_b}Hz)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # B syncs A
    axes[0, 1].plot(t_plot, output_ba[:plot_samples], linewidth=1.5, color='red')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'B ({freq_b}Hz) syncs A ({freq_a}Hz)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectra
    N = len(output_ab)
    fft_ab = fft(output_ab)
    fft_ba = fft(output_ba)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_ab = np.abs(fft_ab[:N//2]) * 2 / N
    mag_ba = np.abs(fft_ba[:N//2]) * 2 / N
    
    axes[1, 0].plot(positive_freqs, mag_ab, linewidth=0.5, color='blue')
    axes[1, 0].set_xlim(0, 2000)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Spectrum: A syncs B (fundamental = A)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].plot(positive_freqs, mag_ba, linewidth=0.5, color='red')
    axes[1, 1].set_xlim(0, 2000)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Spectrum: B syncs A (fundamental = B)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()



compare_hard_soft_sync()
sync_sweep_classic_sound()
sync_ratio_exploration()
reversible_sync()