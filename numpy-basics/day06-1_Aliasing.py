"""
==============================================
DAY 6: Aliasing & Anti-aliasing Techniques
==============================================
Goal: Aliasing의 원인을 이해하고 다양한 anti-aliasing 기법을 학습한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.0

def demonstrate_aliasing():
    """
    Aliasing-Shannon theorem:
        - sample rate must be > 2 * highest freq 
        - nyquist freq = sample_rate / 2

    when signal contains frequencies > nyquist : 
        - they "fold back into audible range 
        - creates unwanted frequencies 
        - sounds harsh 

    Example : 
        - sample rate = 44100Hz -> nyquist freq = 22050Hz
        - If signal has 23000Hz component 
            -> aliases to 21100Hz (44100 -> 23000)

    """

    #Create high-freq sine wave(above nyquist freq)
    freq_high = 25000

    t = np.linspace(0, 0.001, int(SAMPLE_RATE * 0.01), endpoint=False)

    #True signal(theoretical)
    signal_true = np.sin(2 * np.pi * freq_high * t)

    #sampled version (what we actually get)
    #this will alias to: |sample_rate - freq_high| = |44100 - 25000| = 19100Hz

    aliased_freq = abs(SAMPLE_RATE - freq_high)
    signal_aliased = np.sin(2 * np.pi * aliased_freq * t)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    #Time domain: true signal 
    axes[0].plot(t * 1000, signal_true, 'o-', linewidth=1, markersize=3, color='blue')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'True signal: {freq_high}Hz (above Nyquist)')
    axes[0].grid(True, alpha=0.3)    

    #Time domain: what we actually sample
    axes[1].plot(t * 1000, signal_aliased, 'o-', linewidth=1, markersize=3, color='red')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Aliased Signal: {aliased_freq} Hz (folded back)')
    axes[1].grid(True, alpha=0.3)    
    axes[1].text(0.7, 0.9, 'Same sample points, different frequency',
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    

    #Both overlaid 
    axes[2].plot(t * 1000, signal_true, 'b-', linewidth=1, alpha=1.0, label=f'True: {freq_high} Hz')
    axes[2].plot(t * 1000, signal_aliased, 'r--', linewidth=2, label=f'Aliased: {aliased_freq} Hz')
    axes[2].plot(t * 1000, signal_true, 'ko', markersize=4, label='Sample points')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Aliasing: Cannot distinguish from samples alone!')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# demonstrate_aliasing()


def naive_sawtooth_aliasing():
    """
    Naive sawtooth aliasing 문제
    
    Sawtooth contains ALL harmonics:
        - 1f, 2f, 3f, 4f ... infinitely..
        - Many harmonics -> nyquist frequency 초과
        - result : heavy aliasing 
    
    """

    freq = 440
    t = np.linspace(0, DURATION, int(SAMPLE_RATE*DURATION), endpoint=False)

    #Naive sawtooth (Day1 method)
    phase = (freq * t) % 1.0
    saw_naive = 2 * phase - 1

    # FFT analysis
    N = len(saw_naive)
    fft_result = fft(saw_naive)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    magnitude = np.abs(fft_result[:N//2] * 2 / N)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    #time domain
    plot_samples = int(0.01 * SAMPLE_RATE)
    axes[0].plot(t[:plot_samples] * 1000, saw_naive[:plot_samples], linewidth=1.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Naive Sawtooth: {freq} Hz (time domain looks OK)')
    axes[0].grid(True, alpha=0.3)    

    #Frequency domain (reveals the problem)
    axes[1].plot(positive_freqs, magnitude, linewidth=0.5, color='blue')
    axes[1].set_xlim(0, SAMPLE_RATE/2 + 1000)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Naive Sawtooth Spectrum (aliasing visible!)')
    axes[1].axvline(SAMPLE_RATE/2, color='red', linestyle='--', linewidth=2,
                    label='nyquist frequency')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    #ver1. Mark theoretical harmonics
    # for n in range(1, 51):
    #     harmonic_freq = freq * n
    #     if harmonic_freq < SAMPLE_RATE/2:
    #         axes[1].axvline(harmonic_freq, color='green', alpha=0.2, linewidth=0.5)
    #     else:
    #         #This harmonic aliasese
    #         aliased_freq = abs(SAMPLE_RATE - (harmonic_freq % SAMPLE_RATE))
    #         axes[1].axvline(aliased_freq, color='red', alpha=0.3, linewidth=0.5)

    """abs(SAMPLE_RATE - (harmonic_freq % SAMPLE_RATE))
            
            1) harmonic_freq 를 SAMPLE_RATE 로 나눈 나머지를 구함 
                - ex. fs = 44100, harmonic_freq = 26400(440 * 60한 60번째 freq)
                    => 26400 > fs/2 : alising 발생
                    => 26400 % 44100 = 26400

            2) SAMPLE_RATE - (...)
                - 0~fs 범위에서 반사
                - ex. 44100 - 26400 = 17700Hz <- aliased freq
            """
    
    #ver2. 
    for n in range(1, 100):
        harmonic_freq = freq * n
        if harmonic_freq < SAMPLE_RATE/2:
            # 정상 harmonic
            axes[1].axvline(harmonic_freq, color='green', alpha=0.2, linewidth=0.5)
        else:
            # Aliasing 계산
            folded = harmonic_freq % SAMPLE_RATE
            if folded <= SAMPLE_RATE / 2:
                aliased_freq = folded
            else:
                aliased_freq = SAMPLE_RATE - folded
            
            axes[1].axvline(aliased_freq, color='red', alpha=0.3, linewidth=0.5)


    axes[1].text(0.6, 0.9, 'Red lines = aliased harmonics\n(foled back from above Nyquist freq)',
                transform=axes[1].transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.6))
    
    plt.tight_layout()
    plt.show()

# naive_sawtooth_aliasing()

def oversampling_antialiasing():
    """
    Oversampling anti-aliasing 

    method :
    1. generate signal at higher sample rate 
    2. apply low-pass filter
    3. downsample to target rate

    Oversampling factor(배수):
    - 2x, 4x, 8x, etc.
    - Higher = less aliasing but more CPU 
    
    """

    freq = 440
    oversample_factor = 4

    #generate at higher rate
    high_sr = SAMPLE_RATE * oversample_factor
    t_high = np.linspace(0, DURATION, int(high_sr * DURATION), endpoinst=False)

    #naive sawtooth at high sample rate
    phase_high = (freq * t_high) % 1.0
    
