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
    oversample_factor = 501 # x4 oversampling 

    #generate at higher rate
    high_sr = SAMPLE_RATE * oversample_factor 
    t_high = np.linspace(0, DURATION, int(high_sr * DURATION), endpoint=False)

    #naive sawtooth at high sample rate
    phase_high = (freq * t_high) % 1.0
        # 전체 시간 배열(t_high) 에 freq(440Hz)를 곱함
        # = 440 * [0, 0.00001, 0.00002, ...]
        # = [0, 0.0044, 0.0088, 0.0132, ...]. 각 시간마다 몇 사이클이 지났는지 
        # ex. 0.001초면 440 * 0.001 = 0.44cycle, 0.5초면 220 cycle, 1초면 440 cycle
    saw_high = 2 * phase_high - 1 #bi polar로 만들기

    # Low-pass filter (nyquist of target rate)
    # cutoff at original nyquist (SAMPLE_RATE / 2)
    nyquist_high = high_sr / 2
    cutoff = SAMPLE_RATE / 2
    normalized_cutoff = cutoff / nyquist_high
        # normalized_cutoff = (새로운 nyquist) / (기존 nyquist)
        # normalized_cutoff 는 0~1 사이 값으로 변환한 cutoff


    # Design low-pass filter
    # Using FIR filter (Finite Impulse Response)
    numtaps = 101 # filter order (필터 차수)
    lpf = signal.firwin(numtaps, normalized_cutoff, window='blackman')
        # signal.firwin()이 반환하는 것 : 길이 101인 필터 계수 배열 -> filter_coeff
        # 각 입력 샘플에 곱할 가중치(필터 계수). 중앙이 가장 크고 양쪽이 대칭
        # Convolution 으로 필터링 수행

        #Window : hamming, blackman, kaiser

    #Apply filter
    saw_filtered = signal.lfilter(lpf, 1.0, saw_high)
        #lpf : filter coefficient
        #1.0 : 
        #saw_high : oversamling 한 sawtooth

    #Downsample
    saw_downsampled = saw_filtered[::oversample_factor]
        #다시 오버샘플링 한 만큼(x4)의 간격의 것들만 고르기 
    
    #compare with naive version
    t_normal = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    phase_normal = (freq * t_normal) % 1.0
    saw_naive = 2 * phase_normal - 1

    #FFT comparison
    N = len(saw_downsampled)
    fft_over = fft(saw_downsampled)
    fft_naive = fft(saw_naive)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_over = np.abs(fft_over[:N//2] * 2 /N)
    mag_naive = np.abs(fft_naive[:N//2] * 2 /N)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Time domain comparison
    plot_samples = int(0.01 * SAMPLE_RATE)
    t_plot = t_normal[:plot_samples] * 1000

    axes[0, 0].plot(t_plot, saw_naive[:plot_samples], linewidth=1.5, color='red')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Naive Sawtooth')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_plot, saw_downsampled[:plot_samples], linewidth=1.5, color='blue')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'Oversampled {oversample_factor}x + Filtered')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Frequency domain
    axes[1, 0].plot(positive_freqs, mag_naive, linewidth=0.5, color='red', alpha=0.7)
    axes[1, 0].set_xlim(10000, SAMPLE_RATE / 2)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Naive Spectrum (high-freq region)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].plot(positive_freqs, mag_over, linewidth=0.5, color='blue', alpha=0.7)
    axes[1, 1].set_xlim(10000, SAMPLE_RATE / 2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Oversampled Spectrum (cleaner!)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    axes[1, 1].text(0.5, 0.9, 'Much less aliasing!',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    plt.tight_layout()
    plt.show()


def additive_synthesis_antialiasing():
    """
    Additive synthesis = inherently alias-free!

    Method : 
    - synthesize only harmonics below Nyquist (나이퀴스트 이하만 생성)
    - No aliasing possible !

    Drawback : 
    - Computationally expensive for many harmonics (많은 배음 시 CPU 많이 씀)
    - O(n) complexity per harmonic (배음당 O(n)시간복잡도)
    """

    freq = 440

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint = False)

    # Calculate max harmonic (Nyquist limit)
    nyquist = SAMPLE_RATE / 2
    max_harmonic = int(nyquist / freq)

    print(f"Generating {max_harmonic} harmonics (up to{freq * max_harmonic:.0f}Hz)")

    # Additive synthesis : sum all harmonics below nyquist
    saw_additive = np.zeros_like(t)

    for n in range(1, max_harmonic + 1): #max harmonic 까지 만들어야 하니까
        amplitude = (2 / np.pi) * ((-1)** (n + 1)) / n 
        saw_additive += amplitude * np.sin(2 * np.pi * n * freq * t)
            # saw amplitude => 반대부호가 필요한 이유
            # : 급격한 점프(saw의 꼭대기)를 만드려면 무한히 높은 주파수가 필요한데 서로 반대로 진동시켜서 상쇄+보강을 만들기 위함
        # print(saw_additive)
            # 각각의 amplitude 값을 점점 더해서 결과가 나오게 되는데, 
            # [ 0.          0.03988316  0.07960964 ... -0.11902335 -0.07960964 -0.03988316]
            # [ 0.00000000e+00  7.83439135e-05  6.24905792e-04 ... -2.09870862e-03  -6.24905792e-04 -7.83439135e-05]
            # [ 0.          0.03975279  0.07857467 ... -0.11557484 -0.07857467   -0.03975279]
            # 이런식으로 점점 더해감. 각 t에 해당하는 부분들이 점점 더해짐! 
            # 앞에부터 하나하나 채워가는거 아님 !

    # Compare with naive
    phase = (freq * t) % 1.0
    saw_naive = 2 * phase - 1

    # FFT 
    N = len(saw_additive)
    fft_add = fft(saw_additive)
    fft_naive = fft(saw_naive)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_add = np.abs(fft_add[:N//2]) * 2 / N
    mag_naive = np.abs(fft_naive[:N//2]) * 2 / N

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    #Time domain 
    plot_samples = int(0.005 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000


    axes[0, 0].plot(t_plot, saw_naive[:plot_samples], linewidth=1.5, color='red')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Naive Sawtooth')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_plot, saw_additive[:plot_samples], linewidth=1.5, color='green')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'Additive Synthesis ({max_harmonic} harmonics)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.5, 0.1, 'Gibbs phenomenon at edges\n(cannot be avoided with finite harmonics)',
                   transform=axes[0, 1].transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Full spectrum
    axes[1, 0].plot(positive_freqs, mag_add, linewidth=0.5, color='green', label='Additive')
    axes[1, 0].plot(positive_freqs, mag_naive, linewidth=0.5, color='red', alpha=0.5, label='Naive')
    axes[1, 0].set_xlim(0, (SAMPLE_RATE / 2) +400)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Full Spectrum Comparison')
    axes[1, 0].axvline(nyquist, color='blue', linestyle='--', label='Nyquist')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # High-frequency region (zoomed)
    # naive 는 무한 주파수를 시도하기 때문에, 그냥 spectrum 이 aliasing noise 로 꽉참
    axes[1, 1].plot(positive_freqs, mag_add, linewidth=0.5, color='green', label='Additive')
    axes[1, 1].plot(positive_freqs, mag_naive, linewidth=0.5, color='red', alpha=0.5, label='Naive')
    axes[1, 1].set_xlim(15000, SAMPLE_RATE / 2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('High-Freq Region (no aliasing in additive!)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()
    

def frequency_dependent_harmonic_limit():
    """
    Frequency-dependent harmonic limiting (주파수 의존적 배음 제한)
    
    Key insight:
    - Low notes can have many harmonics (낮은 음은 많은 배음 가능)
    - High notes need fewer harmonics (높은 음은 적은 배음만 필요)
    
    Example:
    - A1 (55 Hz): can have ~400 harmonics before Nyquist
    - A7 (3520 Hz): only ~6 harmonics before Nyquist
    """








"""signal.firwin()
    : FIR (Finite Impulse Response) 필터 설계 함수

    signal.firwin(numtaps, cutoff, ...)
        1. numtaps : 필터 탭 개수
            - 필터의 길이 (샘플 개수)
            - 홀수여야 함 (대칭성)
            - 클수록 : 더 정확하지만 더 많은 계산이 필요
        2. cutoff : normalized (0~1)
            - butter()와 동일한 방식

    **FIR vs IIR 
        1. FIR(firwin) : 
            - finite impulse response
            - 선형 위상 (위상 왜곡 없음)
            - 계산량 많음 
        2. IIR(butter) :
            - 효율적 (적은 탭으로 가파른 필터)
            - 위상 왜곡 있음
            - 불안정할 수도 있음
    """

"""
    1) 3-tap 평균 필터 (가장 간단한 low pass filter)
        coefficients = [0.333, 0.333, 0.333]

        입력신호 : [10, 20, 30, 40, 50]

        출력계산 : 
        output[0] = 0.333 * 10 + 0.333 * 20 + 0.333 * 30 = 20
        output[1] = 0.333 * 20 + 0.333 * 30 + 0.333 * 40 = 30
        output[2] = 0.333 * 30 + 0.333 * 40 + 0.333 * 50 = 40
        ...
            -> 필터링 = 주변 샘플들의 가중 평균

    2) # 5-tap 필터 설계
        coefficients = signal.firwin(5, 0.3)
        print(coeff)
            # 출력: [0.05, 0.25, 0.40, 0.25, 0.05] 
                -> 중앙이 가장 큼. 좌우 대칭. 
                -> 합이 1.0 (신호 크기 보존!)

        입력: ... [A] [B] [C] [D] [E] ...

        출력계산 : 
        output[0] = [A] * 0.05 + [B] * 0.25 + [c] * 0.40 + [d] * 0.25 + [e] * 0.05
        ourput[1] = [B] * 0.05 + [c] * 0.25 + [d] * 0.40 + [e] * 0.25 + [f] * 0.05  
        ...

    3) example 

    계수:     [0.05] [0.25] [0.40] [0.25] [0.05]
                ↓      ↓      ↓      ↓      ↓
    입력 위치:   [A]    [B]    [C]    [D]    [E]
                              ↑
                          현재 출력 위치

    슬라이딩 윈도우처럼 한 칸씩 이동:

        Step 0: [A] [B] [C] [D] [E] → output[0]
        Step 1:     [B] [C] [D] [E] [F] → output[1]
        Step 2:         [C] [D] [E] [F] [G] → output[2]
        ...

    """

""" numtaps 와 transition band 의 연관성
    
    1) 상황별 numtaps
        - Anti aliasing(다운 샘플링 전. 높은 품질이 필요함)
            : numtaps = 101 ~ 201
        - 실시간 오디오 처리(지연 최소화. 처리가 빨라야 함)
            : numtaps = 31 ~ 51
        - 오프라인 고품질 처리
            : numtaps = 201 ~ 501
    
    """