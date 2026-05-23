"""
==============================================
DAY 7: Band-Limited Synthesis (BLIT & PolyBLEP)
==============================================
Goal: 실시간으로 사용 가능한 band-limited waveform 생성 기법을 학습한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.0


def blit_impulse_train(freq, duration, sample_rate):
    """
    BLIT : Band-limited Impulse Train (대역 제한 임펄스 열)
        => 단순 파형 생성이 아니라,
           : aliasing 이 없는 saw/square 를 만들기 위함

    **BLIT 가 필요한 이유
    : 디지털에서 saw/square는 갑작스러운 점프(discontinuity)를 일으키므로 문제를 일으킴
      -> 갑작스러운 점프는 수학적으로, 무한한 harmonics 를 필요로 함
      -> but, 디지털에서는 nyquist 이상의 주파수는 표현이 불가하다.
        => aliasing 발생
        => so, 처음부터 nyquist 이하의 harmonics 만 가진 impulse train을 만듦!

    ** why impulse train? **
    : saw wave 의 미분은 수학적으로 => impulse train 
    => impulse train을 leaky 적분하면 saw wave
    (but, 원래의 Impulse는 모든 주파수 포함. BLIT는 Nyquist 이하의 harmonics만 포함->aliasing 없음)

    concept:
    - generate band-limited impulses(대역 제한된 임펄스)
    - integrate to create band-limited waveforms (적분해서 대역 제한 파형 생성)

    Mathematical formula (sinc function):
    BLIT(t) = sin(πMt) / (M * sin(πt))
        - harmonics 를 잘라놓은 Fourier series

    where M = floor(sample_rate / (2 * freq)) 
    (M : harmonics below nyquist)

    Properties:
    - no aliasing!
    - efficient (and additive 보다 빠름. 
        additive는 harmonics 하나씩 sin 계산, BLIT는 수학식 하나로 전체 harmonic 처리)
    - produces sharp discontinuities (날카로운 불연속점) => harmonics 를 제한했지만 여전히 매우 sharp함

    **일반 naive saw vs BLIT 방식
    1)naive saw
        : 바로 점프 -> aliasing
    
    2)BLIT 방식
        : bnad-limited impulse 생성 -> 적분 -> alias-free saw 생성

    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Calculate M (number of harmonics below Nyquist)
    nyquist = sample_rate / 2
    M = int(nyquist / freq)

    # Phase for impulse train
    phase = (freq * t) % 1.0
        # 컴퓨터는 연속시간이 아니고 t 위에서의 이산 time 샘플로 존재하기 때문에, 현재 샘플이 주기 안에서 어디쯤인지를 알기위해서 phase 를 도출해내야 함
        # freq * t = 몇 주기를 돌았는지? 
        # phase 를 0~1 범위로 반복하게 하기 위해서 % 1.0 처리함

    # BLIT formula : sin(πMϕ) / Msin(πϕ)
    # 여기서 sin(πϕ) 이 0이 되는 순간 존재 
    # ϕ = 0,1,2,3,... 일때 sin(0) = 0 이 발생하여 division by zero 발생 가능
    # 근데 극한값은 finite 해서 괜찮지만, 컴퓨터는 극한계산안하고 0/0을 하기 때문에 epsilon을 사용

    # Avoid division by zero : epsilon(아주작은값 = 0 으로 나누는것을 방지하는 안전장치)
        # = 1e-10 = 0.0000000001.
        # 만약 denominator 가 0일 경우, 에러날 수 있으므로, epsilon을 더해서 처리함
        # phase = 0, 1, 2, 3, 4 ..(정수)인 경우 np.sin(np.pi * phase) = 0 이기 때문에 나눗셈 불가능
            # => 그래서 M * np.sin(np.pi * phase + epsilon) 해서 아주 살짝 틀어지게 함
    epsilon = 1e-10
    denominator = M * np.sin(np.pi * phase + epsilon)
        #분모
    numerator = np.sin(np.pi * M * phase)
        #분자

    blit = numerator / (denominator + epsilon)

    # Normalize (정규화)
    blit = blit / M

    return blit, t, M


def blit_to_sawtooth(blit_signal, sample_rate):
    """
    Convert BLIT to sawtooth via integration (적분)
    
    Integration = cumulative sum (누적합)
    Leaky integrator (누설 적분기) to prevent DC buildup
    """
    # Leaky integrator coefficient (DC 차단 계수)
    leak = 0.999
        # leak : 0.999
        # 매 샘플 0.1% 감소하게 함 -> DC 누적 방지, 발산방지, 안정적인 sawtooth. 안정적 적분
        # 순수 적분 = 발산 / Leaky 적분  = 안정적 sawtooth
 
    saw = np.zeros_like(blit_signal)
    accumulator = 0
    
    for i in range(len(blit_signal)):
        accumulator = leak * accumulator + blit_signal[i]
        saw[i] = accumulator
    
    # Normalize to [-1, 1]
    saw = saw / np.max(np.abs(saw))
    saw = 2 * saw - 1  # center around 0
    
    return saw

def polyblep_residual(t, dt):
    # Residual : discontinuity에서 튀어나오는 aliasing 성분만 따로 떼어낸 보정 조각

    """
    PolyBLEP: Polynomial Band-Limited Step (다항식 대역 제한 계단)
        => 불연속점(계단)을 부드럽게 만들어서 aliasing 을 줄이는 기법
    
    Concept:
    - Add correction polynomial at discontinuities (불연속점에 보정 다항식 추가)
    - Very efficient (매우 효율적)
    - Good quality (좋은 품질)
    
    Polynomial residual (잔차 다항식):
    - Smooths out the discontinuity (불연속점 부드럽게)
    - 2nd order polynomial (2차 다항식)
    
    Parameters:
    - t: phase distance from discontinuity (불연속점으로부터 위상 거리)
    - dt: phase increment per sample (샘플당 위상 증가량)
    """
    if t < dt:
        # Rising edge (상승 에지)
        # discontinuity 바로 직후인지 파악후 => jump 근처만 수정
        t = t / dt
        return t + t - t * t - 1.0
            # 짧은 smoothing curve

    elif t > 1.0 - dt:
        # Falling edge (하강 에지)
        # 주기 끝 Discontinuity 근처. 1 -> -1 점프로 correction 필요함
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    
    else:
        return 0.0
            # 그냥 여기는 그대로 naive saw 사용

    #polyBLEP : 점프 직전/직후에만 아주 짧은 anti-aliasing patch 붙이는 것 (tiny smoothing correction)
    

def polyblep_sawtooth(freq, duration, sample_rate):
    """
    PolyBLEP sawtooth generation
    
    Algorithm:
    1. Generate naive sawtooth (기본 톱니파 생성)
    2. Detect discontinuities (불연속점 감지)
    3. Apply PolyBLEP correction (보정 적용)
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    dt = freq / sample_rate  # phase increment (위상 증가량)
    phase = 0
    
    output = np.zeros_like(t)

    for i in range(len(t)):
        # Naive sawtooth
        naive_saw = 2 * phase - 1
        
        # PolyBLEP correction at discontinuity
        correction = polyblep_residual(phase, dt)
        
        output[i] = naive_saw - correction
        
        # Advance phase (위상 진행)
        phase += dt
        if phase >= 1.0:
            phase -= 1.0

    return output, t


def polyblep_square(freq, duration, sample_rate):
    """
    PolyBLEP square wave
    
    Square has TWO discontinuities per cycle (사이클당 2개 불연속점):
    - Rising edge at phase = 0
    - Falling edge at phase = 0.5
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    dt = freq / sample_rate
    phase = 0
    
    output = np.zeros_like(t)
    
    for i in range(len(t)):
        # Naive square
        naive_square = 1.0 if phase < 0.5 else -1.0
        
        # PolyBLEP correction at both edges
        correction = polyblep_residual(phase, dt)  # Rising edge
        correction += polyblep_residual(phase - 0.5, dt)  # Falling edge
        
        output[i] = naive_square - correction
        
        phase += dt
        if phase >= 1.0:
            phase -= 1.0
    
    return output, t

def compare_blit_polyblep_naive():
    """
    BLIT vs PolyBLEP vs Naive 비교
    """
    freq = 440
    
    # Generate signals
    blit, t, M = blit_impulse_train(freq, DURATION, SAMPLE_RATE)
    saw_blit = blit_to_sawtooth(blit, SAMPLE_RATE)
    
    saw_polyblep, _ = polyblep_sawtooth(freq, DURATION, SAMPLE_RATE)
    
    # Naive for reference
    phase = (freq * t) % 1.0
    saw_naive = 2 * phase - 1
    
    # FFT analysis
    N = len(saw_blit)
    fft_blit = fft(saw_blit)
    fft_poly = fft(saw_polyblep)
    fft_naive = fft(saw_naive)
    
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_blit = np.abs(fft_blit[:N//2]) * 2 / N
    mag_poly = np.abs(fft_poly[:N//2]) * 2 / N
    mag_naive = np.abs(fft_naive[:N//2]) * 2 / N
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Time domain
    plot_samples = int(0.01 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000
    
    axes[0, 0].plot(t_plot, saw_naive[:plot_samples], linewidth=1.5, color='red')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Naive Sawtooth')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_plot, saw_blit[:plot_samples], linewidth=1.5, color='blue')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'BLIT Sawtooth (M={M} harmonics)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(t_plot, saw_polyblep[:plot_samples], linewidth=1.5, color='green')
    axes[0, 2].set_ylabel('Amplitude')
    axes[0, 2].set_title('PolyBLEP Sawtooth')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Frequency domain (full spectrum)
    axes[1, 0].plot(positive_freqs, mag_naive, linewidth=0.5, color='red', alpha=0.7)
    axes[1, 0].set_xlim(0, SAMPLE_RATE / 2)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Naive Spectrum (aliasing!)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    axes[1, 0].axvline(SAMPLE_RATE / 2, color='black', linestyle='--', alpha=0.5)
    
    axes[1, 1].plot(positive_freqs, mag_blit, linewidth=0.5, color='blue', alpha=0.7)
    axes[1, 1].set_xlim(0, SAMPLE_RATE / 2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('BLIT Spectrum (clean!)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    axes[1, 1].axvline(SAMPLE_RATE / 2, color='black', linestyle='--', alpha=0.5)
    
    axes[1, 2].plot(positive_freqs, mag_poly, linewidth=0.5, color='green', alpha=0.7)
    axes[1, 2].set_xlim(0, SAMPLE_RATE / 2)
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('Magnitude')
    axes[1, 2].set_title('PolyBLEP Spectrum (clean!)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    axes[1, 2].axvline(SAMPLE_RATE / 2, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def polyblep_all_waveforms():
    """
    PolyBLEP으로 모든 기본 파형 생성
    """
    freq = 220
    
    saw_poly, t = polyblep_sawtooth(freq, DURATION, SAMPLE_RATE)
    square_poly, _ = polyblep_square(freq, DURATION, SAMPLE_RATE)
    
    # Triangle from square (integrate square)
    triangle_poly = np.zeros_like(square_poly)
    leak = 0.999
    accumulator = 0
    
    for i in range(len(square_poly)):
        accumulator = leak * accumulator + square_poly[i]
        triangle_poly[i] = accumulator
    
    triangle_poly = triangle_poly / np.max(np.abs(triangle_poly))
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    
    waveforms = [
        ("PolyBLEP Sawtooth", saw_poly, 'blue'),
        ("PolyBLEP Square", square_poly, 'red'),
        ("PolyBLEP Triangle", triangle_poly, 'green'),
    ]
    
    for idx, (name, wave, color) in enumerate(waveforms):
        # Time domain
        plot_samples = int(0.02 * SAMPLE_RATE)
        t_plot = t[:plot_samples] * 1000
        
        axes[idx, 0].plot(t_plot, wave[:plot_samples], linewidth=1.5, color=color)
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].set_title(f'{name} - Time Domain')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Frequency domain
        N = len(wave)
        fft_result = fft(wave)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N
        
        axes[idx, 1].plot(positive_freqs, magnitude, linewidth=0.5, color=color)
        axes[idx, 1].set_xlim(0, 10000)
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_title(f'{name} - Spectrum')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def blit_vs_additive_efficiency():
    """
    BLIT vs Additive synthesis 효율성 비교
    
    Computational cost (계산 비용):
    - Additive: O(M) per sample (M harmonics)
    - BLIT: O(1) per sample + integration
    - PolyBLEP: O(1) per sample (가장 빠름!)
    """
    freq = 440
    nyquist = SAMPLE_RATE / 2
    M = int(nyquist / freq)
    
    print(f"\nEfficiency comparison for {freq} Hz:")
    print(f"Number of harmonics needed: {M}")
    print(f"\nAdditive synthesis:")
    print(f"  - {M} sine evaluations per sample")
    print(f"  - Total ops: {M} * num_samples")
    print(f"\nBLIT:")
    print(f"  - 1 BLIT evaluation + 1 integration per sample")
    print(f"  - Total ops: 2 * num_samples")
    print(f"\nPolyBLEP:")
    print(f"  - 1 waveform + polynomial correction per sample")
    print(f"  - Total ops: ~3 * num_samples")
    print(f"\nSpeedup: BLIT/PolyBLEP is ~{M/2}x faster than additive!")
    
    # Visual comparison of methods
    methods = ['Additive', 'BLIT', 'PolyBLEP']
    relative_cost = [M, 2, 3]  # relative computational cost
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, relative_cost, color=['red', 'blue', 'green'], alpha=0.7)
    plt.ylabel('Relative Computational Cost (ops per sample)')
    plt.title(f'Efficiency Comparison for {freq} Hz Sawtooth\n({M} harmonics needed)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, cost in zip(bars, relative_cost):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def minblep_concept():
    """
    MinBLEP: Minimum-phase Band-Limited Step (최소위상 대역제한 계단)
    
    Concept:
    - Pre-computed lookup table (미리 계산된 룩업 테이블)
    - Insert residual at discontinuities (불연속점에 잔차 삽입)
    - Higher quality than PolyBLEP (PolyBLEP보다 높은 품질)
    - Used in professional software (전문 소프트웨어에서 사용)
    
    Steps:
    1. Generate ideal step response (이상적 계단 응답 생성)
    2. Window and truncate (창함수 적용 및 절단)
    3. Store in table (테이블에 저장)
    4. Use at runtime (런타임에 사용)
    
    This is conceptual - full implementation requires more code
    """
    # Generate ideal bandlimited step (sinc function)
    table_size = 64
    oversampling = 16
    
    t = np.linspace(-4, 4, table_size * oversampling)
    
    # Sinc function (ideal low-pass filter impulse response)
    sinc = np.sinc(t)
    
    # Integrate to get step (계단으로 적분)
    step = np.cumsum(sinc) / np.sum(sinc)
    
    # Window to reduce artifacts (아티팩트 감소용 창함수)
    window = np.hanning(len(step))
    minblep_table = step * window
    
    # Normalize
    minblep_table = minblep_table / np.max(np.abs(minblep_table))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # MinBLEP table
    axes[0].plot(t, minblep_table, linewidth=2, color='purple')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('MinBLEP Residual Table (pre-computed)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].text(0.5, 0.9, 'This table is inserted at each discontinuity\n(like PolyBLEP but more accurate)',
                transform=axes[0].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Comparison of correction methods
    t_compare = np.linspace(-2, 2, 200)
    
    # PolyBLEP approximation (simplified)
    polyblep_approx = np.zeros_like(t_compare)
    for i, tc in enumerate(t_compare):
        if abs(tc) < 1:
            polyblep_approx[i] = tc - tc**2 * np.sign(tc)
    
    # Ideal sinc
    ideal = np.cumsum(np.sinc(t_compare))
    ideal = ideal / np.max(ideal)
    
    axes[1].plot(t_compare, ideal, linewidth=2, color='blue', label='Ideal (MinBLEP-like)', alpha=0.7)
    axes[1].plot(t_compare, polyblep_approx, linewidth=2, color='green', 
                linestyle='--', label='PolyBLEP (polynomial approximation)')
    axes[1].set_xlabel('Time (samples)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Correction Function Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_blit_polyblep_naive()
polyblep_all_waveforms()
blit_vs_additive_efficiency()
minblep_concept()


"""
    1) Impulse : 한 점에서만 값이 있고 나머지는 0인 신호 + 모든 주파수를 동시에 포함
        ex. impulse = [0, 0, 0, 1, 0, 0, 0] (한 부분만 1, 나머지는 0)

        # Impulse 생성
        impulse = np.zeros(100)
        impulse[50] = 1  # 50번째 위치에만 1

    2) Impulse train(임펄스 열) : 일정한 간격으로 반복되는 임펄스 열
        ex. 주기 t 마다 임펄스 발생 
        ex. impulse train의 FFT = 또 다른 impulse train
    
    """

""" Naive vs PolyBLEP vs BLIT vs Additive 

1)

2)

3)

4)

"""

"""BLIT formula 

: sin(πMϕ) / Msin(πϕ)

 1) 주기적으로 반복되는 Impulse를 만들고 싶음
 2) 완벽한 impulse는 무한 harmonic을 필요로 함 => nyquist 이하의 Harmonic 만 사용한 impulse를 만들어야 겠다
 3) 이상적인 Impulse train : 1 + 2cos(x) + 2cos(2x) + 2cos(3x) + ... 이렇게 모든 harmonic 을 다 더한것
    => 1f + 2f + 3f + 4f + ...
 4) but, BLIT 는 Harmonic 을 제한함 
    => so, 1 + 2cos(x) + 2cos(2x) + ... + 2cos((M - 1)x) 이 유한 harmonic 합을 수학적으로 정리하면

    sin(πMϕ) / Msin(πϕ) 이 공식이 나옴


    - 분모의 M : 이 더 커지면, 더 많은 Harmonics 가 있다는 뜻이므로 더 좁은 impulse 가 생성됨
    - 분모의 sin(πϕ) : 반복 구조 생성 (ϕ=0,1,2,...)
    - M으로 나눠야 하는 이유 : normalization . harmonic을 많이 더하면 amplitude 가 커지니까 
        => 크기 보정


"""

"""사전 지식

필수:
- Sinc 함수 모양
- Low-pass filter 개념
- 푸리에 변환 기본 (주파수↔시간)

선택:
- 복소수 (더 깊은 이해)
- 푸리에 급수 (수학적 증명)
"""