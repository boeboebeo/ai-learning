"""
==============================================
DAY 2: Harmonic Analysis & Fourier Series Deep Dive
==============================================
Goal: Fourier series를 이용해 파형을 직접 합성하고
      각 harmonic의 역할을 시각적으로 이해한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100
DURATION = 1.0
FREQUENCY = 220  # A3

def synthesize_from_harmonics(fundamental_freq, harmonics, amplitudes, duration, sample_rate):
    """
    Fourier series 를 이용한 additive synthesis

    Math : y(t) = Σ A_n * sin(2π * n * f * t * φ_n) -> Fourier Series를 사용하는 방법. Band-limited 버전 가능 (Aliasing 제거 가능)
        - y(t) : 출력파형
        - A_n : 진폭(n차 배음의 크기)
        - n : 배음 번호 (1차, 2차, 3차 ....)
        - f : 기본 주파수 (기음)
        - φ_n : 위상 . n차 배음의 시작위치 
        
            ex. 440Hz sine(A = 1.0) + 880Hz sine(A = 0.5),  φ₁=0, φ₂=0 이라면:
            y(t) = 1.0 * sin(2π * 440 * t)
                   + 0.5 * sin(2π * 880 * t)

            => Fourier 를 사용해서 계산하는 방법이 물리적 의미의 배음의 합과 같음. 

    Math(2) : y(t) = 2 * (t * f - floor(t * f + 0.5)) -> 기하학적인 floor 사용 방법 (수학적 트릭. 정규화 불필요(이미 -1 ~ +1))
    Math(3) -> Modulo 사용 방법 (수학적 트릭. 정규화 불필요(이미 -1 ~ +1)) => Aliasing 발생
        phase = (freq * t) % 1.0
        wave = 2 * phase - 1    

    Parameters : 
        - harmonics : list of harmonic numbers
        - amplitudes : corresponding amplitudes for each harmonic

    FOUNDATION of additive synthesis !
    """

    t = np.linspace(0, duration, int(sample_rate*duration), endpoint = False)
        #int(sample_rate*duration) 샘플 개수 자리, 정수여야 함. => int 변형 (duration이 0.5인 경우 float 형태 될 수 도 있음)
    result = np.zeros_like(t)   #t와 같은 크기의 0으로만 채워진 배열을 만듦.

    for harmonic, amplitude in zip(harmonics, amplitudes): #zip : 두 리스트 짝지어줌. Index 없이 그냥 앞 순서대로 두 개 짝지어서 표현하고 싶을때 사용
        result += amplitude * np.sin(2 * np.pi * harmonic * fundamental_freq * t) 
            # 한번의 루프 안에서 t 배열 전체가 한꺼번에 연산됨 (numpy 가 배열 전체를 한꺼번에 처리해서 편함)
            # 1. 2 * np.pi * harmonic * fundamental_freq * t
            #    = 2 * 3.14159 * 1 * 440 * [0, 0.0000227, 0.0000454, ..., 0.9999773] => 요기 이 배열이 t  
            #    = [0, 0.0627, 0.1254, ..., 2763.85]  # 44100개 한꺼번에!
        
        """
        t =  [0, 0.0000227, 0.0000454, ..., 0.9999773]  (44100개) 
        
        result = np.zeros_like(t) 이걸 통해서 똑같이 44100개의 0이 들어있는 배열 만듦
        result = [0, 0, 0, 0, 0, ..., 0]  (44100개, 모두 0)

        (1)첫번째 루프 (1차 배음이 harmonic과 amplitude 를 짝지어서 입력)
        result = [0, 0.0627, 0.1253, ..., -0.0627]

        (2)두번째 루프 (2차 배음)
        result = [0, 0.0627, 0.1253, ..., -0.0627] + [0, 0.0314, 0.0627, ..., -0.0314]
               = [0, 0.0941, 0.1880, ...] ..

            => 이렇게 전체 t(각각의 샘플) 만큼의 배열에서의 각각의 모든 값을 동시에 더할 수 있음!
            (배열 전체를 한번에 더할 수 있음)

        """

    return result, t 

def build_sawtooth_progressive():
    """
    Sawtooth progressive synthesis + visualization

    y(t) = (2/π) * Σ[(-1)^(n+1) * sin(2π * n * f * t) / n]
         = (2/π) * [sin(2πft) - sin(4πft)/2 + sin(6πft)/3 - ...]

         2/π 존재이유? 
    """

    fig, axes = plt.subplots(3, 2, figsize = (10, 8))
        # subplots(3, 2)의 결과로는 3행 2열의 2D 배열이 만들어짐 (3 x 2)
        # 2D -> 1D 로 평탄화.
    """ 2D -> 1D. flatten()

        ex. axes = [[ax00, ax01],
                    [ax10, ax11]]  # 2D

            => axes.flatten() 후 

            axes = [ax00, ax01, ax10, ax11]  # 1D

    2D 배열은 복잡하고 (axes[0, 1]이렇게 표현해야 함), for 루프도 복잡해짐. enumerate 도 가능해짐
       +for 루프도 복잡:
        for i in range(2):
            for j in range(2):
                axes[i, j].plot(...)

    """

    axes = axes.flatten() 
        # axes.flatten() : 2D -> 1D 

    num_harmonics_list = [1, 2, 3, 5, 10, 50] # 더할 하모닉스의 개수

    for idx, num_harmonics in enumerate(num_harmonics_list):
        harmonics = list(range(1, num_harmonics + 1))
            # sawtooth : amplitude = (2/π) * ((-1)^(n+1)) / n 
        amplitudes = [(2/np.pi) * ((-1)**(n+1)) / n for n in harmonics]

        wave, t = synthesize_from_harmonics(FREQUENCY, harmonics, amplitudes,
                                            0.01, SAMPLE_RATE) #10ms
        
        axes[idx].plot( t * 1000, wave, linewidth=1.5, color='blue') #ms 변환. 소수점이 너무많은거에서 소수점만 줄인것. wave 는 각 t 의 값 배열
        axes[idx].set_title(f'Sawtooth with {num_harmonics} harmonics')
            #하모닉스 개수에 따른 파형
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_ylim(-1.2, 1.2)
        axes[idx].grid(True, alpha=0.3)

        # Gibbs phenomenon 표기. (overshooting at discontinuities)
        if num_harmonics >= 10:
            axes[idx].text(0.02, 0.95, 'Note : Gibbs phenomenon (overshoot)',
                           transform=axes[idx].transAxes, fontsize =8,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
    axes[-1].set_xlabel('Time(ms)')
    axes[-2].set_xlabel('Time(ms)')
    plt.tight_layout()
    plt.show()

def build_square_progressive():
    """
    Square wave progressive synthesis + visualization 
    
    Square Fourier sereis (odd harmonics only):
    y(t) = (4/π) * Σ[sin(2π * (2k-1) * f * t) / (2k-1)]
         = (4/π) * ( sin(2πft) + sin(6πft)/3 + sin(10πft)/5 + ...)
    """
    
    fig, axes = plt.subplots(3, 2, figsize = (10, 8))
    axes = axes.flatten()

    num_harmonics_list = [1, 2, 3, 5, 10, 25]

    for idx, num_harmonics in enumerate(num_harmonics_list):
        harmonics = [2*k - 1 for k in range(1, num_harmonics + 1)] # 1, 3, 5, 7 ...
        amplitudes = [(4/np.pi) / h for h in harmonics]
            # 4/np.pi * 1/h -> 정규화 상수 및 진폭 나누기

        wave, t = synthesize_from_harmonics(FREQUENCY, harmonics, amplitudes,
                                            0.01, SAMPLE_RATE)
        
        axes[idx].plot(t * 1000, wave, linewidth=1.5, color='red')
        axes[idx].set_title(f'Square with {num_harmonics} odd harmonics')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_ylim(-1.5, 1.5)
        axes[idx].grid(True, alpha =0.3) # 그래프 내에 그리드 표기. alpha = 투명도 표기

    axes[-1].set_xlabel('Time(ms)') # python의 음수 인덱스는 [-1] : 마지막
    axes[-2].set_xlabel('Time(ms)') # [-2] : 마지막에서 두번째
        # axes[-1] 는 내 for 문에서의 마지막 값인 axes[5] : [-1] / axes[4] : [-2]
        # axes 의 idx 개수가 늘어날 수 도 있으므로, 저렇게 표기함

    plt.tight_layout()
    plt.show()

def harmonic_amplitude_comparison():
    
    harmonics = np.arange(1, 21)

    #saw
    saw_amps = 1 / harmonics

    #square
    square_harmonics = harmonics[::2] #두개의 간격으로
    square_amps = 1 / square_harmonics

    #triangle 
    tri_harmonics = harmonics[::2]
    tri_amps = 1 / (tri_harmonics**2)

    plt.figure(figsize = (10, 8))

    plt.subplot(1, 2, 1) #마지막은 index
    plt.stem(harmonics, saw_amps, linefmt='b-', markerfmt='bo',
             basefmt='gray', label='Sawtooth(1/n)')
    plt.stem(square_harmonics, square_amps, linefmt='r-', markerfmt='rs',
             basefmt='gray', label='Square(odd, 1/n)')
    plt.stem(tri_harmonics, tri_amps, linefmt='g-', markerfmt='g^',
             basefmt='gray', label='Triangle (odd, 1/n^2)')
    
    plt.xlabel('Harmonic Number')
    plt.ylabel('Relative Amplitude')
    plt.title('Harmonic Amplitude(Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)



    #log scale for better visualization

    plt.subplot(1, 2, 2)
    plt.stem(harmonics, saw_amps, linefmt='b-', markerfmt='bo',
             basefmt='gray', label='Sawtooth(1/n)')
    plt.stem(square_harmonics, square_amps, linefmt='r-', markerfmt='rs',
             basefmt='gray', label='Square(odd, 1/n)')
    plt.stem(tri_harmonics, tri_amps, linefmt='g-', markerfmt='g^',
             basefmt='gray', label='Triangle(odd, 1/n^2)')
    plt.xlabel('Harmonic Number')
    plt.ylabel('Relative Amplitude(log scale)')
    plt.title('Harmonic Amplitude(Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

    
    
# build_sawtooth_progressive()
# build_square_progressive()
# harmonic_amplitude_comparison()

def custom_timbre_design():
    # 각 harmonic의 amplitude 를 자유롭게 조절하기

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    #example 1 : Even hormonis only (opposite of square)
    harmonics_even = list(range(2, 1000, 2)) 
        # 이렇게 간격을 넣으면 짝수만 빠짐 (마지막 2 가 간격)
        # 근데 이렇게 하면 사실 2를 기준으로 정수배가 되어서, 2가 기본 주파수가 되고, 440Hz가 기음인 sawtooth 가 되어버림
    amps_even = [1/h for h in harmonics_even]
    wave1, t = synthesize_from_harmonics(FREQUENCY, harmonics_even, amps_even,
                                        0.01, SAMPLE_RATE)
    
    axes[0, 0].plot(t*1000, wave1, linewidth=1.5)
    axes[0, 0].set_title('Even harmonics only')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    print(harmonics_even)

    #example 2 : random
    harmonics_rand = list(range(1, 11))
    amps_rand = np.random.rand(10) / harmonics_rand #random but ..?
    wave2, t = synthesize_from_harmonics(FREQUENCY, harmonics_rand, amps_rand,
                                        0.01, SAMPLE_RATE)
    
    axes[0, 1].plot(t*1000, wave2, linewidth=1.5, color='purple')
    axes[0, 1].set_title('Random harmonic amplitudes (Unipue timbre)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    # example 3: Prime number harmonics only
    harmonics_prime = [2, 3, 5, 7, 11, 13]
    amps_prime = [1/h for h in harmonics_prime]
    wave3, t = synthesize_from_harmonics(FREQUENCY, harmonics_prime, amps_prime,
                                        0.01, SAMPLE_RATE)
    axes[1, 0].plot(t * 1000, wave3, linewidth=1.5, color='green')
    axes[1, 0].set_title('Prime Number Harmonics (bell-like)')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    #example 4: Harmonic amplitude envelope (shaped decay)
    harmonics_env = list(range(1, 21))
    #exponential decay instead of 1/n
    amps_env = [np.exp(-0.3 * h) for h in harmonics_env] #? 
    wave4, t = synthesize_from_harmonics(FREQUENCY, harmonics_env, amps_env,
                                        0.01, SAMPLE_RATE)
    axes[1, 1].plot(t * 1000, wave4, linewidth=1.5, color='orange')
    axes[1, 1].set_title('Exponential Harmonic Decay (warm timbre)')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    build_sawtooth_progressive()
    build_square_progressive()
    harmonic_amplitude_comparison()
    custom_timbre_design()


"""Gibbs phenomenon
: 불연속점(Discontinuity. 값이 순간적으로 점프하는 -1 -> +1 지점)
  근처에서 유한개 배음으로 근사하면 약 9% 튀어나오는 현상

 **overshooting 발생
    : 원래의 목표 진폭은 +1이었지만, 실제는 +1.09로 overshoot!
    : 부드러운 곡선(sine)으로 급격한 변화(수직 점프)를 만드려는게 근본적으로 불가능하기 때문에 일어나는
      Fourier Series 의 한계

      1. y=sin(x) . 연속적. 부드러운 곡선이기 때문에 무한히 미분 가능
      2. square 는 불연속적. 미분 불가능(non-differentiable).
          => 부드러운 것들의 합이 급격한 것이 될수는 없음 -> ringing 

          but, 순수 아날로그 square 는 직접 전압을 스위칭 하기 때문에 gibbs 는 없음
               + 현실 아날로그는 회로 대역폭 제한때문에 slew rate 의 제한이 있어서 약간 둥그러짐
               => 디지털에서만 Fourier series 로 파형생성. gibbs 불가피
                  +wavetable 또한 미리 계산된 파형을 사용하지만, gibbs 포함된 채로 저장됨
               => Additive synthesis 는 실시간으로 sine 을 더해서 gibbs 발생.

++이론적 Square (무한 배음):
square(t) = (4/π) * [sin(ωt)/1 + sin(3ωt)/3 + sin(5ωt)/5 + ...]
            무한 배음 → 완벽한 사각파

++유한 배음 (N개만):
square_N(t) = (4/π) * [sin(ωt)/1 + ... + sin((2N-1)ωt)/(2N-1)]
++            유한 배음 → Gibbs 현상 발생!




"""

