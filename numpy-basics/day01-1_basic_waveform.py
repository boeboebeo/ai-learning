"""
==============================================
DAY 1: Basic Waveform Generation & Phase Accumulator
==============================================
Goal: sine, saw, square, triangle 파형을 수학적으로 생성하고
      phase accumulator의 개념을 이해한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq  #scipy vs scipy.fft 의 차이가 뭐지?
from scipy import signal

#Audio parameters

# ====== 설정 (전역 상수) ======
SAMPLE_RATE = 44100  #Hz. 대문자 변수 : 전역 상수 (constant). = python 관례. 이 값을 바꾸지 말라는 뜻
DURATION = 1.0  #seconds
FREQUENCY = 440  #A4 note 

# 시간 배열 생성 : t = np.linspace(0, 1, 44100)
# phase 계산 : phase = 2 * np.pi * freq * t
# 파형 생성 : wave = np.sin(phase)   <- 이 phase 를 파형으로 변환


def generate_sine(freq, duration, sample_rate):
    """ Sine wave generation using phase accumulator
    math : y(t) = A * sin(2π * f * t)
        - most fundamental waveform (only 1st harmonic)
        - Pure tone, no overtones 
        - A : amplitude(기본값 1.0) , f = freq , t = time(밑의 np.linespace로 출력한 배열을 사용)
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = 2 * np.pi * freq * t   
        #phase accumulator . np.pi = 3.14159... => 2π = 2*np.pi(360도 = 2π 라디안. 삼각함수는 라디안 기준)
    
    return np.sin(phase), t  
        #phase변수가 위에서 처리한 2pi * f * t 의 배열이기 때문에 여기서는 그 배열 전체에 sin을 적용함 


def generate_sawtooth(freq, duration, sample_rate):
    """Sawtooth wave - naive implementation
    math : y(t) = 2 * (t * f - floor(t * f + 0.5)) . floor(x) : 소수점 아래 버리는 함수 
        - contains All harmonics (1, 2, 3, 4, 5...)
        - Harmonic amplitude = 1/n 
        - Brightest sound among basic waveform 

    Warning : this will cause aliasing ! (day 6 내용에서 고칠 예정) 
        => nyquist freq 보다 높은 주파수 까지도 다 계산 하기 때문에 22050 이 넘어가는 주파수들은 다 문제가 됨 
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = (freq * t) % 1.0  # phase accumulator normalized to [0, 1]
                              # % 1.0 (modulo) 1로 나눈 나머지 = 0~1 사이로 계속 리셋
                              # 여기 이 수식 자체가 22050을 넘어가는 harmonics 까지도 포함하고 있는 것 
                              # saw 완전 뾰족한 그 모양은 이론적으로 무한 hamonics를 필요로 함. 
                              # but, digital로는 불가능 함(근사치까지만 가능)
    return 2 * phase - 1, t 


def generate_square(freq, duration, sample_rate):
    """Square wave 
    (1) math : y(t) = sign(sin(2π * f * t))  # sign ?
        + sign : 부호(sign)만 추출하는 함수. sine 파의 형태에서 +1, +1, 0, -1, -1 의 부호만 빼와서 비연속적으로 만들어서 square 만듦 
    
    Or Fourier series : sum of odd harmonics
    (2) y(t) = (4/π) * Σ[sin(2π * (2k-1) * f * t) / (2k-1)]

        - only odd harmonics (1, 3, 5, 7, ...)
        - harmonic amplitude = 1/n for odd n
        - hollow sound (even harmonics 가 없음)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine = np.sin(2 * np.pi * freq * t)
    return np.sign(sine), t  #여기서는 (1) sign함수는 이용하는 방법으로 square 생성


def generate_triangle(freq, duration, sample_rate):
    """Triangle wave
    math (Fourier series):
    y(t) = (8/π²) * Σ[(-1)^k * sin(2π * (2k-1) * f * t) / (2k-1)²]

        - only odd harmonics 
        - amplitude = 1/n^2 (제곱으로 감쇠)
        - softer than square wave
        - similar to filtered sqaure 
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = (freq * t) % 1.0
        #Triangle = rises from -1 to +1, then falls back 
        #Saw 랑 같게 modulo 계산 후 나머지값을 챙기는데, 
    tri = 2 * np.abs(2*phase -1) - 1
    return tri, t


#파형 확인
# wave, _ = generate_sine(FREQUENCY, DURATION, SAMPLE_RATE) 
    # 이 함수를 써서 wave, 혹은 t 값만 빼올 수 있음
# print(wave)


def plot_waveform_comparison():
    """모든 waveform 을 시간도메인에서 비교"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        # 여기서 subplots 로 객체를 생성하고, 
        # axes[0], axes[1], axes[2], axes[3] 생성 - 빈상태

    waveforms = [
        ("Sine", generate_sine), #파형이름, 파형메이킹 함수이름
        ("Sawtooth", generate_sawtooth),
        ("Square", generate_square),
        ("Triangle", generate_triangle)
    ]

    for idx, (name, gen_func) in enumerate(waveforms):
        wave, t = gen_func(FREQUENCY, 0.01, SAMPLE_RATE) #10ms plot.
            #함수 호출 후 각 함수에서 wave 와 t 를 return 해옴
        axes[idx].plot(t * 1000, wave, linewidth=1.5) #convert to ms
            # 여기서 위의 각 axes[0].. 안에 데이터가 복사 & 저장이 됨 -> 그래서 for 문이 다 지나가도, 내용이 담겨져 있는것
            # wave, t 는 다음 루프에서 덮어씌워짐 
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'{name}wave - Time domain')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-1.2, 1.2)

    axes[-1].set_xlabel('Time(ms)')
    plt.tight_layout()
    plt.show()

def analyze_spectrum(wave, sample_rate, title):
    #사실 이 함수는 안 쓰이고 있음. 그래도 주석 써놓은게 아까워서 지우지는 않겠다.
    """FFT를 이용한 주파수 스펙트럼 분석
    
    Fast Fourier Transform : 
        - time domain -> freq domain transform
        - 각 주파수 성분의 amplitude 와 phase 추출
    """

    N = len(wave) #샘플 개수
    #Compute FFT
    fft_result = fft(wave)  #시간도메인(wave) -> 주파수 도메인 변환 ( 각 주파수 성분이 얼마나 있는지 분석 )
                            # 결과는 complex(복소수). 크기+위상 정보
    freqs = fftfreq(N, 1/sample_rate) 
        # N : 샘플 개수, 1/sample_rate : 샘플 하나당 간격 (초)
        # fftfreq() : 각 FFT bin 이 몇 Hz 인지?
        # 주파수:   0   1   2   3  -4  -3  -2  -1 => 이렇게 FFT 경과는 대칭이 됨 (음수 주파수 : 양수 주파수의 복제(대칭))

    #Positive frequencies only (FFT는 대칭이므로)
    positive_freqs = freqs[:N//2]
    magnitude = np.abs(fft_result[:N//2]) * 2 / N 
        #fft_result 의 N(샘플개수) 를 2 로 나눈 부분까지만 사용
        # * 2 / N : 뒤쪽 절반을 버렸으니, 에너지가 반으로 줄어들기 때문에 *2배 함 
        # /N : FFT는 샘플 개수에 비례해서 커짐으로 N으로 나누어서 원래 진폭으로 normalize
        # (1) np.abs() : 복소수의 크기 

    #Plot only up to 10kHz for clarity
    mask = positive_freqs < 10000
        # 이 때의 mask = [True, True, True, False, False ..]
        # 사람의 귀는 20kHz 까지 들음. 10kHz 까지만 봐도 충분하다.

    plt.figure(figsize=(12, 5))
    plt.plot(positive_freqs[mask], magnitude[mask], linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    return positive_freqs, magnitude

def plot_all_spectrums():
    """모든 파형 FFT 스펙트럼 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    waveforms = [
        ("Sine", generate_sine), #파형이름, 파형메이킹 함수이름
        ("Sawtooth", generate_sawtooth),
        ("Square", generate_square),
        ("Triangle", generate_triangle)
    ]

    for idx, (name, gen_func) in enumerate(waveforms):
        wave, _ = gen_func(FREQUENCY, DURATION, SAMPLE_RATE)

        N = len(wave)
        fft_result = fft(wave)
        freqs = fftfreq(N, 1/SAMPLE_RATE)

        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N

        #Plot up to 10kHz
        mask = positive_freqs < 10000
        
        #여기의 [::10]이 [::100]으로 되어있어서 sine 파에 뭐가 되게 많게 보였다. 
        axes[idx].stem(positive_freqs[mask][::10], magnitude[mask][::10], 
                       linefmt='b-', markerfmt='bo', basefmt='gray')
        axes[idx].set_xlabel('Frequency (Hz)')
        axes[idx].set_ylabel('Magnitude')
        axes[idx].set_title(f'{name} - Frequency Spectrum')
        axes[idx].grid(True, alpha=0.3)
        # ===== 로그 scale 로 x 축 바꾸기 =====
        axes[idx].set_xscale('log')

        #Annotate(주석을 달다) harmonics 
        if name == "Sine" : 
            axes[idx]. text(0.5, 0.9, 'Only fundamental (1st harmonic)',
                            transform=axes[idx].transAxes, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        elif name == "Sawtooth":
            axes[idx].text(0.5, 0.9, 'All harmonics: 1/n decay',
                          transform=axes[idx].transAxes, fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif name == "Square":
            axes[idx].text(0.5, 0.9, 'Odd harmonics only: 1/n decay',
                          transform=axes[idx].transAxes, fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif name == "Triangle":
            axes[idx].text(0.5, 0.9, 'Odd harmonics: 1/n² decay (faster!)',
                          transform=axes[idx].transAxes, fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()
    print("✓ Spectrum comparison saved")

# Execute Day 1 
if __name__ == "__main__":  # 파이썬 해당 파일에서 직접 실행했을때만 작동되는 것 . 
                            #직접 실행한다면 __name__ == "__main__" 이 True 가 됨 
    print("="*50)
    print("DAY 1 : Basic waveform generation")
    print("="*50)

    plot_waveform_comparison()
    plot_all_spectrums()

    





""" generate_sine
(1)np.linespace(0, 1, 5, endpoint=False)
    - 결과 : [0.0, 0,2, 0.4, 0.6, 0.8] => 0부터 1까지를 5개로 쪼개되, 끝점(1.0)은 포함안함

ex. np.linespace(0, duration, int(sample_rate * duration), endpoint=False)
    - 결과 : 0부터 duration 까지를 samplerate*duration(44100 * 1.0) 한 만큼으로 나누고, 끝점은 포함안함
        [0, 0.0000227, 0.0000454, ... , 0.9999773] 의 배열
    
    => 따라서 np.linespace 의 출력은 array 


(2)phase = 2 * np.pi * freq * t
    =>  phase = [0, 0.0631, 0.1262, ..., 6283.1]  (44100개) . 이런 배열을 출력하기 때문에
        (얘는 x값이기 떄문에 점점 커짐)

return np.sin(phase), t  => 여기에서 np.sin(phase)와 같이 배열 전체에 sin을 적용한다.
    => result = [0, 0.0629, 0.1253, ..., -0.0629]  (44100개)
        (얘는 y값이라서 -1 ~ +1 까지로 수렴함)

(3)Trigonometric functions (삼각함수)
위에서 phase = x
그리고 그걸 np.sin(x) = np.sin(phase) 하면 y축이 -1 ~ +1 사이의 값으로 표현됨
 => sin 함수가 phase 를 높이로 변환해주는 변환기

    ex. phase = np.array[0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        wave = np.sin(phase)

        [0, π/4, π/2, π, 3π/2, 2π]
      = [0도, 45도, 90도, 180도, 270도, 360도]
      = [0.  0.707  1.    0.   -1.    0.  ] => 이렇게 각도가 형성되어 있으므로 

    print(phase) = [0. 0.785  1.571  3.142   4.712   6.283] => 이게 x 값 (np.pi = 3.14 이고, 거기서 *1/2 , *2/3 하면 )
    print(wave)  = [0. 0.707  1.     0.      -1.     0.   ] => 이게 y 값 (그거에 y값을 구하면 (-1 ~ +1))

"""

""" generate_saw
** sine 은 함수자체가 주기적이지만, saw 는 수동으로 주기를 만들어 줘야함 
** 't*f - floor(t*f)' 의 역할을 코드에서는 '(t*f) % 1.0' 으로 표현함 (나머지를 표현해서 리셋되게 함)
    => floor(x) 는 소수점 아래를 버리는 함수. ex. floor(3.7) = 3
        => t * f - floor(t*f) = 소수점 아래만 남기는 것임! 
        ex. t*f = 3.7이라면 -> 3.7 - 3 = 0.7 이 됨 (소수부분만 남음)

    ex. freq = 2 , t = np.linspace(0, 2, 20) -> 0부터 2 까지를 20으로 나누는 time

    (1)
    product = freq * t  => 얘의 출력도 array 
    print(product) 
        # [0.   0.21 0.42 0.63 0.84 1.05 1.26 1.47 1.68 1.89 
        #  2.11 2.32 2.53 2.74 2.95 3.16 3.37 3.58 3.79 4.  ]
    
    (2) saw 만들기
    phase = product % 1.0 (나머지만 출력!)
    print(phase)

    (3) 2 * phase - 1 (범위를 -1 ~ +1 로 바꾸고 싶어서 2를 곱한뒤에 -1 함. 원래의 범위는 0~1까지 였음)
    wave = 2 * phase - 1 => 이것도 출력이 array
    print(wave)
        # [-1.   -0.58 -0.16  0.26  0.68 -0.89 -0.47 -0.05  0.37  0.79
        #  -0.79 -0.37  0.05  0.47  0.89 -0.68 -0.26  0.16  0.58  1.  ]


"""

""" generate_square

(1) sign 함수를 사용하는 방법 
sign(x) = {
   +1  if x > 0   (양수)
    0  if x = 0   (0)
   -1  if x < 0   (음수)
}

    ex. x = [3, -5, 0, 2.7, -0.1, 100, -200]
    result = sign(x)
    print(result)
        => [1, -1, 0, 1, -1, 1, -1] 이런식으로 +1, 0, -1 중에서만 출력이 됨 

    ex. sine = np.sin(2 * np.pi * f * t) 
        [0, 0.5, 0.9, 1.0, 0.5, 0, -0.5, -0.9, -1.0, -0.5, 0, ...] (부드러운 곡선)

        square = np.sign(sine) 이면
        [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, ...]  (딱딱 끊김!)
        => 이런식으로 +1, 0, -1 만 남게 되어 sine 파의 부드러운 곡선을 +1 or -1 로 강제로 만들어 버림!

(2) Fourier series 로 만드는 방법 (홀수 배음 sine 파의 합으로 표현)

y(t) = (4/π) * Σ[sin(2π * (2k-1) * f * t) / (2k-1)]

# Square wave의 Fourier series:
y(t) = (4/π) * [
    sin(1 * ω * t) / 1 +     (1차 harmonic)
    sin(3 * ω * t) / 3 +     (3차)
    sin(5 * ω * t) / 5 +     (5차)
    sin(7 * ω * t) / 7 +     (7차)
    ...
] 여기서 ω = 2πf

    ** 4/pi의 의미 : 진폭을 맞춰주는 상수 (print(4/np.pi) = 약 1.273)

"""


"""if __name__ == "__main__": -> 이 파일을 직접 실행할 때만 실행함 (import 로 불러올 떄는 실행 안됨)


# calculator.py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# 메인 실행 부분
if __name__ == "__main__":
    print("계산기 테스트")
    print(f"2 + 3 = {add(2, 3)}")
    print(f"5 - 2 = {subtract(5, 2)}")

    -> 이런 파일이 있고, 직접 이 파일 내에서 실행한다면, print("계산기 테스트") 이 부분도 출력 되는데, 
    
    다른 프로젝트에서 

    import calculator

    result = calculator.add(10, 20) 이렇게 쓴다면

    # 그냥 30 만 출력됨 ( __name__ 이 부분 출력 안됨 )
    
    

"""
