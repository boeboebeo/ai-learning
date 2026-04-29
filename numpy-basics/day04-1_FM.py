"""
==============================================
DAY 4: Frequency Modulation (FM) & Bessel Functions
==============================================
Goal: FM의 수학적 원리를 이해하고 modulation index가 spectrum에 미치는 영향을 분석한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.special import jv # Bessel function of the first kind

SAMPLE_RATE = 44100
DURATION = 1.0

def frequency_modulation(carrier_freq, modulation_freq, mod_index, duration, sample_rate):
    """
    Frequency Modulation(FM)

    Mathematical formula:
    y(t) = A * sin(2π * f_c * t + I * sin(2π * f_m * t))

    where :
        -f_c : carrier frequency
        -f_m : modulator frequency
        -I : modulation index (변조 지수) - controls spectrum complexity (스펙트럼 복잡도)
            => I = k * A_m(모듈레이터 진폭)

    Instantaneous frequency (순간 주파수):
    f(t) = f_c + I * f_m * cos(2π * f_m * t)

    Frequency deviation (주파수 편차):
    Δf = I * f_m (maximum frequency change from carrier)
        근데 I = k * Am

    Sidebands :
    FM generates INFINITE sidebands at frequencies 
    f_c +/- n * f_m (n = 1, 2, 3, 4, ...)

    Sideband amplitudes determined by Bessel functions:
    Amplitude of nth sideband = J_n(I) (bessel function of order n)

    C:M ratio (Carrier to Modulator ratio)
        - integer ratios ( 1:1, 2:1, 3:2 -> harmonic )
        - non-integer ratios -> inharmonic sound

    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    #modulator signal 
    modulator = np.sin(2 * np.pi * modulation_freq * t)
    
    #FM : modulate the phase of carrier 
    # phase = 2pi * f_c * t + modulation_term
    phase = 2 * np.pi * carrier_freq * t + mod_index * modulator

    fm_signal = np.sin(phase)
    return fm_signal, t 

def bessel_sideband_prediction(mod_index, num_sidebands=10):
        #mod_index(변조지수) 가 주어졌고, 사이드밴드는 10개 일때의 각 사이드밴드의 크기를 예측
    """
    Bessel functions을 이용한 sideband amplitude 예측

    for FM with modulatiion index I:
        - carrier amplitude = J_0(I) (0th order Bessel function)
        - n th sideband amplitude = J_n(I)

    Bessel function properties:
        - J_n(I) oscillates and decays as n increases
            => n이 커질수록 값이 작아짐. 중간중간 0이 될수도 있음 (멀리 있는 사이드밴드는 점점 작아짐)
        - Higher I -> more sidebands with significant amplitude
        - when I = 2.4, J_0(2.4) 약 0 (Carrier disappears!)
            => 특정 I 에서는 중심 주파수(fc)0이 됨

    **Amplitude of nth sideband = J_n(I)
        n = 0 -> carrier
        n = 1 -> 1st sideband
        n = 2 -> 2nd sideband 

        => 그리고 J_n(I) 가 각 주파수의 크기를 결정함 
    
    """

    n_values = np.arange(0, num_sidebands + 1)
        # np.arange : 규칙적으로 증가하는 숫자 리스트를 만드는 함수
        # np.arange(start, stop, step) -> 내 코드에서는 np.arange(0, num_sidebands + 1, 1)
        # 끝 값을 포함시키려고 +1 을 처리함 
        # 따라서 n_values = [0, 1, 2, 3, ...., 10]
    amplitudes = [abs(jv(n, mod_index)) for n in n_values]
        # jv(n, mod_index) 
        # : scipy 의 bessel 함수 J_n(I) 계산 . n번째 사이드 밴드의 크기 = J_n(I)
        # 근데 jv(0, 2.4)라면 거의 0이나와서 캐리어가 없어짐
        # jv(1, 2.4), jv(2, 2.4)는 꽤 큼 
        # jv(n, I) 는 n번째 사이드밴드의 진폭을 계산해주는 함수
    

    return n_values, amplitudes

def fm_modulation_index_sweep():
    """
    Modulation index 를 변화시키며 spectrum 분석

    I 가 증가하면:
        - 더 많은 sidebands 생성
    """

    carrier_freq = 440
    mod_freq = 110 #C:M = 4:1 (harmonic ratio)

    mod_indices = [0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(len(mod_indices), 2, figsize=(10, 8))

    for idx, mod_index in enumerate(mod_indices):
        #generate FM signal
        fm_signal, t = frequency_modulation(carrier_freq, mod_freq, mod_index, DURATION, SAMPLE_RATE)

        #FFT 
        N = len(fm_signal)
        fft_result = fft(fm_signal)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N

        #Plot spectrum
        axes[idx, 0].plot(positive_freqs, magnitude, linewidth=1, color='blue')
        axes[idx, 0].set_xlim(0, 5000)
        axes[idx, 0].set_ylabel('Magnitude')
        axes[idx, 0].set_title(f'FM Spectrum: MOD Index = {mod_index}')
        axes[idx, 0].grid(True, alpha=0.3)

        #Mark expected sideband positions
        for n in range(-10, 11):
            sideband_freq = carrier_freq + n * mod_freq
            if 0 < sideband_freq < 5000:
                axes[idx, 0].axvline(sideband_freq, color='red', alpha=0.2, linestyle='--')

        #Bessel function prediction (이론적 예측)
        n_values, bessel_amps = bessel_sideband_prediction(mod_index, 15)

        axes[idx, 1].stem(n_values, bessel_amps, linefmt='r-', markerfmt='ro', basefmt='gray')
        axes[idx, 1].set_xlabel('Sideband Number(n)')
        axes[idx, 1].set_ylabel('Amplitude(Bessel J_n)')
        axes[idx, 1].set_title(f'Bessel Function Prediction (I={mod_index})')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim(-0.5, 15)

        #Annotage key info
        bandwidth = 2 * (mod_index + 1) * mod_freq # Carson's rule
        axes[idx, 0].text(0.6, 0.7, f'Theoretical BW : {bandwidth:.0f}Hz\n(Carson\'s rule)',
                        transform=axes[idx, 0].transAxes, fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
    axes[-1, 0].set_xlabel('Frequency(Hz)')
    plt.tight_layout()
    plt.show()

def fm_cm_ratio_harmonic_vs_inharmonic():
    """
    C:M ratio (Carrier to Modulator ratio)의 영향
    
    Integer ratios (정수비):
        - 1:1, 2:1 .. -> harmonic spectrum
        - sidebands align with harmonic series (배음렬과 일치)

    Non-integer ratios (비정수비):
        -1:1.414, 3.7:! -> inharmonic spectrum

    """
    carrier_freq = 440
    mod_index = 3.0

    #Different C:M ratios 
    cm_ratios= [
        (1, 1, "1:1 (Harmonic)"),
        (2, 1, "2:1 (Harmonic)"),
        (3, 2, "3:2 (Harmonic)"),
        (1, 1.414, "1:√2 (Inharmonic)"),
        (1, 0.05, "1:0.05 (Harmonic)"),
    ]

    fig, axes = plt.subplots(len(cm_ratios), 1, figsize = (10, 8))

    for idx, (c_ratio, m_ratio, label) in enumerate(cm_ratios):
        #Modulator frequency based on C:M ratio 
        mod_freq = carrier_freq * (m_ratio / c_ratio)
        fm_signal, _ = frequency_modulation(carrier_freq, mod_freq, mod_index,
                                            DURATION, SAMPLE_RATE)
        
        #FFT
        N = len(fm_signal)
        fft_result = fft(fm_signal)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2] * 2 / N)

        axes[idx].plot(positive_freqs, magnitude, linewidth = 1)
        axes[idx].set_xlim(0, 4000)
        axes[idx].set_ylabel('Magnitude')
        axes[idx].set_title(f'C:M = {label}, Mod freq = {mod_freq}Hz')
        axes[idx].grid(True, alpha =0.3)

        #Harmonic markers (배음 위치 표시)

        if "Harmonic" in label:
            for n in range(1, 11):
                harmonic_freq = carrier_freq * n
                if harmonic_freq < 4000:
                    axes[idx].axvline(harmonic_freq, color='green', alpha=0.3, linestyle=':')
            axes[idx].text(0.7, 0.7, 'Green lines = harmonics\nSidebands align',
                               transform= axes [idx].transAxes, fontsize =9,
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else :
            axes[idx].text(0.7, 0.7, 'Sidebands do Not align\nwith harmonics -> inharmonic',
                           transform =axes[idx].transAxes, fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
            
    axes[-1].set_xlabel('Frequency(Hz)')
    plt.tight_layout()
    plt.show()


def bessel_function_visualization():
    """Bessel functions 시각화 
    
    Jn(I) : nth order Bessel function of the first kind
        - Determines amplitude of nth sideband in FM
        - Oscillates(진동) and decays(감쇠) with increasing n
    """

    #indices = index의 복수형
    mod_indices = np.linspace(0, 10, 200)
        #이건 0~10까지를 균등하게 나누는 200개의 값 생성

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # plot J0 ot J5

    for n in range(6):
        bessel_values = [jv(n, I) for I in mod_indices]
        axes[0].plot(mod_indices, bessel_values, label=f'J_{n}(I)', linewidth=2)
            # y축 : bessel 함수 값 = 해당 sideband의 진폭
            # J_0 = carrier, J_1 = 1st sideband, j_2 = 2nd sideband...

    axes[0].set_xlabel('Modulation Index(I)')
    axes[0].set_ylabel('Bessel Function Value')
    axes[0].set_title('Bessel Functions: Sideband Amplitude Prediction')
    axes[0].legend(loc='upper right')
        #그래프 위에 범례를 오른쪽 위에 표시하란 말. 라벨 박스
        #axes[0].plot(. . , label = _ ) 이 label 있어야 읽어서 표시함
    axes[0].grid(True, alpha= 0.3)
    axes[0].axhline(0, color='black', linewidth=0.5)

    #carrier null points (carrier disappears)
    #J0(I) = 0 at I = 약 2.4, 5.5, 8.7 .. 일때 J0(I)가 0이 나옴
    carrier_nulls = [2.4048, 5.5201, 8.6537]
    for null in carrier_nulls:
        if null < 10:
            axes[0].axvline(null, color='red', linestyle='--', alpha=0.5)
            axes[0].text(null, 0.8, f'I={null:.1f}\nCarrier=0!',
                            fontsize=8, ha='center',
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            
    #Practical example : spectrum at differenct I values
    example_indices = [0.5, 2.4, 5.0]
    colors = ['b', 'r', 'g']
        #['blue', 'red', 'green] 은 밑에서 blue-, blueo 때문에 에러가 남

    for I, color in zip(example_indices, colors):
        n_vals = np.arange(0, 16)
        bessel_amps = [abs(jv(n, I)) for n in n_vals]
            # .stem : 막대그래프
        axes[1].stem(n_vals + np.random.uniform(-0.1, 0.1, len(n_vals)),
                        bessel_amps, linefmt=f'{color}-', markerfmt=f'{color}o',
                        basefmt='gray', label=f'I = {I}')
        
    axes[1].set_ylabel('Sideband Order (n)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Sideband Amplitudes for Different Modulation Indices')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
            



def fm_time_domain_visualization():
    """
    FM의 시간도메인의 분석

    Instantaneous frequency (순간 주파수) 변화 시각화
    """

    carrier_freq = 110
    mod_freq = 5 #slow modulation to see frequency changes
    mod_index = 20

    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
        #0.5초 까지니까 sample_rate 의 절반까지
    
    #generate FM
    modulator = np.sin(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * carrier_freq * t + mod_index * modulator
    fm_signal = np.sin(phase) # FM 결과

    #calculate instantaneous frequency (순간 주파수)
    #f(t) = (1/2π) * dφ/dt => 주파수 : 위상을 미분한 것. (한 주기 기준이기때문에 2π를 나누는것)

    instantaneous_freq = carrier_freq + mod_index * mod_freq * np.cos(2 * np.pi * mod_freq * t)
        # 각각 미분해서 * (1/2π) 함

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Modulator
    axes[0].plot(t * 1000, modulator, linewidth=1.5, color='green')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Modulator')
    axes[0].grid(True, alpha=0.3)

    # FM signal 
    axes[1].plot(t * 1000, fm_signal, linewidth=1.5, color='green')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('FM signal')
    axes[1].grid(True, alpha=0.3)

    # Instantaneous frequency : 그 순간에서의 Carrier 가 몇 Hz 로 진동하는지!
    axes[2].plot(t * 1000, instantaneous_freq, linewidth=2, color='red')
    axes[2].set_ylabel('Frequency(Hz)')
    axes[2].set_xlabel('Time(ms)')
    axes[2].set_title('Instantaneous Frequency (Carrier Hz at each moment)')
    axes[2].axhline(carrier_freq, color='blue', linestyle='--', alpha=0.5,
                    label=f'Carrier : {carrier_freq}Hz')
    axes[2].axhline(carrier_freq + mod_index * mod_freq, color='green',
                    linestyle=':', alpha=0.5, label=f'Max: {carrier_freq + mod_index * mod_freq}Hz')
    axes[2].axhline(carrier_freq - mod_index * mod_freq, color='orange',
                    linestyle=':', alpha=0.5, label=f'Min: {carrier_freq - mod_index * mod_freq}Hz')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


"""
마지막 time_domain_visualization 을 시각화해보았을때 
뭔가 FM에서의 형태가 뭔가 Modulator 의 진폭이 작아질때 결과가 느려지기 시작하는것 같은 현상 발견. 
(제일 높은 진폭에서 제일 주파수가 빨라보이지 않음) 

**sin vs cos(위상차이)**
: sine 을 미분하면 cosine 이 됨 (90도 앞당겨짐)

: f(t) = 1/2pi * dϕ(t)/dt - 주파수를 구하는 방법. 위상부분을 t로 미분하고 1/2파이 를 곱합 
       = fc + I * fm *cos(2pi * fm * t) . cos 이 됨 
       = 위상에 sine을 넣었는데, 주파수는 "미분 결과"라서 cos 이 튀어나온 것.

sin 과 cos 는 90(pi/2) 위상차이가 있다

    위 코드에서
    modulator = sin(..)
    instantaneous frequency = cos(..) => 이렇게 각각 다른 위상으로 제어되고 있기 때문에

    => 주파수 변화는 "모듈레이션 값"이 아니라 "모듈레이터의 변화율"에 의해 결정됨

    +modulator(sine)의 값이 0(음수로 떨어지는 0점)점이 되는 그 순간(중간값)
    : 제일 주파수가 느려짐

    +modulator(sine)의 값이 0(양수로 올라가는 0점)점이 되는 그 순간
    : 제일 주파수가 빨라짐

    => 주파수는 sine 값이 아니라, sin 의 기울기(=cos. 미분하면 cos가 나오니까)에 의해 결정됨
        => 그리고 그 기울기의 부호(+, -) 때문에 차이가 생겨남

**f(t) = fc + I * fm * cos(2pi*fm*t) 이기 때문에 cos 값이 주파수를 결정함.

=> cos = modulator 의 변화속도 = 위상의 증가속도 = 그게 높아지면 carrier 주파수가 빠르게 진동함

"""

"""bessel_function_visualization() graph 가 보여주는 것 .

I=0일때, J_0 = 1, 나머지는 0.
-> carrier 만 있고, sideband 없음

I=2일때, J_0 = 0.22, J_1 = 0.58, J_2 = 0.35..
-> carrier 가 약해지고, sideband 가 생김

I=2.4일때, J_0 = 0
-> carrier 가 완전히 사라짐 

"""

"""Jacobi-anger 전개

* β = 0 (FM 없음)
    sin(x + 0*sin(y)) = sin(x)

    # Jacobi-Anger로 계산하면:
    = J_0(0)*sin(x) + J_1(0)*sin(x+y) + J_2(0)*sin(x+2y) + ...

    # Bessel 함수 값:
    J_0(0) = 1
    J_1(0) = 0
    J_2(0) = 0
    ...

    # 따라서:
    = 1*sin(x) + 0 + 0 + ... = sin(x) ✓. 캐리어만 남음

* β = 1 (약한 FM)
    sin(x + 1*sin(y))

    # Bessel 함수 값:
    J_0(1) ≈ 0.765
    J_1(1) ≈ 0.440  
    J_2(1) ≈ 0.115
    J_3(1) ≈ 0.020
    J_4(1) ≈ 0.002  (거의 0)

    # Jacobi-Anger 전개:
    = 0.765 * sin(x)           # carrier
    + 0.440 * sin(x+y)         # 1st upper sideband
    + 0.440 * sin(x-y)         # 1st lower sideband  
    + 0.115 * sin(x+2y)        # 2nd upper
    + 0.115 * sin(x-2y)        # 2nd lower
    + 0.020 * sin(x+3y)        # 3rd (거의 무시 가능)
    + ...

                            => 이렇게 사이드 밴드가 생겨나게 됨!


"""


# fm_modulation_index_sweep()
# fm_cm_ratio_harmonic_vs_inharmonic()
bessel_function_visualization()
# fm_time_domain_visualization()