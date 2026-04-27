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

fm_modulation_index_sweep()