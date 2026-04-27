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