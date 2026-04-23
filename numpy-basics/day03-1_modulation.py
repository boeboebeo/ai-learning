"""
==============================================
DAY 3: Amplitude Modulation (AM) & Ring Modulation (RM)
==============================================
Goal: AM과 RM의 수학적 원리를 이해하고 sideband 생성을 분석한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100
DURATION = 1.0

def amplitude_modulation(carrier_freq, modulator_freq, mod_depth, duration, sample_rate):
    """
    Amplitude Modulation (AM)
    y(t) = carrier(t) * [1 + depth*modulator(t)]
    y(t) = A_c * sin(2π * f_c * t) * [1 + m * sin(2π * f_m * t)]

    Expanding this (trigonometric identity):
    y(t) = A_c * sin(2π * f_c * t) +
           (A_c * m / 2) * cos(2π * (f_c - f_m) * t ) - 
           (A_c * m / 2) * cos(2π * (f_c + f_m) * t )

    Frequency components (sidebands) :
    1. Carrier : f_c (original carrier frequency)
    2. Lower sideband : f_c - f_m
    3. Upper sideband : f_c + f_m

    Parameters:
    - carrier freq : 반송파 주파수(main pitch)
    - modulator_freq : 변조 주파수
    - mod_depth : 0~1, modulation intensity (0 = no AM, 1 = full AM)

    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulator = np.sin(2 * np.pi * modulator_freq * t)

    # AM formula
    am_signal = carrier * (1 + mod_depth * modulator)

    return am_signal, t, carrier, modulator

