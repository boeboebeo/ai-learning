"""
==============================================
DAY 5: Phase Modulation (PM) & FM vs PM Comparison
==============================================
Goal: PM의 원리를 이해하고 FM과의 차이점을 분석한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100
DURATION = 1.0

def phase_modulation(carrier_freq, modulator_freq, mod_index, duration, sample_rate):
    """
    Phase modulation(PM)

    Mathematical formula:
    y(t) = A * sin(2π * f_c * t + I * m(t))

    where m(t) is the modulator signal (usually sine wave)

    For sine modulator:
    y(t) = A * sin(2π * f_c * t + I * sin(2π * f_m * t))

    Relationship between FM and PM:
    PM with modulator m(t) = FM with modulator dm(t)/dt (derivative)

    For sine wave modulator:
        - PM : modulator = sin(2π * f_m * t)
        - Equivalent FM : modulator = cos(2π * f_m * t) * f_m

    In practice:
        - PM and FM produce identical spectra for sine modulators
        (사인파 모듈레이터일때는 PM과 FM이 같은 스펙트럼을 만듦. but, 시간에서의 동작(위상 vs 주파수 변화 방식)은 다름
        - but different for complex modulators!

    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    #PM : modulator directly affects phase
    modulator = np.sin(2 * np.pi * modulator_freq * t)
    phase = 2 * np.pi * carrier_freq * t + mod_index * modulator 

    pm_signal = np.sin(phase)

    return pm_signal, t, modulator



