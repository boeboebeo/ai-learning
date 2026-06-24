"""
==============================================
DAY 11: Z-Transform & Transfer Functions
==============================================

Mathematical Foundation Required :
    - complex numbers (복소수)
    - Differential equations (미분 방정식)
    - Partial fraction decomposition (부분분수 분해)
    - Pole - zero concopt (극점-영점)

Understand discrete-time filters using Z-domain analysis
(Z영역을 이용한 이산시간 필터 분석 이해)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100

def z_transform_basics():
    """
    Z-Transform: Bridge between time-domain and frequency-domain
    (시간 영역과 주파수 영역을 연결하는 다리)

    Mathematical Definition (수학적 정의):
    X(z) = Σ x[n] * z^(-n) for n = -∞ to +∞

    Where z is a complex variable: z = e^(jω) (j = imaginary unit)


    """