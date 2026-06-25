"""
==============================================
DAY 11: Z-Transform & Transfer Functions
==============================================

Mathematical Foundation Required :
    - complex numbers (복소수)
    - Differential equations (미분 방정식)
    - Partial fraction decomposition (부분분수 분해)
    - Pole - zero concopt (극점-영점)...

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

    Where z is a complex variable: z = e^(jω) (j = imaginary unit. 허수부)

    Intuition (직관):
    - Laplace transform (라플라스 변환): continuous-time (연속시간)
    - Z-transform: discrete-time (이산시간)

    Transfer Function (전달함수): H(z)
    -H(z) = Y(z) / X(z) = (b₀ + b₁z⁻¹ + b₂z⁻² ...) / (1 + a₁z⁻¹ + a₂z⁻² ...)
                                : numerator(분자)       : denominator(분모)

    Key Properties:
    - Poles (극점): roots of denominator → stability (안정성 결정)
    - Zeros (영점): roots of numerator → frequency response shape (주파수 응답 형태)
    
    Stability Criterion (안정성 조건):
    All poles must be INSIDE the unit circle |z| < 1
    (모든 극점이 단위원 내부에 있어야 안정적)

    
    Example 1: Simple delay
    H(z) = z^(-1) (delay by 1 sample)

        - Time domain : y[n] = x[n-1]
        - Pole : None (no poles!)
        - Zero : At z=0

    Example 2: First-order system
    H(z) = 1 / (1 - {a}*z^(-1) where |a| < 1)

        - Pole : At z = {a}
        - Zero : None
        - Stable: {abs(a) < 1}

    """

def create_pole_zero_plot():
    """
    Pole-Zero Diagram: Visualize filter characteristics
    (필터 특성을 시각화하는 극점-영점 다이어그램)
    
    Interpretation (해석):
    - Poles near unit circle → peaking in frequency response (주파수 응답에서 피크)
    - Poles far from unit circle → flat response (평평한 응답)
    - Zero on unit circle → notch (노치, 특정 주파수 감쇠)
    """

    fig, axes = plt.subplots(2, 2, figsize = (12, 8))

    # Example 1: Simple first-order low-pass
    # H(z) = 0.1 / (1 - 0.9*z^(-1))

    b1 = [0.1]
    a1 = [1, -0.9]

    z1, p1, k1 = signal.tf2zpk(b1, a1)

    ax = axes[0, 0]
    circle = plt.Circle((0, 0), 1, fill=False, color='k', linewidth=1)
    ax.add_patch(circle)
    ax.plot(np.real(p1), np.imag(p1), 'rx', markersize=12, markeredgewidth=2, label='Poles')
    ax.plot(np.real(z1), np.imag(z1), 'bo', markersize=8, label='Zeros')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Low-pass Filter: Pole at z=0.9')
    ax.legend()
    ax.text(0, -1.3, '|z|=1 (Unit Circle)', ha='center')

    plt.tight_layout()
    plt.show()


create_pole_zero_plot()