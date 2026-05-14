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
    => impulse train을 적분하면 saw wave
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