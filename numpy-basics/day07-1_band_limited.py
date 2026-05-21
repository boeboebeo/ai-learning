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

    # Phase for impulse train
    phase = (freq * t) % 1.0

    # BLIT formula
    # Avoid division by zero : epsilon(아주작은값 = 0 으로 나누는것을 방지하는 안전장치)
        # = 1e-10 = 0.0000000001.
        # 만약 denominator 가 0일 경우, 에러날 수 있으므로, epsilon을 더해서 처리함
        # phase = 0, 1, 2, 3, 4 ..(정수)인 경우 np.sin(np.pi * phase) = 0 이기 때문에 나눗셈 불가능
            # => 그래서 M * np.sin(np.pi * phase + epsilon) 해서 아주 살짝 틀어지게 함
    epsilon = 1e-10
    denominator = M * np.sin(np.pi * phase + epsilon)
        #분모
    numerator = np.sin(np.pi * M * phase)
        #분자

    blit = numerator / (denominator + epsilon)

    # Normalize (정규화)
    blit = blit / M

    return blit, t, M


def blit_to_sawtooth(blit_signal, sample_rate):
    """
    Convert BLIT to sawtooth via integration (적분)
    
    Integration = cumulative sum (누적합)
    Leaky integrator (누설 적분기) to prevent DC buildup
    """
    # Leaky integrator coefficient (DC 차단 계수)
    leak = 0.999
        # leak : 0.999
        # 매 샘플 0.1% 감소하게 함 -> DC 누적 방지, 발산방지, 안정적인 sawtooth. 안정적 적분
        # 순수 적분 = 발산 / Leaky 적분  = 안정적 sawtooth
 
    saw = np.zeros_like(blit_signal)
    accumulator = 0
    
    for i in range(len(blit_signal)):
        accumulator = leak * accumulator + blit_signal[i]
        saw[i] = accumulator
    
    # Normalize to [-1, 1]
    saw = saw / np.max(np.abs(saw))
    saw = 2 * saw - 1  # center around 0
    
    return saw

def polyblep_residual(t, dt):
    """
    PolyBLEP: Polynomial Band-Limited Step (다항식 대역 제한 계단)
        => 불연속점(계단)을 부드럽게 만들어서 aliasing 을 줄이는 기법
    
    Concept:
    - Add correction polynomial at discontinuities (불연속점에 보정 다항식 추가)
    - Very efficient (매우 효율적)
    - Good quality (좋은 품질)
    
    Polynomial residual (잔차 다항식):
    - Smooths out the discontinuity (불연속점 부드럽게)
    - 2nd order polynomial (2차 다항식)
    
    Parameters:
    - t: phase distance from discontinuity (불연속점으로부터 위상 거리)
    - dt: phase increment per sample (샘플당 위상 증가량)
    """
    if t < dt:
        # Rising edge (상승 에지)
        t = t / dt
        return t + t - t * t - 1.0
    elif t > 1.0 - dt:
        # Falling edge (하강 에지)
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    else:
        return 0.0
    

def polyblep_sawtooth(freq, duration, sample_rate):
    """
    PolyBLEP sawtooth generation
    
    Algorithm:
    1. Generate naive sawtooth (기본 톱니파 생성)
    2. Detect discontinuities (불연속점 감지)
    3. Apply PolyBLEP correction (보정 적용)
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    dt = freq / sample_rate  # phase increment (위상 증가량)
    phase = 0
    
    output = np.zeros_like(t)

    for i in range(len(t)):
        # Naive sawtooth
        naive_saw = 2 * phase - 1
        
        # PolyBLEP correction at discontinuity
        correction = polyblep_residual(phase, dt)
        
        output[i] = naive_saw - correction
        
        # Advance phase (위상 진행)
        phase += dt
        if phase >= 1.0:
            phase -= 1.0

    return output, t


"""
    1) Impulse : 한 점에서만 값이 있고 나머지는 0인 신호 + 모든 주파수를 동시에 포함
        ex. impulse = [0, 0, 0, 1, 0, 0, 0] (한 부분만 1, 나머지는 0)

        # Impulse 생성
        impulse = np.zeros(100)
        impulse[50] = 1  # 50번째 위치에만 1

    2) Impulse train(임펄스 열) : 일정한 간격으로 반복되는 임펄스 열
        ex. 주기 t 마다 임펄스 발생 
        ex. impulse train의 FFT = 또 다른 impulse train
    
    """

""" Naive vs PolyBLEP vs BLIT vs Additive 

1)

2)

3)

4)

"""