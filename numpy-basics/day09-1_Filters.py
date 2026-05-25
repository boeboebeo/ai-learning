"""
==============================================
DAY 9: Filters (Biquad & State Variable Filters)
==============================================
Goal: 필터의 수학적 원리를 이해하고 다양한 필터 타입을 구현한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.0

def biquad_filter(signal_input, filter_type, cutoff_freq, resonance, sample_rate):
    """
    Biquad Filter 

    Biquad = "Bi-quadratic" (2차 필터)

    Transfer function (전달함수)
    H(z) = (b0 + b1*z^(-1) + b2*z^(-2)) / (a0 + a1*z^(-1) + a2*z^(-2))

        차분방정식과 완전히 같은 내용임 (위 전달함수 H(z))
        z^(-1) : 1샘플 과거(Delay 1)
        z^(-2) : 2샘플 과거(Delay 2)
        => 차분방정식은 어떻게 계산하냐를 알려주지만, 전달 함수는 주파수별로 어떻게 반응하냐를 분석할 수 있게 함
        
        z = e^(jw) 를 대입하면 (w=각주파수) -> 각 주파수에서 필터가 신호를 얼마나 증폭/감쇠하는지 바로 계산 가능함
        H(z) → z = e^(jω) 대입 → H(e^(jω)) = 주파수 응답



    Difference equation (차분방정식)
    y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0

        지금 출력값 y[n]은, 지금/과거 입력값들과 과거 출력값들의 가중합이다 (b 계수 : 입력에 곱하는 가중치, a 계수 : 과거 출력에 곱하는 가중치. feedback)
        지금 출력 = 지금 입력 + 1샘플 전 입력 + 2샘플 전 입력 - 1샘플 전 출력 - 2샘플 전 출력
        (각각에 숫자(계수)를 곱해서 더함)

        Ex. b0=1, b1=1, 나머지가 다 0이면 y[n] = x[n] + x[n-1] 이게 전부
            => 지금 샘플이랑 1샘플 전 거 더해라! 
                : 저음 강조, 고음이 깎임 -> 고음은 샘플값이 빠르게 왔다갔다 하는데, 직전값이랑 더하면 서로 상쇄되어버림. 
                  근데 저음은 천천히 변하니까 더하면 오히려 커짐

        x[n]   지금 이 순간 들어온 입력
        x[n-1] 1샘플 전 입력
        x[n-2] 2샘플 전 입력
        y[n-1] 1샘플 전 출력 (자기 자신의 과거)
        y[n-2] 2샘플 전 출력
            b 계수들 : 입력 쪽 가중치(feedforward)
            a 계수들 : 출력 쪽 가중치(feedback. 자기 자신을 되먹임) => 위 수식은 그냥 레시피임

        => y[n-1], y[n-2] 항 : 출력이 다시 입력으로 들어오는 구조. 아날로그 필터에서의 커패시터/인덕터가 하던 일
        (아날로그에서 커패시터가 과거 전압을 기억하듯이, 디지털에서도 이전 샘플값을 메모리에 저장해서 같은 효과를 낸다)

        
    Parameters:
    - cutoff_freq: filter cutoff frequency (차단 주파수)
    - resonance: Q factor (공명, 0.5 ~ 20+)
      - Low Q: gentle slope (부드러운 기울기)
      - High Q: sharp peak (날카로운 피크), self-oscillation (자기 발진)
    
    Filter types:
    - lowpass (LPF): passes low frequencies (저역 통과)
    - highpass (HPF): passes high frequencies (고역 통과)
    - bandpass (BPF): passes middle band (대역 통과)
    - notch: rejects middle band (대역 차단)
   """
    
    # Normalize frequency (정규화된 주파수)
    omega = 2 * np.pi * (cutoff_freq / sample_rate)
        # 주파수를 각도로 변환하는것 
        # 2pi 안에서 전체에서 차지하는 비율이라고 보면 됨
        # Cutoff _Freq / sample_rate = 0 ~ 1 사이의 값으로 정규화를 하는것 -> * 2pi는 라디안 단위로 값을 변화시킴

        # 디지털 필터에서는 Hz 를 직접 모르고, 1샘플 안에서 몇 바퀴 도는가로 주파수를 이해함
        # ex. 2pi * 1000 / 44100 = 약 0.1425 라디안 => 한 샘플마다 원을 0.1425 라디안씩 돌음
        # 샘플레이트가 높을수록 omega 도 작아짐 (같은 주파수도 샘플이 많으면 한 샘플당 조금씩만 전진)
    sin_omega = np.sin(omega)
        # 그 각도의 세로위치
    cos_omega = np.cos(omega)
        # 그 각도의 가로위치

    # Q factor (공명 계수)
    Q = resonance
    alpha = sin_omega / (2 * Q)
        # b0, b1, b2, a1, a2 를 계산할때 거의 모든 식에 alpha가 들어감
        # 얼마나 넓은 주파수 범위에 필터가 작용하는지를 결정해야 함(필터가 작용하는 주파수 범위의 너비)
        # Q가 커지면 alpha가 작아짐 -> 필터 폭이 좁아짐
        # alpha 는 절반짜리 한쪽을 나타내는 값이기때문에 2 * Q 를 처리함

    # Calculate coefficients based on filter type (필터종류에 따라서 b, a 계수를 다르게 셋팅)
    # a0, a1, a2 는 모두 동일함 -> LP, HP, BP, Notch 는 오직 b의 계수만 바꾸는 것
    # : a계수는 feedback 부분이기 때문에 항상 동일

    if filter_type == 'lowpass': # 느린 변화의 낮은 주파수는 유지, 빠른 변화의 높은 주파수는 통과
        b0 = (1 - cos_omega) / 2
        b1 = 1 - cos_omega
        b2 = (1 - cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
        """ cos_omega = cutoff freq 에서의 cos 값

        - cutoff freq 가 낮으면 -> omega가 작음 -> cos_omega 가 1에 가까움
            : 1 - cos_omega 가 0에 가까움

        - cutoff freq 가 높으면 -> omega가 큼 -> cos_omega 가 -1에 가까움
            : 1 - cos_omega 가 2에 가까움

            => b1 이 b0, b2 의 딱 두배. 셋다 양수이기 -> 저음을 모으는 모양이 됨
        
        """
        
    elif filter_type == 'highpass':
        b0 = (1 + cos_omega) / 2
        b1 = -(1 + cos_omega)
        b2 = (1 + cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    elif filter_type == 'bandpass':
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    elif filter_type == 'notch':
        b0 = 1
        b1 = -2 * cos_omega
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha

    # Normalize coefficients (정규화)
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    # Apply filter using difference equation
    filtered = signal.lfilter(b, a, signal_input)

    return filtered, b, a