"""
==============================================
DAY 12: IIR vs FIR Comparison
==============================================

MATHEMATICAL PREREQUISTITES :
    - Difference equations
    - Convolution Multipyling and adding signals
    - Z- transform 

KEY CONCEPT:
there are TWO main types of digital filters:
1. FIR - (Finite impulse response )  - NO feedback
2. IIR - (Infinite impulse response) - HAS feedback 


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_fir_vs_iir():
    """what is the difference? : IIR vs FIR
    
1. FIR
    - Difference Equation
    : y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + ... => only uses PREVIOUS INPUTS
    => no feedback
    (After some time, output becomes zero (응답이 유한 길이))

    ex. y[n] = 0.5*x[n] + 0.3*x[n-1] + 0.2*x[n-2]


2. IIR
    -Difference Equation
    : y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - a2*y[n-2] - ...
    => Uses PREVIOUS OUTPUTs => has feedback
    (Impulse response continues indefinitely Due to "feedback" loop)

    ex. y[n] = 0.5*x[n] - 0.9*y[n-1] 여기서 y[n-1] 이 feedback! 

    
    """

    """FIR vs IIR 구현방법의 차이

    1) FIR - tap의 개수가 중요 (현재 입력, 과거입력1, 과거입력 2... )

    => 위를 구현하려면 Input Beffer 에 x[n], x[n-1], x[n-2].. 이 버퍼만 있으면 됨!
    +여기서 tap 이란 Delay 하나 + 계수 하나
    (tap 이 많아질수록 더 오래전 과거 까지 보는것 -> impulse response 가 더 길어짐
    => 몇개의 tap 을 사용하여 FIR 필터의 주파수 응답을 더 정교하게 할것인가가 가장 중요한 설계 변수)

    2) IIR - 출력을 다시 입력으로 집어넣어야 함
    
    - IIR first-order 에는 y[n-1] 만 존재
    - IIR second-order 에는 y[n-1], y[n-2] 존재
    - IIR third-order 에는 y[n-1], y[n-2], y[n-3] 존재 
    - IIR fourth-order 에는 y[n-1], y[n-2], y[n-3], y[n-4] 존재 

    => 즉 ORder 는 몇단계의 피드백을 사용하는가! 의 중요한 사항

    signal.butter(4, 0.3)은 4차 Butterworth 필터 
        => Butterwordth 라는 아날로그 필터를 디지털로 변환한 결과 Pole 가 4개 생긴다.
        => 차분 방정식에도 y[n-1], y[n-2], y[n-3], y[n-4] 가 생김

    **즉, 
    
    FIR의 firwin 은 원하는 Ideal sinc -> window -> 계수 완성
        (처음부터 Impulse response 를 만드는 방식)

        - 원하는 impulse response 설계
        - Tap 계수가 중요
        - Window method 기반
        - 선형위상 가능

    IIR의 Butterworth는 아날로그 회로 -> Pole 위치 계산 -> 디지털 변환(Bilinear Transform)
        -> Difference Equation

        - butterworth, heby1, elip, bessel 등의 방법이 있음
        - 아날로그 필터 프로토타입 기반
        - 선형위상 불가능 

    
    """

def create_fir_and_iir_filters():
    """
    Create concrete FIR and IIR filters
    Show their structure clearly  
    """
    
    # 1. FIR filter (low pass filter - length : 51 taps)
    fir_b = signal.firwin(51, 0.3) 
        # FIR low pass filter 의 계수(b)를 자동으로 만들어주는 함수 b=[1, -1]의 이 계수!
        # 근데 firwin 의 win : window method(창 함수 방법)을 사용한 설계기법을 사용함
        # 51-taps low-pass fir (51개 짜리의 배열 , FIR filter 의 길이)
        # y[n] = b0​x[n] + b1​x[n−1] + ⋯ + b50​x[n−50] 이렇게 과거 50개의 입력을 가중평균 하는것!
        # 탭이 많을수록 필터가 더 날카롭고 정확해지지만 계산량 늘고, 지연도 커짐 
        # 0.3 : cutoff (nyquist 의 30% 지점)
        # 여기서 fir_b 는 계수 배열
    """여기서 fir_b 이 돌려주는 51개의 숫자는 sinc 함수 모양 
        (가운데가 가장 크고, 양옆으로 갈수록 출렁이며 작아지는 대칭 배열)
       .
      . .
     .   .
. . .     . . .      ← 가운데 크고 양옆으로 진동하며 감소 (좌우 대칭)

    => 여기서 sinc 는 무한히 긴데, 그걸 51개로 뚝 자르면 잘린 끝부분 때문에 주파수 응답에 ripple 이 생김
       그래서 양끝을 부드럽게 0으로 깎아내리는 window 함수(기본값은 hamming window)를 곱해줌
        
        """
    fir_a = np.array([1]) # no feedback term (just 1)
        # FIR 은 피드백이 없으니 그냥 분모가 1 임 
        # H(z) = fir_b(51개의 계수) / 1 <- 이게 분모가 1인것 (그럼 pole 이 없는것?)
        # => 전달함수의 분모가 1인것 
        # FIR 에는 그럼 극점(pole). 정확히는 "단위원 위/밖에 있는 의미있는 극점이 없다"
        # 특정 주파수 죽이는건 잘하고, 영점 배치로 간접적으로 완만한 봉우리는 만들 수 있지만
        # => 뾰족하게 키우는 공진(resonanc)를 만들지는 못함
        # FIR로는 자가발진 하는 공진은 못만들기 때문에
        # Moog ladder filter 는 본질적으로 IIR (디지털 구현시 IIR 이여야 함)

    print(f"FIR filter")
    print(f"\nCoefficients (b) : {len(fir_b)} values")
    print(f"    First 5: {fir_b[:5]}")
    print(f"Denominator (a) : {fir_a}")
    print(" Just [1] = No feedback")

    print(f"\nDifference Equation:")
    print(f"y[n] = {fir_b[0]:.4f}*x[n] + {fir_b[1]:.4f}*x[n-1] + {fir_b[2]:.4f}*x[n-2]....")
    print(f"        (uses {len(fir_b)} previous inputs)")

    # 2. IIR filter (low pass filter(Butterworth) - order : 4th-older)
    iir_b, iir_a = signal.butter(4, 0.3) 
        # 4th-older butterworth filter 
    
    print(f"\nIIR filter")
    print(f"\nNumerator (b) : {len(iir_b)} values")
    print(f"    {iir_b}")
    print(f"Denominator (a) : {len(iir_a)} values")
    print(f"     {iir_a}")
    print("     NOT just [1]! HAS feedback terms!")

    print(f"\nDifference Equation:")
    print(f"y[n] = {iir_b[0]:.4f}*x[n] + {iir_b[1]:.4f}*x[n-1]+... - {iir_a[1]:.4f}*y[n-1] - {iir_a[2]:.4f}*y[n-2]-...")
    print(f"    Feeds back previous output!")

    return fir_b, fir_a, iir_b, iir_a

def compare_fir_iir_properties(fir_b, fir_a, iir_b, iir_a):
    
    fig, axes = plt.subplots(2, 3, figsize=(12,8))

    # get impulse response (_ir)
    impulse = np.zeros(200)
    impulse[0] = 1

    fir_ir = signal.lfilter(fir_b, fir_a, impulse)
    iir_ir = signal.lfilter(iir_b, iir_a, impulse)

    # get frequency responses 
    w, fir_h = signal.freqz(fir_b, fir_a, worN=500)
    w, iir_h = signal.freqz(iir_b, iir_a, worN=500)

    fir_mag = 20 * np.log10(np.abs(fir_h) + 1e-10) #꼭 크기 구하려면 절대값 씌워야함
    iir_mag = 20 * np.log10(np.abs(iir_h) + 1e-10)

    # plot 1 : FIR impulse response
    ax = axes[0, 0]
    ax.stem(fir_ir, basefmt = ' ')
    ax.grid(True, alpha = 0.3)
    ax.set_ylabel('Amplitude')
    ax.set_title('FIR : Impulse Response\n(FINITE - ends)')
    ax.text(0.5, 0.9, f'length: {len(fir_b)}taps', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor = 'lightblue', alpha=0.5))
    
    # plot 2 : IIR impulse response
    ax = axes[0, 1]
    ax.stem(iir_ir, basefmt = ' ')
    ax.grid(True, alpha = 0.3)
    ax.set_ylabel('Amplitude')
    ax.set_title('IIR : Impulse Response\n(INFINITE - keeps going)')
    ax.text(0.5, 0.9, f'Never truly ends\n(continues indefinitely)', 
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # plot 3 : summary table
    ax = axes[0, 2]
    ax.axis('off')

    comparison_text = """
    FIR vs IIR SUMMARY
    ──────────────────
    
    FIR (Finite IR):
    • No feedback
    • Always stable ✓
    • Linear phase possible ✓
    • High order needed (more taps)
    • More CPU intensive
    
    IIR (Infinite IR):
    • Has feedback
    • Must check stability
    • Minimum phase
    • Low order sufficient
    • Less CPU intensive
    """
    
    ax.text(0.05, 0.95, comparison_text, fontsize=9, verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # plot 4 : FIR frequency response
    ax = axes[1, 0]
    ax.plot(w/np.pi, fir_mag, linewidth=2, color='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(-80, 5)
    ax.grid(True, alpha = 0.3, which='both')
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title('FIR : Magnitude Response\n(51 taps)')
    
    # plot 5 : IIR frequency response
    ax = axes[1, 1]
    ax.plot(w/np.pi, iir_mag, linewidth=2, color='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(-80, 5)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('IIR: Magnitude Response\n(4th order = fewer taps)')
    ax.grid(True, alpha=0.3, which='both')

    # Plot 6: Order Comparison
    ax = axes[1, 2]
    ax.axis('off')
    
    order_text = f"""
    FILTER ORDERS
    ──────────────
    
    FIR:
    • Taps = {len(fir_b)}
    • Filter length = ~51
    
    IIR:
    • Order = 4
    • Effective = ~5
    
    For SAME cutoff freq:
    IIR needs ~10x fewer taps!
    
    Why?
    IIR uses feedback (recursive)
    Feedback allows compact response
    
    Trade-off:
    FIR: Always stable, linear phase
    IIR: Must verify stability
         but much more efficient
    """
    
    ax.text(0.05, 0.95, order_text, fontsize=9, verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
def understand_linear_phase():
    """
    Linear phase: FIR advantage (모든 주파수가 동일하게 지연됨)
        - 파형형태 보존
        - 임펄스 왜곡 없음
        - 오디오에 좋음
        - FIR can achieve linear phase (symmetric coefficients)
        - IIR cannot achieve linear phase (minumum phase always) !

    **sinc function의 모양을 보면 ex. 51탭일때 25번째의 가운데 index 가 제일 큼 & 좌우대칭

        + linear phase 를 만드려면 임펄스 응답이 반드시 좌우 대칭(symmetric)이여야 함
        + 대칭인 FIR 은 모든 주파수를 같은 시간만큼 지연시키기 때문
        + 과거, 현재, 미래를 다 가지고 있어야 이상적인 sinc
        + 근데 실시간에서는 미래를 아직 모르므로 실제 FIR 에서는 전체를 N/2(N: sinc 함수의 tap 수) 만큼 뒤로 밈
           (이게 linear phase 사용시 딜레이가 생기는 이유

        + linear phase FIR 에서는! 현재가 Impulse response 의 가운데! 가 되는것 **


              메인 피크
                 │
      ╱╲        ╱│╲        ╱╲
─────╱──╲──────╱─│─╲──────╱──╲─────
         ╲╱                   ╲╱
    ↑ 피크 "이전"의 출렁임     ↑ 피크 "이후"의 출렁임
    = PRE-ringing          = POST-ringing

    => 필터에 신호를 적용하면, 출력은 '입력의 각 지점에 이 임펄스 응답을 깔아서 합침(convolution 곱)

    
    
    """
    fir_b_sym = signal.firwin(51, 0.3) # symmetric

    w, h = signal.freqz(fir_b_sym, [1], worN=500)
        # freq response 알려줌 . h 는 500개로 나눈 각 주파수의 복소수 배열(복소응답)
        # w : 3.14 까지의 반바퀴 회전을 500개로 나눔  => w = nyquist/2인데 
        # [1] : 분모 (feedback 없으므로 pole(극점) 없음)
    # print(w)
    # print(h)
    phase = np.unwrap(np.angle(h))
        # 시간상 얼마나 밀어내는지 (위상)
        # np.angle(h) : 복소수의 각도 뽑기 (np.abs(h) : 크기(이득) 뽑는것)
        # np.unwrap(..) : np.angle()은 각도를  −π ∼ +π 범위로만 돌려주므로 
        # 위상은 주파수가 올라가면서 계속 누적되어서 -pi를 넘어가버림 (확점프됨 -> saw 파형처럼)
        # => 2pi 씩 더하거나 빼서 끊긴 부분을 매끄럽게 이어줌

    # check if coefficients are symmetric
    is_symmetric = np.allclose(fir_b_sym, fir_b_sym[::-1])
    if is_symmetric:
        print("✓ This FIR filter HAS LINEAR PHASE!")

    """Delay 의 두가지 종류! 

    1) x[n-1] . 메모리에 저장된 과거 샘플 (시간상의 흐름) 이건 무조건 1/sr 차이의 간격을 가짐



    2) Linear phase 에서의 delay 

    y[n] = 0.2x[n]+0.5x[n-1]+0.2x[n-2] 이 식에서는 현재 출력을 계산하려면 현재, 과거1, 과거2 를 다 읽음
        => 근데 linear phase FIR 은 계수가 반드시 대칭이여야 함 
        => but, 가장 큰 계수가 가운데 있음

        ex. 0.2 , 0.5 , 1.0 , 0.5 , 0.2 (대칭)

        이 필터의 식은 y[n] = 0.2x[n] + 0.5x[n-1] + 1.0x[n-2] + 0.5x[n-3] + 0.2x[n-4] 이렇게 됨
        이 출력은 사실상 이 다섯샘플의 중앙 시점을 대표하는 값이 됨 x[n-2]

            => 그래서 평균적으로 2 sample 정도 늦게 출력됨 
            Group delay = (N-1)/2

            ex. 51 tap이면 (51-1)/2 = 25 samples delay 가 발생됨 

    ** 근데 모든 주파수를 동일하게 지연시킨다는건 ? 
    입력에 100Hz, 1000Hz, 5000Hz 가 있을때 

    100 Hz   → 25 sample 늦음
    1000 Hz  → 25 sample 늦음
    5000 Hz  → 25 sample 늦음 => 이렇게 모두 같은 sample 만큼 지연됨
        (출력이 다시 입력으로 들어가는 과정이 없음 => 구조자체가 항상 동일)

    but, 일반 IIR 에서는 
    100 Hz   → 10 sample
    1000 Hz  → 17 sample
    5000 Hz  → 5 sample => 이렇게 각 주파수 성분이 다른 시점에 도착해서 파형이 변형된다.
        (피드백으로 인해서 복소평면에서 벡터들이 주파수마다 다른 방식으로 누적되기 때문)

    
    **차분방정식에서의 계수의 의미
    y[n] = 0.2x[n] + 0.5x[n-1] + 0.2x[n-2]

    0.2x[n] : 현재 샘플의 20%만 반영함
    0.5x[n-1] : 1샘플전 값을 50%만 반영함
    0.2x[n-2] : 2샘플전 값을 20%만 반영함


    """

    




# fir_b, fir_a, iir_b, iir_a = create_fir_and_iir_filters()
# compare_fir_iir_properties(fir_b, fir_a, iir_b, iir_a)

understand_linear_phase()




