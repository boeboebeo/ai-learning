"""
==============================================
DAY 13: Window Methods for FIR Filter Design
==============================================

MATHEMATICAL PREREQUISITES:
    - Fourier Series: Functions represented as sum of sines / cosines
    - Convolution (합성곱): basic idea = multiply and sum
    - Frequency domain : Different way to look at signals

KEY CONCEPT:
    - We can't create a "perfect" low-pass filter in practice.
So we use WINDOWS to make imperfect filters that are "good enough"

**FIR 저역통과 필터를 만들때 

1) 이상적 계수 : 이상적인 저역통과의 임펄스응답은 sinc . 무한히 김
    => 잘린 sinc 51개를 뚝 자름

2) window 를 곱 하기 (여기가 windowing)
    => 그냥 자르면 양 끝이 갑자기 끊겨서 주파수 응답에 잔 물결이 생김
        + 그래서 51개 짜리의 window 를 만들어서 계수에 하나씩 곱함
        + firwin 은 내부적으로 Hamming 을 기본 Window로 사용함

3) 각 window 차이 : tradeoff 
    - transition : 통과 <-> 차단 전환이 얼마나 급한지 (날카로울수록 확실하게 컷 할수있는 것)
    - stopband attenuation : 막아야 할 주파수를 얼마나 깊게 누르는지

    + Rectanular : 사실상 window 안 쓴것 (걍 모든 계수가 *1 하는거임 )
        - transition 은 제일 날카롭지만 ripple 이 넘 심함

    + Kaiser : 베타 라는 손잡이 가 있어서 사용자가 날카로움 vs 잡음억제 밸런스를 조절할 수 있음

** window 는 크기만 바꿈! 
    => 어떤 window 를 곱하든 window 자체가 좌우 대칭이면 계수의 대칭은 안 깨진다.
    => 여전히 linear phase 유지가능함

** window 는 그냥 "곱할 숫자 배열 !"

windowed_b = sinc_coeffs * window   # 51개 * 51개, 원소끼리 곱


**필터 만들때 딱 한번 window를 곱하고 -> 필터링 할때는 신호 처리시에 매 샘플마다 곱함!
    (sinc 계수 * window )


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_why_windows_needed():
    """what's the problem they solve?
    
    **PROBLEM : creating a perfect low-pass filter

    + Ideal Low -pass filter response:
        0 to 5000Hz: PASS ( gain : 1 )
        5000+ Hz: Block ( gain : 0 )
        => but here's the problem 

    + To create this mathematically, we need:
        - INFINITE number of filter coefficients
        - Infinite length filter
        - Can't implement in real computer..

    **SOLUTION: Use a window
        -Instead of infinite filter, use FINITE filter:
        1) Design ideal filter (mathematically)
        2) Truncate to finite length (51 taps, 101 taps, etc.) [truncate : 잘리다, 삭제하다]
        3) Apply WINDOW function to smooth edges
        4) Result: Practical FIR filter that works!

        => But truncating creates ARTIFACTS (ripples)!
        Windows reduce these artifacts by tapering edges (끝부분을 가늘어지게 함으로써)


    • Sharp edges = ringing/artifacts
    • Smooth edges = less artifacts
    
    """

def understand_window_functions():
    """Different window functions and their properties(특성)
    
    **Windows = mathematical functions that:
    1. Are 1 in the middle (middle=full)
    2. Taper to 0 at edges (양쪽 끝=0으로)
    """

    """ [WINDOWS] Different Choices and their tradeoffs [tradeoffs : 절충안. 한가지 이점을 얻기위해서 버릴것 버리고, 취할건 취함]

        ** 여기서 말하는 Ripple 은 시간도메인에서의 artifacts 가 아님
            => window 의 Ripple은 frequency domain 에서의 ripple 

            - 통과대역이 평평하지 않게 출력되는것. 차단대역 또한 출렁거림 (0이 아니고)
            - 이상적인 상태는 통과대역은 1, 차단대역은 0임

            => 통과대역 ripple : 통과시키려던 주파수들이 정확히 1배가 아니라, 
                어떤건 1.05배, 어떤건 0.95배 이렇게 미세하게 다른 크기로 출력됨
            => stopband ripple : 막으려던 고주파가 완전히 0이 아니라 조금씩 새어나옴 
                -> 제거하려던 주파수 대역이 미세하게 남음

                ! gibbs ! 

        ** 둘다 sinc 를 유한하게 자른것의 결과! 
        (pre-rining 은 시간 도메인 ripple -> impulse reponse "lfilter" 출력)
        (passband/stopband 은 주파수 도메인 ripple -> freq response "freqz" 출력)

    sinc 자름 → 시간 ripple(ringing) + 주파수 ripple (푸리에 한 쌍) 
    → 대칭(linear phase)이라 앞쪽에도 = pre-ringing 
    → minimum phase면 비대칭이라 pre-ring 제거 (minBLEP이 이걸 함)

    1) Rectangular window

    Formula : w[n] = 1 for all n
    shape : Flat top, sharp edges

        - Pros:
            Narrowest transition band ( 가장 좁은 과도대역 )
            Sharp cutoff
        - Cons:
            Ripples in passband and stopband
            ~21 dB stopband attenuation 
            Pre-ringing audible

    2) HAMMING Window (해밍)

    Formula: w[n] = 0.54 - 0.46*cos(2π*n/(N-1))
    Shape: Raised cosine, smooth edges

        - Pros:
            Good balance
            ~43dB stopband attenuation
            Slightly wider transition than rect, but much less ripple
        - Cons:
            Slightly wider transition band
            still some ripple

    3) BLACKMAN Window (블랙맨)

    Formula: w[n] = 0.42 - 0.5*cos(...) + 0.08*cos(...)
    (More complex cosine sum)
    Shape: Smooth, gradually tapers

        - Pros:
            Low ripple (리플 매우 적음)
            ~74 dB stopband attenuation!
            Clean response

        - Cons:
            Wider transition band
            Slower cutoff

    4) KAISER Window (카이저)

    Formula: w[n] = I0(β*sqrt(1-(2n/N-1)^2)) / I0(β)
    (Uses Bessel functions)
    Shape: Adjustable via parameter β [adjustable:조절할 수 있는] [via : -를 통해]

        - Pros:
            TUNABLE! Control ripple vs transition (조절 가능!)
            Best all-around (가장 유연함)
            Can meet any specification

        - Cons:
            Must choose parameter β (하나의 선택사항 더)

    [SUMMARY TABLE]

    print("Window      | Ripple | Transition | Attenuation | Complexity")
    print("────────────┼────────┼────────────┼─────────────┼──────────")
    print("Rectangular | High   | Narrow     | 21 dB       | Simple")
    print("Hamming     | Medium | Medium     | 43 dB       | Easy")
    print("Blackman    | Low    | Wide       | 74 dB       | Medium")
    print("Kaiser      | Tuned  | Tuned      | Tuned       | Complex")
    print("────────────┴────────┴────────────┴─────────────┴──────────")

        [tuned: 조율된]

    """

def create_and_visualize_windows():
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    N = 51  #windows length
    n = np.arange(N) # N 의 배열화 

    # Create different windows
    rect_window = signal.windows.boxcar(N)
    hamm_window = signal.windows.hamming(N)
    black_window = signal.windows.blackman(N)
    kaiser_window = signal.windows.kaiser(N, beta=8.6) #beta control shape

    windows = [
        ('Rectangular', rect_window, axes[0, 0], 'red'),
        ('Hamming', hamm_window, axes[0, 1], 'green'),
        ('Blackman', black_window, axes[1, 0], 'blue'),
        ('Kaiser (β=8.6)', kaiser_window, axes[1, 1], 'purple'),
    ]

    for name, window, ax, color in windows:
        #Plot window
        ax.plot(n, window, linewidth=2.5, color=color, marker='o', markersize=3)
        ax.fill_between(n, 0, window, alpha=0.2, color=color)
            # 그래프 내부 채우기
            # fill_between => 두 선 사이의 공간을 채워주는거라서 y1, y2 사이를 채움
            # 0(y1)과 window(y2) 값 사이를 채우는 것 

        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Sample')
        ax.set_title(f'{name} Window\n(51 samples)')
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, alpha = 0.3)

        # Add explanation 

        if name == 'Rectangular':
            ax.text(0.5, 0.1, 'Sharp edges\n→ Ripples!', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        elif name == 'Hamming':
            ax.text(0.5, 0.1, 'Smooth edges\n→ Less ripple',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        elif name == 'Blackman':
            ax.text(0.5, 0.1, 'Very smooth\n→ Minimal ripple!',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax.text(0.5, 0.1, 'Tunable!\n→ Best choice often',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
            
    plt.tight_layout()
    plt.show()

def design_filters_with_windows():
    """Actually design FIR filters using windows
    
    **Designing FIR filter with window
    
    1) Define specifications
    2) Choose window type and length
    3) Create ideal filter
    4) Examine coefficients
    5) Compare frequency responses
    
    """

    # 1. Define specifications 

    cutoff = 5000 #Hz
    fs = 44100 #sample rate
    normalized_cutoff = cutoff / (fs / 2) 
        # nyquist로 cutoff 나누면 0-1 range 안으로 들어옴 몇 % 정도의 주파수 인지
        # nyquist / nyquist = 1
    
    # 2. Choose window type and length
    # Decision : Use hamming window, 51taps.
        # why? : good ripple vs transition tradeoff
        # 51 taps = reasonable length (not too long)
        # ~ 43dB attenuation = acceptable for audio

    # 3. create ideal filter
    # scipy.signal.firwin does all this automatically

    # Design filter
    fir_rect = signal.firwin(51, normalized_cutoff, window='rectangular')
    fir_hamm = signal.firwin(51, normalized_cutoff, window='hamming')
    fir_black = signal.firwin(51, normalized_cutoff, window='blackman')

    # 4. Examine coefficients
    # hamming window FIR (first 10 coefficients)
    # window 'tapering' effect : larger in middle, smaller at edges 

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # get frequency responses
    w, h_rect = signal.freqz(fir_rect, [1], worN=1000)
    w, h_hamm = signal.freqz(fir_hamm, [1], worN=1000)
    w, h_black = signal.freqz(fir_black, [1], worN=1000)

    # Convert to dB
    mag_rect = 20 * np.log10(np.abs(h_rect) + 1e-10)
    mag_hamm = 20 * np.log10(np.abs(h_hamm) + 1e-10)
    mag_black = 20 * np.log10(np.abs(h_black) + 1e-10)

    # Convert frequency to Hz 
    freq_hz = w * fs / (2*np.pi)
        # 여기서 w(=약 3.14) 는 0부터 3.14까지를 1000개로 나눈 것 
        # 근데 다시 w * fs 곱하고, 그걸 2*np.pi 한걸로 나누게 되면 Hz 가 나오게 됨
        # ex. w = 3.14 라면 (nyquist) 
        # 3.14 * 44100 / 2*3.14.. 하면 약 22050 hz 나옴

    # print((3.14 * 44100) / (2*np.pi))
    # print(w)
    # print(freq_hz[-1]) => 여기서 22050 나와야 함 (뒤에서 부터 세는것)
    # freqz가 주는 마지막 w가 정확히 π\pi π가 아니라, π보다 아주 살짝 작은 값(끝 점 포함 안함)


    #Plot 1: Full view
    ax = axes[0]
    ax.plot(freq_hz, mag_rect, linewidth=2, label='Rectangular', color='red', alpha=0.7)
    ax.plot(freq_hz, mag_hamm, linewidth=2, label='Hamming', color='green')
    ax.plot(freq_hz, mag_black, linewidth=2, label='Blackman', color='blue')
    ax.axvline(cutoff, color='k', linestyle='--', alpha=0.5, label='Cutoff')
    ax.axhline(-6, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-3, color='gray', linestyle=':', alpha=0.5)

    # 여기서 firwin의 기본정의는 -6dB인 그 지점을 0.5배 (진폭 절반) 으로 cut off 로 지정함
    ax.set_xlim(0, 15000)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('FIR filters: Different windows\n(51 taps, cutoff at 5 kHz)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    #Plot 2: Zoomed stopband ( to see ripple )
    ax = axes[1]
    stopband_idx = freq_hz > 7000 
        # COF 보다 2000Hz 이상이여야 transition band 넘어서 Stopband 의 ripple이 보임
        # ripple 은 stopband 성질
        # stopband_idx = [F, F, F, ..... T, T ... ] <- 7000Hz 넘는 지점부터 True 
            # => 이걸 배열에 []로 넣으면 boolean indexing True 인 자리만 골라냄
    ax.plot(freq_hz[stopband_idx], mag_rect[stopband_idx], linewidth=2,
            label='Rectangular (RIPPLY)', color = 'red', alpha=0.7)
        # freq_hz[i] 와 mag_rect[i]는 같은 인덱스로 짝지어진 한쌍이기 때문에 
        # x, y 에 둘 다 똑같이 넣어야 '주파수-크기' 짝이 유지됨
    ax.plot(freq_hz[stopband_idx], mag_hamm[stopband_idx], linewidth=2,
           label='Hamming (less ripple)', color='green')
    ax.plot(freq_hz[stopband_idx], mag_black[stopband_idx], linewidth=2,
           label='Blackman (smooth)', color='blue')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Zoomed: Stopband Ripple Comparison\n(notice rectangular has bumps!)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # rectanluar - transition band 좁음 , 큰 ripple (-21dB)
    # hamming - transition band 중간 , 중간 ripple (-53dB)
    # blackman - transition bnad 넓음 , 작은 ripple (-74dB)
    # 이런 tradeoff 는 "빨리 떨어지기" <=> "깨끗하게 막기" 로 서로 맞바꿈

    """
    1) FIR - transition 가파름 정도
        => 탭수 + window 가 결정함

        **탭수가 많을수록 transition 좁아짐(가파름)
        **탭수 고정시에는 rect 가파름 / blackman 완만함

    2) IIR - dB/oct ( pole ) 
        => 필터 차수(order) = 극점 개수 가 결정함

    **그래서 dB/oct 는 사실 IIR 의 개념
    FIR 은 직선으로 쭉 내려가는게 아니고, ripple 로 출렁이며 감 
    
        -> dB/oct 라는 하나의 기울기 숫자로 표현이 안됨

    **self oscillation 이 안되는 필터도 거의 대부분 IIR 임
    => 극점은 있는데(=IIR) 단위원에 바짝 안 붙여서, 공진은 하지만 발진까지는 안하게 함
    (피드백은 있음 -> IIR , but, 발진점까지 안 올린 것 (limiter 가 발진을 막음))
    """

    plt.tight_layout()
    plt.show()


"""window trade off - 

transition band <=> ripple 을 좋게 동시게 가질 수 는 없다

    **Kaiser 는 beta 파라미터로 그 사이를 조절

    **MATHMATICAL 
    : 시간 영역에서 좁으면 <-> 주파수 영역에서 넓어짐 ( 둘다 좁게는 불가능 )

    + window 를 짧게(시간축에서 좁게) 자르면 -> 주파수 측에서 넓게 퍼짐(transition band 넓음)
    + window 를 길게(탭 많이) -> 주파수 축에서 좁아짐(가파름

"""




# create_and_visualize_windows()
design_filters_with_windows()