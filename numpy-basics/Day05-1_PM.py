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

def frequency_modulation_equivalent(carrier_freq, modulator_freq, mod_index, duration, sample_rate):
    """
    FM implementation (for comparison with PM)

    integrate = 적분
    (derivative = 미분)
    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    modulator = np.sin(2 * np.pi * modulator_freq * t)

    # FM : integrate modulator for phase (모듈레이터를 이용해 정의된 '주파수'를 적분해서 위상을 만듦)
    # For sine modulator, integral of sin is -cos 
    # FM 에서는 sine modulator 가 위상으로 들어갈 때 적분때문에 cosine 형태로 바뀜
    phase = 2 * np.pi * carrier_freq * t + mod_index * modulator

    fm_signal = np.sin(phase)

    return fm_signal, t, modulator 

def compare_fm_pm_sine_modulator():
    """
    Sine wave modulator를 사용한 FM vs PM 비교

    Result : IDENTICAL spectra! (완전히 동일한 스펙트럼)
    This is because sine's derivative is cosine(same shape, 90degree phase shift)

    """

    carrier_freq = 440
    mod_freq = 110
    mod_index = 3.0

    pm_signal, t, pm_mod = phase_modulation(carrier_freq, mod_freq, mod_index, 
                                             DURATION, SAMPLE_RATE)
    fm_signal, t, fm_mod = frequency_modulation_equivalent(carrier_freq, mod_freq,
                                                           mod_index, DURATION, SAMPLE_RATE)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Time domain comparison (짧은 구간)
    plot_samples = int(0.02 * SAMPLE_RATE) # 20ms
    t_plot = t[:plot_samples] * 1000

    axes[0, 0].plot(t_plot, pm_signal[:plot_samples], linewidth=1, color='blue', label='PM')
    axes[0, 0].plot(t_plot, fm_signal[:plot_samples], linewidth=1, color='red',
                    linestyle='--', alpha=0.7, label='FM')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_xlabel('Time(ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    #PM spectrum
    N = len(pm_signal)
    fft_pm = fft(pm_signal)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_pm = np.abs(fft_pm[:N//2]) * 2 / N
    
    axes[0, 1].plot(positive_freqs, mag_pm, linewidth=1, color='blue')
    axes[0, 1].set_xlim(0, 3000)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('PM Spectrum')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FM spectrum
    fft_fm = fft(fm_signal)
    mag_fm = np.abs(fft_fm[:N//2]) * 2 / N
    
    axes[1, 0].plot(positive_freqs, mag_fm, linewidth=1, color='red')
    axes[1, 0].set_xlim(0, 3000)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('FM Spectrum')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Difference (should be ~0)
    axes[1, 1].plot(positive_freqs, np.abs(mag_pm - mag_fm), linewidth=1, color='green')
    axes[1, 1].set_xlim(0, 3000)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude Difference')
    axes[1, 1].set_title('Spectrum Difference (PM - FM)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.5, 0.9, 'Nearly zero! PM ≈ FM for sine modulator',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    plt.tight_layout()
    plt.show()



def complex_modulator_fm_pm_difference():
    """complex modulator
    (복잡한 변조 신호)를 사용하면 FM 과 PM 의 차이 발생

    Example : square wave modulator
        -PM : phase jumps instantly (즉시 위상 점프)
        -FM : frequency changes (because square's dereivative = impulse train)

    """
    carrier_freq = 440
    mod_freq = 55
    mod_index = 5.0

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    #Square wave modulator
    square_mod = np.sign(np.sin(2 * np.pi * mod_freq * t ))

    #PM with square modulator
    phase_pm = 2 * np.pi * carrier_freq * t + mod_index * square_mod 
    pm_signal = np.sin(phase_pm)

    # FM with square modulator
    # square wave  derivative = impulse train (매우 다른 결과)
        # => 사각파를 미분하면 순간적으로 튀는 spike 가 됨 (무한대 같은 스파이크)
    # Approximate by integrating square wave 

    from scipy import integrate
    square_integral = integrate.cumtrapz(square_mod, t, initial = 0)
        # 적분해서 위상을 만듦 
    phase_fm = 2 * np.pi * carrier_freq * t + mod_index * square_integral 
    fm_signal = np.sin(phase_fm)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))

    #Modulator 
    plot_samples = int(0.03 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000
    zero_crossings = np.where(np.diff(np.signbit(square_mod)))[0]

    axes[0, 0].plot(t_plot, square_mod[:plot_samples], linewidth=1.5, color='green')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Square Wave Modulator')
    axes[0, 0].grid(True, alpha=0.3)

   
    
    axes[0, 1].plot(t_plot, square_integral[:plot_samples], linewidth=1.5, color='orange')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Integrated Square (for FM)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PM signal
    axes[1, 0].plot(t_plot, pm_signal[:plot_samples], linewidth=1, color='blue')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('PM with Square Modulator (abrupt phase jumps)')
    axes[1, 0].grid(True, alpha=0.3)

    for ct in zero_crossings:
        if ct < plot_samples:
            ct_time = t_plot[ct]
            axes[1, 0].axvspan(ct_time - 0.5, ct_time + 0.5, color='lime', alpha=0.5)
                #axvspan(x_start, x_end , ...) : x_start 부터, x_end 까지 세로 영역을 칠함
                #square 는 불연속적이기 때문에 +1 에서 -1로 순간적으로 점프함. integral 도 기울기가 갑자기 바뀜 => 위상 점프
            axes[1, 0].text(ct_time, 1.0, 'zero-crossing',
                                    fontsize=7, ha='center')
                                    # bbox=dict(boxstyle='round', alpha=0.3)

    # FM signal
    axes[1, 1].plot(t_plot, fm_signal[:plot_samples], linewidth=1, color='red')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('FM with Square Modulator (smooth frequency shifts)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Spectra comparison
    N = len(pm_signal)
    fft_pm = fft(pm_signal)
    fft_fm = fft(fm_signal)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_pm = np.abs(fft_pm[:N//2]) * 2 / N
    mag_fm = np.abs(fft_fm[:N//2]) * 2 / N
    
    axes[2, 0].plot(positive_freqs, mag_pm, linewidth=1, color='blue')
    axes[2, 0].set_xlim(0, 3000)
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Magnitude')
    axes[2, 0].set_title('PM Spectrum (richer harmonics)')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(positive_freqs, mag_fm, linewidth=1, color='red')
    axes[2, 1].set_xlim(0, 3000)
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude')
    axes[2, 1].set_title('FM Spectrum (different character)')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def pm_feedback_example():
    """
    PM with feedback (self-feedback)

    y(t) = sin(2π * f_c * t + I * y(t-1))

    Feedback creates extremely complex spectra!
    Used in Yamaha DX7
        (discrete time(이산 시스템. 샘플단위로 계산), 
         fixed operator graph->모든 routing이 미리 정해져 있음,
         feedback gain 제한/phase -> 에너지가 무한히 커지지 않게 설계됨)
         => DX7 은 정해진 구조의 operator FM 

    =>캐리어(사인파) 가 존재, 현재 소리의 위상에 '이전 순간의 자기 자신'을 다시 넣음
    (그 출력이 자기 자신을 계속 변조에 다시 씀)
    (모듈레이터가 외부 신호가 아님. 바로 이전 출력 값)

    """
    carrier_freq = 220
    feedback_amount = 0.7 # 0 to ~0.9

    t = np.linspace(0, DURATION, int(SAMPLE_RATE*DURATION), endpoint=False)

    #Initialize output
    pm_feedback = np.zeros(len(t))
        # 결과를 빈 저장할 배열 -> 소리가 0이라는게 아니라, 아직 계산 안했다는 뜻 
        # 매번 append 를 하면 느리기 때문에 틀 먼적 만드는 것.

    #Generate with feedback (iterative process)
    #시간 t를 쪼갠 샘플 인덱스 => 시간을 아주 잘게 쪼갠 이산(discrete 시스템)
    for i in range(len(t)):
        if i ==0:   #첫 샘플에는 '이전 값'이 없음 -> i=0이면 y[-1]같은게 존재하지 않음. 시작 값.
            feedback_signal = 0
        else:   #그 다음 샘플부터는 바로 직전 출력값을 가지고 옴 . 아 그래서 for 문 안에서 phase를 구함(밑에)
            feedback_signal = pm_feedback[i-1]
                #pm_feedback 배열의 '이전 값 하나'를 꺼내서 feedback_signal 변수에 넣는다.
                #현재 값을 만들때 '직전 출력값'을 다시 사용하려는 것 

        phase = 2 * np.pi * carrier_freq * t[i] + feedback_amount * feedback_signal
            # feedback_amount : index, feedback_signal = 그 전의 샘플의 값 
        pm_feedback[i] = np.sin(phase)
            # 각각을 계산해서 pm_feedback[i]의 자리에 하나씩 집어넣음.
            # np.sin(phase)의 결과 '단일 숫자 1개'를 배열의 i번째 칸에 저장함 
            # 현재 Phase는 t[i]기준으로 계산된 값 -> 스칼라(숫자 하나) ! => np.sin(phase)도 결과가 숫자 하나
            # 각 i 는 하나의 시간 샘플이고, 그 순간의 출력값을 순서대로 계산하는 구조


    #Compare with no feedback 
    pm_no_feedback = np.sin(2 * np.pi * carrier_freq * t)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Time domain
    plot_samples = int(0.02 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000
    
    axes[0, 0].plot(t_plot, pm_no_feedback[:plot_samples], linewidth=1, color='blue')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Pure Sine (no feedback)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_plot, pm_feedback[:plot_samples], linewidth=1, color='red')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'PM with Feedback ({feedback_amount})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectra
    N = len(pm_no_feedback)
    fft_clean = fft(pm_no_feedback)
    fft_feedback = fft(pm_feedback)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_clean = np.abs(fft_clean[:N//2]) * 2 / N
    mag_feedback = np.abs(fft_feedback[:N//2]) * 2 / N
    
    axes[1, 0].plot(positive_freqs, mag_clean, linewidth=1, color='blue')
    axes[1, 0].set_xlim(0, 5000)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Spectrum: Pure Sine (single peak)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(positive_freqs, mag_feedback, linewidth=1, color='red')
    axes[1, 1].set_xlim(0, 5000)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Spectrum: Feedback PM (complex harmonics!)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.5, 0.9, 'Feedback creates rich harmonic content!',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    

    plt.tight_layout()
    plt.show()


def multi_operator_fm():
    """Multi-operater FM
    
    Classic FM synthesis architecture:
        - operator = oscillator
        - operators can modulate each other in various topologies(위상배치)

    **topology = 연결 그래프.
    ex. 2-operator stack : Op2 -> Op1 
    ex. parallel : Op1 + Op2 

    Examples 2-operator stack(2-OP stack)
        - Op2 modulates Op1
        - Op1 is the carrier 

    **operator 
     = oscillator + envelope + amplitude control + routing 
     => operator (연산 블록)

    **FM stack 
    : 한 operator가 다른 operator를 직렬로 변조하는 구조

    """

    carrier_freq = 440 # Op1
    modulator_freq = 440 * 3 #Op2 (C : M = 1 : 3)
    mod_index = 2.0

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    # operator 2 (modulator)
    op2 = np.sin(2 * np.pi * modulator_freq * t)

    # operator 1 (carrier, modualated by op 2)
    phase_op1 = 2 * np.pi * carrier_freq * t + mod_index * op2
    op1 = np.sin(phase_op1)

    # Compare with 2 parallel operators (no modulation)
    parallel_sum = np.sin(2 * np.pi * carrier_freq * t) + np.sin(2 * np.pi * modulator_freq * t)
    parallel_sum /= 2  #normalize

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Modulated (FM stack)
    plot_samples = int(0.005 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000
    
    axes[0, 0].plot(t_plot, op1[:plot_samples], linewidth=1, color='blue')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('2-Operator FM Stack (Op2 → Op1)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_plot, parallel_sum[:plot_samples], linewidth=1, color='red')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('2-Operator Parallel (no modulation)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectra
    N = len(op1)
    fft_stack = fft(op1)
    fft_parallel = fft(parallel_sum)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    mag_stack = np.abs(fft_stack[:N//2]) * 2 / N
    mag_parallel = np.abs(fft_parallel[:N//2]) * 2 / N
    
    axes[1, 0].plot(positive_freqs, mag_stack, linewidth=1, color='blue')
    axes[1, 0].set_xlim(0, 8000)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('FM Stack Spectrum (complex sidebands)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(positive_freqs, mag_parallel, linewidth=1, color='red')
    axes[1, 1].set_xlim(0, 8000)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('Parallel Spectrum (just 2 peaks)')
    axes[1, 1].grid(True, alpha=0.3)


    plt.tight_layout()
    plt.show()


complex_modulator_fm_pm_difference()
# pm_feedback_example()
# multi_operator_fm()