"""
==============================================
DAY 3: Amplitude Modulation (AM) & Ring Modulation (RM)
==============================================
Goal: AM과 RM의 수학적 원리를 이해하고 sideband 생성을 분석한다.

** AM , RM 은 곱셈변조 **
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

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
    am_signal = am_signal / (1+mod_depth) # AM의 결과값이 1을 넘어가길래 정규화 해줌 

    # print(carrier)

    return am_signal, t, carrier, modulator




def ring_modulation(carrier_freq, modulator_freq, duration, sample_rate):
    """
    Ring modulation(RM)

    Mathematical formula:
    y(t) = carrier(t) * modulator(t)
    y(t) = A_c * sin(2π * f_c * t) * A_m * sin(2π * f_m * t)

    Using trigonometric identity:
    sin(A) * sin(B) = [cos(A-B) - cos(A+B)] / 2 

    Therefore:
    y(t) = (A_c * A_m / 2) * [cos(2π * (f_c - f_m) * t) -
                              cos(2π * (f_c + f_m) * t)]

    ***Carrier Is Suppressed ! ***
    This creates Inharmonic sounds when f_c and f_m are not harmonically related.

    """

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    carrier = np.sin(2 * np.pi * carrier_freq * t )
    modulator = np.sin(2 * np.pi * modulator_freq * t)

    #Ring modulation is simply multiplication
    rm_signal = carrier * modulator

    return rm_signal, t, carrier, modulator 

def compare_am_rm_timedomain():

    carrier_freq = 440
    mod_freq = 10

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    #generate signals
    am_signal, t, carrier, modulator = amplitude_modulation(
        carrier_freq, mod_freq, 1, 0.5, SAMPLE_RATE
    )   # am 은 mod_depth 까지 넣어줘야 함 
    rm_signal, t, carrier, modulator = ring_modulation(
        carrier_freq, mod_freq, 0.5, SAMPLE_RATE
    )

    #Plot short section for visibility
    plot_samples = int(0.1 * SAMPLE_RATE) # 100ms
    t_plot = t[:plot_samples] * 1000  #convert to ms

    #Envelope
    envelope = 1 + 1.0 * modulator[:plot_samples]

    #Find critical points
    max_points, _ = find_peaks(modulator[:plot_samples], height = 0.99)
    min_points, _ = find_peaks(-modulator[:plot_samples], height = 0.99)
    zero_crossings = np.where(np.diff)

    # ============================================
    # AM (왼쪽)
    # ============================================
    
    # AM Modulator → Unipolar Envelope
    axes[0, 0].plot(t_plot, modulator[:plot_samples], linewidth=1, 
                   color='green', alpha=0.5, linestyle='--', label='Bipolar mod')
    axes[0, 0].plot(t_plot, envelope, linewidth=2, color='blue', label='Unipolar env')
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    axes[0, 0].axhline(y=1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    # 화살표 표시
    arrow_x = t_plot[int(len(t_plot)*0.3)]
    axes[0, 0].annotate('', xy=(arrow_x, envelope[int(len(t_plot)*0.3)]), 
                       xytext=(arrow_x, modulator[int(len(t_plot)*0.3)]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=3))
    axes[0, 0].text(arrow_x + 5, 0.5, 'Shift up!', 
                   fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    axes[0, 0].set_ylabel('Amplitude', fontsize=9)
    axes[0, 0].set_title('AM: Bipolar → Unipolar', fontsize=11, fontweight='bold', color='purple')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-1.5, 2.0)
    
    # AM Output
    axes[1, 0].plot(t_plot, am_signal[:plot_samples], linewidth=0.8, color='purple', alpha=0.7)
    axes[1, 0].plot(t_plot, envelope, 'b--', linewidth=1.5, alpha=0.6, label='Envelope')
    axes[1, 0].plot(t_plot, -envelope, 'b--', linewidth=1.5, alpha=0.6)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    axes[1, 0].set_ylabel('Amplitude', fontsize=9)
    axes[1, 0].set_xlabel('Time (ms)', fontsize=9)
    axes[1, 0].set_title('AM Output', fontsize=11, fontweight='bold', color='purple')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # ============================================
    # RM (오른쪽)
    # ============================================
    
    # RM Modulator (Bipolar 그대로)
    axes[0, 1].plot(t_plot, modulator[:plot_samples], linewidth=2, color='red', label='Bipolar mod')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    
    # "No shift!" 표시
    axes[0, 1].text(t_plot[int(len(t_plot)*0.5)], 0.7, 
                   'No shift!', 
                   fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    axes[0, 1].set_ylabel('Amplitude', fontsize=9)
    axes[0, 1].set_title('RM: Bipolar', fontsize=11, fontweight='bold', color='red')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-1.5, 2.0)
    
    # RM Output
    axes[1, 1].plot(t_plot, rm_signal[:plot_samples], linewidth=0.8, color='red', alpha=0.8)
    axes[1, 1].axhline(y=0, color='black', linewidth=1, alpha=0.7)
    axes[1, 1].set_ylabel('Amplitude', fontsize=9)
    axes[1, 1].set_xlabel('Time (ms)', fontsize=9)
    axes[1, 1].set_ylim(-2.0, 2.0)
    axes[1, 1].set_title('RM Output (Carrier suppressed)', fontsize=11, fontweight='bold', color='red')
    axes[1, 1].grid(True, alpha=0.3)  
    
    plt.tight_layout()
    plt.show()

def compare_am_rm_spectrum():
    
    carrier_freq = 1000
    mod_freq = 200

    am_signal, _, _, _ = amplitude_modulation(carrier_freq, mod_freq, 1,
                                            DURATION, SAMPLE_RATE)
    rm_signal, _, _, _ = ring_modulation(carrier_freq, mod_freq, 
                                         DURATION, SAMPLE_RATE)
    
    #Compute FFT
    def get_spectrum(signal): #함수 내부 함수
        N = len(signal) # N = 입력되는 신호배열의 개수 (am_signal도 배열임)
        fft_result = fft(signal) #fft 함수? : 시간 -> 주파수 변환(복소수 결과를 출력함)
            # X[k] = a+ jb = a -> cosine 성분, b -> sine 성분
            # np.abs(X[K]) = 진폭(magnitude)
        freqs = fftfreq(N, 1/SAMPLE_RATE) 
            #fft 값에 대응되는 freq . inverse of the sampling rate
            #fftfreq(N, d) . d = 샘플간격 = 1/SR
            #결과 : [0, 1Hz, 2Hz, ..., SR/2, ..., -1Hz]
        positive_freqs = freqs[:N//2] # 위에서 양수주파수만 가지고 옴 
        magnitude = np.abs(fft_result[:N//2]) / N 
        magnitude[1:-1] *= 2 
            # / N = 진짜 진폭으로 정규화. sin 을 1000개 더하면 값이 1000배 커지니까
                # fft 는 전체 시간 동안의 그 주파수의 합을 구한것이기 때문에 / N 을 해야 실질적인 평균치 나옴
            # fft 는 합.
            # 근데 우리는 +freq, -freq 중에서 절반만 본거고, 그렇기 때문에 에너지 절반이 날라감
            # * 2 해서 복원. 
            # 정확히는 DC (0Hz)랑 Nyquist 는 *2 안해야함. -> 얘네는 짝이 없기때문에 
        return positive_freqs, magnitude 
    
    freqs_am, mag_am = get_spectrum(am_signal) #위의 함수에서의 signal -> am_signal
    freqs_rm, mag_rm = get_spectrum(rm_signal)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # AM spectrum
    axes[0].plot(freqs_am, mag_am, linewidth=5, color='red')
    axes[0].set_xlim(0, 2000)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('AM Spectrum: Carrier + Sidebands')
    # axes[0].set_xscale('log')
    # axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)


    # Annotate peaks
    axes[0].axvline(carrier_freq, color='blue', linestyle='--', alpha=0.7,
                   label=f'Carrier: {carrier_freq} Hz')
    axes[0].axvline(carrier_freq - mod_freq, color='green', linestyle='--', alpha=0.7,
                   label=f'Lower SB: {carrier_freq - mod_freq} Hz')
    axes[0].axvline(carrier_freq + mod_freq, color='orange', linestyle='--', alpha=0.7,
                   label=f'Upper SB: {carrier_freq + mod_freq} Hz')
    axes[0].legend(loc='upper right')
    
    # RM spectrum
    axes[1].plot(freqs_rm, mag_rm, linewidth=5, color='purple')
    axes[1].set_xlim(0, 2000)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('RM Spectrum: ONLY Sidebands (Carrier Suppressed!)')
    axes[1].grid(True, alpha=0.3)
    
    axes[1].axvline(carrier_freq - mod_freq, color='green', linestyle='--', alpha=0.7,
                   label=f'Lower SB: {carrier_freq - mod_freq} Hz')
    axes[1].axvline(carrier_freq + mod_freq, color='orange', linestyle='--', alpha=0.7,
                   label=f'Upper SB: {carrier_freq + mod_freq} Hz')
    axes[1].axvline(carrier_freq, color='red', linestyle=':', alpha=0.5,
                   label=f'Carrier (ABSENT!): {carrier_freq} Hz')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def modulation_depth_experiment():
    """AM modulation depth 의 영향 분석
        Depth 0 -> 1 증가할때의 sideband의 amplitude 변화    
    """

    carrier_freq = 1000
    mod_freq = 200
    depths = [0.2, 0.5, 0.8, 1.0]
        # m = depth (modulation depth)

    fig, axes = plt.subplots(len(depths), 1, figsize=(10, 8))

    for idx, depth in enumerate(depths):
        am_signal, _, _, _ = amplitude_modulation(carrier_freq, mod_freq,
                                                  depth, DURATION, SAMPLE_RATE)
        
        #FFT
        N = len(am_signal)
        fft_result = fft(am_signal)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N

        axes[idx].plot(positive_freqs, magnitude, linewidth=1)
        axes[idx].set_xlim(0, 2000)
        axes[idx].set_ylabel('Magnitude')
        axes[idx].set_title(f'AM with Depth = {depth}')
        axes[idx].grid(True, alpha=0.3)

        #Theoretical sideband amplitude
        theoretical_sb_amp = depth / 2  
            # 삼각함수 계산공식에서 sb 는 1/2 이 됨 
            # AM 에서는 sideband 의 크기가 depth 에 의해 직접 결정됨!
        axes[idx].text(0.7, 0.8, f'Theoretical SB amplitude : {theoretical_sb_amp:.2f}',
                       transform = axes[idx].transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    axes[-1].set_xlabel('Frequency(Hz)')
    plt.tight_layout()
    plt.show()

def inharmonic_rm_example():
    """RM 으로 inharmonic spectrum 생성.
    when carrier and modulator are NOT harmonically related :
        - sidebands become inharmonic 
        - bell like, metallic sound
    """

    carrier_freq = 440
    mod_freq = 311

    rm_signal, t, _, _ = ring_modulation(carrier_freq, mod_freq, DURATION, SAMPLE_RATE)

    # FFT
    N = len(rm_signal)
    fft_result = fft(rm_signal)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    magnitude = np.abs(fft_result[:N//2]) * 2 / N 

    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, magnitude, linewidth=5, color='purple')
    plt.xlim(0, 2000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Inharmonic RM: Carrier=440Hz, Mod=311Hz (not related!)')
    plt.grid(True, alpha=0.3)

    #Calculate and annotate sidebands
    lower_sb = abs(carrier_freq - mod_freq)
    upper_sb = carrier_freq + mod_freq 

    plt.axvline(lower_sb, color='green', linestyle='--', alpha=0.7,
               label=f'Lower SB: {lower_sb} Hz')
    plt.axvline(upper_sb, color='orange', linestyle='--', alpha=0.7,
               label=f'Upper SB: {upper_sb} Hz')
    
    plt.text(0.5, 0.9, f'These frequencies ({lower_sb}Hz, {upper_sb}Hz) are NOT\n'
                       f'harmonics of any fundamental! → Inharmonic/metallic sound',
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    plt.legend()
    plt.tight_layout()
    plt.show()


compare_am_rm_timedomain()
compare_am_rm_spectrum()
modulation_depth_experiment()
inharmonic_rm_example()


"""Compute FFT

signal = [0, 1, 0, -1] (시간순서 데이터. 시간에 따라 측정된 진폭) 로 단순화 해보면
N = 4

fft_result = fft(signal) -> 여기서부터는 주파수 순서 데이터. (k=0 -> 0Hz, k=1 -> 1번 주파수bin..)
fftfreq(N, 1/SR) -> 각 freq bin1, 2, 3, 4.. 등이 실제로 어떤 주파수에 매핑되어야 할지 파악

**FFT 가 실제로 하는 일 (각 주파수 k에 대해서)
X[k] = sum( x[n] * e^(-j2π*kn)/ N ) -> 이걸 n=0부터 N-1 까지 다 더함

=> fft : 이 시간 구간안에서 평균적으로 어떤 주파수가 있었는지 파악.
         average over time 
         so, FFT 는 시간에 따른 변화를 잃어버림 ! 
        => 그래서 stft 를 했던 것 .
        (그래서 spectrogram 은 stft 결과를 나타낸것. 시간에 따른 결과 알수있음! )
        (FFT 는 한 구간 전체를 평균낸 주파수)
        (STFT 는 시간에 따라 변하는 주파수 정보도 담고 있음) -> 시간을 쪼개서 FFT를 여러번 한 것 .
"""
