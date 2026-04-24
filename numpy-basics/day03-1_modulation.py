"""
==============================================
DAY 3: Amplitude Modulation (AM) & Ring Modulation (RM)
==============================================
Goal: AM과 RM의 수학적 원리를 이해하고 sideband 생성을 분석한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

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

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))

    #generate signals
    am_signal, t, carrier, modulator = amplitude_modulation(
        carrier_freq, mod_freq, 0.8, 0.5, SAMPLE_RATE
    )   # am 은 mod_depth 까지 넣어줘야 함 
    rm_signal, t, carrier, modulator = ring_modulation(
        carrier_freq, mod_freq, 0.5, SAMPLE_RATE
    )

    #Plot short section for visibility
    plot_samples = int(0.1 * SAMPLE_RATE) # 100ms
    t_plot = t[:plot_samples] * 1000  #convert to ms

    #carrier 
    axes[0].plot(t_plot, carrier[:plot_samples],  linewidth=1, color='blue')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Carrier: {carrier_freq}Hz (pure sine)')
    axes[0].grid(True, alpha=0.3)


    #modulator
    axes[1].plot(t_plot, modulator)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Modulator: {mod_freq}Hz')
    axes[1].grid(True, alpha=0.3)

    #AM output
    axes[2].plot(t_plot, am_signal[:plot_samples], linewidth=1, color='red')
    axes[2].set_ylable('Amplitude')
    axes[2].set_title('AM Output: Carrier present + sidebands')
    axes[2].grid(True, alpha=0.3)
    axes[2].text(0.02, 0.95, 'Note: amplitude varies but pitch stays the same',
                 transform=axes[2].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def compare_am_rm_spectrum():
    
    carrier_freq = 1000
    mod_freq = 200

    am_signal, _, _, _ = amplitude_modulation(carrier_freq, mod_freq, 0.8,
                                            DURATION, SAMPLE_RATE)
    rm_signal, _, _, _ = ring_modulation(carrier_freq, mod_freq, 
                                         DURATION, SAMPLE_RATE)
    
    #Compute FFT
    def get_spectrun(signal):
        N = len(signal)
        fft_result = fft(signal)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N
        return positive_freqs, magnitude
    
    freqs_am, mag_am = get_spectrun(am_signal)
    freqs_rm, mag_rm = get_spectrun(rm_signal)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    


compare_am_rm_timedomain()