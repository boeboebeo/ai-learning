import numpy as np
import matplotlib.pyplot as plt

# Simple signal: low freq + high freq
sample_rate = 1000
t = np.linspace(0, 1, sample_rate)

low_freq = 5   # Hz (low tone)
high_freq = 50  # Hz (high tone)

signal = np.sin(2 * np.pi * low_freq * t) + 0.5 * np.sin(2 * np.pi * high_freq * t)

# Graph
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Low freq only
axes[0].plot(t[:200], np.sin(2 * np.pi * low_freq * t[:200]), linewidth=2)
axes[0].set_ylabel('Amplitude', fontsize=10)
axes[0].set_title('Low Frequency (5 Hz) - Slow wave', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# High freq only
axes[1].plot(t[:200], 0.5 * np.sin(2 * np.pi * high_freq * t[:200]), 
             linewidth=2, color='red')
axes[1].set_ylabel('Amplitude', fontsize=10)
axes[1].set_title('High Frequency (50 Hz) - Fast wave', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Mixed signal
axes[2].plot(t[:200], signal[:200], linewidth=2, color='green')
axes[2].set_xlabel('Time (s)', fontsize=10)
axes[2].set_ylabel('Amplitude', fontsize=10)
axes[2].set_title('Mixed Signal (5Hz + 50Hz)', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Low-Pass Filter effect
from scipy import signal as sig

# Low-Pass Filter design (below 50Hz only)
cutoff = 10  # Hz
nyquist = sample_rate / 2
normal_cutoff = cutoff / nyquist

# Butterworth filter
b, a = sig.butter(4, normal_cutoff, btype='low')

# Apply filter
filtered = sig.filtfilt(b, a, signal)

# Comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(t[:200], signal[:200], linewidth=1.5, label='Original (5Hz + 50Hz)')
axes[0].set_ylabel('Amplitude', fontsize=10)
axes[0].set_title('Before Low-Pass Filter', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[:200], filtered[:200], linewidth=1.5, color='green', 
             label='Filtered (5Hz only)')
axes[1].set_xlabel('Time (s)', fontsize=10)
axes[1].set_ylabel('Amplitude', fontsize=10)
axes[1].set_title('After Low-Pass Filter (cutoff=10Hz)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Frequency response of Low-Pass Filter
from scipy import signal as sig

# Multiple cutoff comparison
cutoffs = [10, 30, 100]  # Hz

fig, axes = plt.subplots(len(cutoffs), 1, figsize=(10, 8))

for idx, cutoff in enumerate(cutoffs):
    # Filter design
    normal_cutoff = cutoff / nyquist
    b, a = sig.butter(4, normal_cutoff, btype='low')
    
    # Calculate frequency response
    w, h = sig.freqz(b, a, worN=8000, fs=sample_rate)
    
    # Graph
    axes[idx].plot(w, np.abs(h), linewidth=2)
    axes[idx].axvline(cutoff, color='red', linestyle='--', linewidth=2,
                     label=f'Cutoff = {cutoff} Hz')
    axes[idx].axhline(0.707, color='green', linestyle='--', alpha=0.5,
                     label='-3dB point')
    axes[idx].set_ylabel('Gain', fontsize=9)
    axes[idx].set_title(f'Low-Pass Filter (cutoff={cutoff}Hz)', 
                       fontsize=10, fontweight='bold')
    axes[idx].set_xlim(0, 200)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    
    # Fill regions
    axes[idx].fill_between([0, cutoff], 0, 1.2, alpha=0.2, color='green',
                          label='Pass')
    axes[idx].fill_between([cutoff, 200], 0, 1.2, alpha=0.2, color='red',
                          label='Stop')

axes[-1].set_xlabel('Frequency (Hz)', fontsize=10)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

freq = np.linspace(0, 500, 1000)
cutoff = 100

# 1. Ideal Low-Pass Filter (Perfect Rectangle!)
ideal_filter = np.ones_like(freq)
ideal_filter[freq > cutoff] = 0

axes[0].plot(freq, ideal_filter, linewidth=3, color='blue')
axes[0].axvline(cutoff, color='red', linestyle='--', linewidth=2)
axes[0].set_ylabel('Gain', fontsize=10)
axes[0].set_title('IDEAL Low-Pass Filter (Perfect Rectangle!)', 
                  fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.1, 1.2)
axes[0].fill_between([0, cutoff], 0, 1.2, alpha=0.2, color='green')
axes[0].fill_between([cutoff, 500], 0, 1.2, alpha=0.2, color='red')

# Feature annotation
axes[0].annotate('Perfectly FLAT\n(all low freq pass equally)', 
                xy=(50, 1), xytext=(50, 1.15),
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

axes[0].annotate('Perfectly SHARP\n(cuts exactly here!)', 
                xy=(cutoff, 0.5), xytext=(200, 0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# 2. Real Low-Pass Filter (Smooth curve)
from scipy import signal as sig
b, a = sig.butter(4, cutoff/(500/2), btype='low')
w, h = sig.freqz(b, a, worN=1000, fs=500)

axes[1].plot(w, np.abs(h), linewidth=2, color='orange')
axes[1].axvline(cutoff, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Frequency (Hz)', fontsize=10)
axes[1].set_ylabel('Gain', fontsize=10)
axes[1].set_title('REAL Low-Pass Filter (Smooth curve)', 
                  fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.2)

# Problem annotation
axes[1].annotate('Slightly tilted', 
                xy=(80, 0.95), xytext=(30, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')

axes[1].annotate('Gradual rolloff\n(slow cutoff)', 
                xy=(cutoff, 0.707), xytext=(200, 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

plt.tight_layout()
plt.show()

# Special property of Sinc
t = np.linspace(-10, 10, 1000)
sinc = np.sinc(t)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Sinc function
axes[0, 0].plot(t, sinc, linewidth=2)
axes[0, 0].axhline(0, color='black', linewidth=0.5)
axes[0, 0].set_ylabel('sinc(t)', fontsize=10)
axes[0, 0].set_title('Time Domain: Sinc', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Feature annotation
axes[0, 0].annotate('Main Lobe\n(center peak)', 
                   xy=(0, 1), xytext=(2, 0.8),
                   arrowprops=dict(arrowstyle='->', color='blue'),
                   fontsize=9, color='blue')

axes[0, 0].annotate('Side Lobes\n(side ripples)', 
                   xy=(2, 0.1), xytext=(4, 0.4),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=9, color='red')

# 2. Sinc frequency (conceptual)
freq = np.linspace(-5, 5, 1000)
rectangle = np.ones_like(freq)
rectangle[np.abs(freq) > 1] = 0

axes[0, 1].plot(freq, rectangle, linewidth=3, color='green')
axes[0, 1].set_ylabel('Magnitude', fontsize=10)
axes[0, 1].set_title('Frequency Domain: Rectangle!', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-0.1, 1.2)

axes[0, 1].annotate('Perfect rectangle!', 
                   xy=(0, 0.5), xytext=(2, 0.7),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   fontsize=11, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 3. Why? - Frequency decomposition
axes[1, 0].text(0.5, 0.8, 'Sinc = Sum of limited frequencies', 
               ha='center', fontsize=11, fontweight='bold',
               transform=axes[1, 0].transAxes)

# Multiple frequencies
for i, f in enumerate([1, 3, 5, 7, 9]):
    y_pos = 0.6 - i * 0.12
    axes[1, 0].text(0.5, y_pos, f'{f} Hz: included [check]', 
                   ha='center', fontsize=10,
                   transform=axes[1, 0].transAxes,
                   color='green')

axes[1, 0].text(0.5, 0.05, 'High freq (>cutoff): none [X]', 
               ha='center', fontsize=10, color='red', fontweight='bold',
               transform=axes[1, 0].transAxes)
axes[1, 0].axis('off')

# 4. Conclusion
axes[1, 1].text(0.5, 0.7, 'Conclusion:', ha='center', fontsize=13, fontweight='bold',
               transform=axes[1, 1].transAxes)
axes[1, 1].text(0.5, 0.5, 'Sinc contains only frequencies\nbelow certain cutoff!', 
               ha='center', fontsize=11,
               transform=axes[1, 1].transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[1, 1].text(0.5, 0.2, '-> Perfect Low-Pass Filter!', 
               ha='center', fontsize=12, fontweight='bold', color='green',
               transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.suptitle('Why Sinc <-> Rectangle?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. 좁은 시간 → 넓은 주파수
t_narrow = np.linspace(-10, 10, 1000)
time_narrow = np.exp(-t_narrow**2 / (2 * 0.1**2))  # 매우 좁음

axes[0, 0].plot(t_narrow, time_narrow, linewidth=2)
axes[0, 0].set_ylabel('Amplitude', fontsize=10)
axes[0, 0].set_title('Time: Very Narrow', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

from scipy.fft import fft, fftshift, fftfreq
fft_narrow = np.abs(fftshift(fft(time_narrow)))
freq_narrow = fftshift(fftfreq(len(t_narrow), t_narrow[1] - t_narrow[0]))

axes[0, 1].plot(freq_narrow, fft_narrow, linewidth=2, color='red')
axes[0, 1].set_ylabel('Magnitude', fontsize=10)
axes[0, 1].set_title('Frequency: Very Wide!', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(-2, 2)

# 2. 넓은 시간 → 좁은 주파수
time_wide = np.exp(-t_narrow**2 / (2 * 2**2))  # 매우 넓음

axes[1, 0].plot(t_narrow, time_wide, linewidth=2)
axes[1, 0].set_xlabel('Time', fontsize=10)
axes[1, 0].set_ylabel('Amplitude', fontsize=10)
axes[1, 0].set_title('Time: Very Wide', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

fft_wide = np.abs(fftshift(fft(time_wide)))
axes[1, 1].plot(freq_narrow, fft_wide, linewidth=2, color='red')
axes[1, 1].set_xlabel('Frequency', fontsize=10)
axes[1, 1].set_ylabel('Magnitude', fontsize=10)
axes[1, 1].set_title('Frequency: Very Narrow!', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(-2, 2)

plt.suptitle('Time-Frequency Trade-off', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
