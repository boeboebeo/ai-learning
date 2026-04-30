import numpy as np
import matplotlib.pyplot as plt

fs = 100
t = np.linspace(0, 1, fs, endpoint=False)

# 순수 +20Hz만 있는 복소 신호 (불가능한 실제 신호)
signal_complex_plus = np.exp(1j * 2 * np.pi * 20 * t)

# 순수 -20Hz만 있는 복소 신호 (불가능한 실제 신호)
signal_complex_minus = np.exp(-1j * 2 * np.pi * 20 * t)

# 실수 신호: sin(2π·20·t)
signal_real_sin = np.sin(2 * np.pi * 20 * t)

# FFT
fft_plus = np.fft.fft(signal_complex_plus)
fft_minus = np.fft.fft(signal_complex_minus)
fft_sin = np.fft.fft(signal_real_sin)
freqs = np.fft.fftfreq(len(t), 1/fs)

# 그래프
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# 복소 신호: 순수 +20Hz
axes[0].stem(freqs, np.abs(fft_plus), basefmt=' ')
axes[0].set_title('Pure +20Hz (complex signal: e^(j2π·20·t))')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude')
axes[0].set_xlim(-50, 50)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(20, color='red', linestyle='--', alpha=0.5)
axes[0].axvline(-20, color='blue', linestyle='--', alpha=0.5)

# 복소 신호: 순수 -20Hz
axes[1].stem(freqs, np.abs(fft_minus), basefmt=' ')
axes[1].set_title('Pure -20Hz (complex signal: e^(-j2π·20·t))')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_xlim(-50, 50)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(20, color='red', linestyle='--', alpha=0.5)
axes[1].axvline(-20, color='blue', linestyle='--', alpha=0.5)

# 실수 신호: sin(20Hz)
axes[2].stem(freqs, np.abs(fft_sin), basefmt=' ')
axes[2].set_title('Real signal: sin(2π·20·t) = has BOTH +20Hz AND -20Hz')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Magnitude')
axes[2].set_xlim(-50, 50)
axes[2].grid(True, alpha=0.3)
axes[2].axvline(20, color='red', linestyle='--', alpha=0.5, label='+20 Hz')
axes[2].axvline(-20, color='blue', linestyle='--', alpha=0.5, label='-20 Hz')
axes[2].legend()

plt.tight_layout()
plt.show()

print("복소 신호 e^(j2π·20·t): +20Hz에만 피크")
print("복소 신호 e^(-j2π·20·t): -20Hz에만 피크")
print("실수 신호 sin(2π·20·t): +20Hz와 -20Hz 둘 다 피크!")
print("\n실수 세계에서는 +20Hz와 -20Hz를 분리할 수 없습니다.")
print("왜냐하면 실수 신호는 항상 양/음 주파수 쌍으로 존재하기 때문!")