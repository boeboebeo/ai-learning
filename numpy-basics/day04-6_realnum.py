import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
fs = 1000  # 샘플링 주파수 (Hz)
duration = 2  # 신호 길이 (초)
f = 5  # 주파수 (Hz)

t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 두 신호 생성
signal_plus = np.sin(2 * np.pi * f * t)
signal_minus = np.sin(-2 * np.pi * f * t)

# FFT 계산
fft_plus = np.fft.fft(signal_plus)
fft_minus = np.fft.fft(signal_minus)
freqs = np.fft.fftfreq(len(t), 1/fs)

# 그래프 그리기
fig, axes = plt.subplots(3, 2, figsize=(10, 8))

# 시간 도메인 - 전체
axes[0, 0].plot(t, signal_plus, label='sin(+2πft)', linewidth=1.5)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('sin(+2πft) - Full view')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(t, signal_minus, label='sin(-2πft)', color='orange', linewidth=1.5)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_title('sin(-2πft) - Full view')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 시간 도메인 - 확대 (처음 1초)
zoom_samples = int(fs * 0.2)
axes[1, 0].plot(t[:zoom_samples], signal_plus[:zoom_samples], label='sin(+2πft)', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].set_title('sin(+2πft) - Zoomed')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(t[:zoom_samples], signal_minus[:zoom_samples], label='sin(-2πft)', color='orange', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].set_title('sin(-2πft) - Zoomed')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# 복소 FFT - 양수/음수 주파수 모두 표시
axes[2, 0].stem(freqs, np.abs(fft_plus), basefmt=' ')
axes[2, 0].set_xlabel('Frequency (Hz)')
axes[2, 0].set_ylabel('Magnitude')
axes[2, 0].set_title('FFT of sin(+2πft)')
axes[2, 0].set_xlim(-20, 20)
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].axvline(f, color='red', linestyle='--', alpha=0.5, label=f'+{f} Hz')
axes[2, 0].axvline(-f, color='blue', linestyle='--', alpha=0.5, label=f'-{f} Hz')
axes[2, 0].legend()

axes[2, 1].stem(freqs, np.abs(fft_minus), basefmt=' ', linefmt='orange', markerfmt='o')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_ylabel('Magnitude')
axes[2, 1].set_title('FFT of sin(-2πft)')
axes[2, 1].set_xlim(-20, 20)
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].axvline(f, color='red', linestyle='--', alpha=0.5, label=f'+{f} Hz')
axes[2, 1].axvline(-f, color='blue', linestyle='--', alpha=0.5, label=f'-{f} Hz')
axes[2, 1].legend()

plt.tight_layout()
plt.show()

# 수학적 설명 출력
print("=" * 60)
print("sin(+2πft) vs sin(-2πft) 비교")
print("=" * 60)
print(f"\n주파수: {f} Hz")
print(f"\n1. 시간 도메인:")
print(f"   sin(+2πft) 와 sin(-2πft) 는 시간축에서 반대 방향으로 회전")
print(f"   sin(-x) = -sin(x) 이므로 부호만 반대")
print(f"\n2. 복소 지수 표현:")
print(f"   sin(2πft) = (e^(j2πft) - e^(-j2πft)) / 2j")
print(f"   sin(-2πft) = (e^(-j2πft) - e^(j2πft)) / 2j = -sin(2πft)")
print(f"\n3. FFT 결과:")
print(f"   sin(+2πft): 피크가 +{f} Hz와 -{f} Hz에 나타남")
print(f"   sin(-2πft): 피크가 똑같이 +{f} Hz와 -{f} Hz에 나타남")
print(f"   → 크기는 같지만 위상이 반대 (180도 차이)")
print(f"\n4. 왜 같은 주파수로 보이는가?")
print(f"   실수 신호의 FFT는 대칭 (Hermitian symmetry)")
print(f"   음의 주파수 성분도 양의 주파수에 반영됨")
print(f"   sin()는 양/음 주파수 성분을 모두 가짐")
print("=" * 60)

# 복소수 FFT 위상 비교
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))

# 위상 계산 (음이 아닌 주파수만)
phase_plus = np.angle(fft_plus)
phase_minus = np.angle(fft_minus)

axes2[0].plot(freqs, phase_plus, label='sin(+2πft)', linewidth=2)
axes2[0].set_xlabel('Frequency (Hz)')
axes2[0].set_ylabel('Phase (radians)')
axes2[0].set_title('Phase of FFT: sin(+2πft)')
axes2[0].set_xlim(-20, 20)
axes2[0].grid(True, alpha=0.3)
axes2[0].axhline(0, color='black', linewidth=0.5)
axes2[0].legend()

axes2[1].plot(freqs, phase_minus, label='sin(-2πft)', color='orange', linewidth=2)
axes2[1].set_xlabel('Frequency (Hz)')
axes2[1].set_ylabel('Phase (radians)')
axes2[1].set_title('Phase of FFT: sin(-2πft)')
axes2[1].set_xlim(-20, 20)
axes2[1].grid(True, alpha=0.3)
axes2[1].axhline(0, color='black', linewidth=0.5)
axes2[1].legend()

plt.tight_layout()
plt.show()

print("\n위상 차이:")
idx_pos = np.argmin(np.abs(freqs - f))
idx_neg = np.argmin(np.abs(freqs + f))
print(f"  +{f} Hz에서 위상 차이: {phase_plus[idx_pos] - phase_minus[idx_pos]:.4f} rad")
print(f"  -{f} Hz에서 위상 차이: {phase_plus[idx_neg] - phase_minus[idx_neg]:.4f} rad")
print(f"  π (180도) = {np.pi:.4f} rad")