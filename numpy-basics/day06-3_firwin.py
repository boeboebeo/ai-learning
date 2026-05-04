import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 입력: 사각파 (여러 주파수 성분 포함)
t = np.linspace(0, 0.01, 1000)
square = signal.square(2 * np.pi * 440 * t)

# 대칭 필터 (firwin은 항상 대칭)
symmetric_coeff = signal.firwin(51, 0.3)
output_sym = signal.lfilter(symmetric_coeff, 1.0, square)

# 비대칭 필터 (억지로 만듦)
asymmetric_coeff = symmetric_coeff * np.linspace(0.5, 1.5, 51)
output_asym = signal.lfilter(asymmetric_coeff, 1.0, square)

# 그래프
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t*1000, square)
plt.title('Original Square Wave')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t*1000, output_sym)
plt.title('Symmetric Filter (Shape Preserved)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t*1000, output_asym, color='red')
plt.title('Asymmetric Filter (Shape Distorted)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()