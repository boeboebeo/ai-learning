import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 서로 다른 탭 개수
numtaps_values = [11, 51, 101]

plt.figure(figsize=(12, 8))

for i, numtaps in enumerate(numtaps_values):
    # 필터 설계
    coeff = signal.firwin(numtaps, 0.3)
    
    # 주파수 응답 계산
    w, h = signal.freqz(coeff, worN=8000)
    
    # 계수 그래프
    plt.subplot(3, 2, i*2 + 1)
    plt.stem(coeff, basefmt=' ')
    plt.title(f'Filter Coefficients (numtaps={numtaps})')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 주파수 응답 그래프
    plt.subplot(3, 2, i*2 + 2)
    plt.plot(w/np.pi, np.abs(h), linewidth=2)
    plt.axvline(0.3, color='red', linestyle='--', label='Cutoff')
    plt.title(f'Frequency Response (numtaps={numtaps})')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()