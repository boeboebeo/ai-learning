# square - fourier series 의 4/pi 의 존재 이유

import numpy as np


# t = π/2 (최댓값 지점. pi/2 = 90도)
t = np.pi / 2

# 4/π 없이 harmonics 더하기
sum_without = 0
for k in range(1, 1000):  # 999개 harmonic
    n = 2*k - 1  # 홀수
    sum_without += np.sin(n * t) / n  


print(f"4/π 없이: {sum_without:.4f}")
print(f"4/π 곱함: {(4/np.pi) * sum_without:.4f}")

# 출력:
# 4/π 없이: 0.7854  (≈ π/4) 
    # sin sum = 1/1 - 1/3 + 1/5 - 1/7 + 1/9 ... = pi/4 (Leibniz. 라이프니츠 공식)
# 4/π 곱함: 1.0000

