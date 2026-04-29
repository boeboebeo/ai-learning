"""jacobi-anger 전개로 알 수 있는것들 

1. ✅ Sideband 개수 
   → |J_n(I)| > 0.01 인 것들만 세면 됨
   → I 값이 클수록 많아짐

2. ✅ Sideband 위치
   → f_c ± n*f_m (정확히 계산 가능)

3. ✅ Sideband 레벨
   → A * |J_n(I)| (정확한 진폭)

4. ✅ Sideband 분포 패턴
   → Bessel 함수 그래프로 시각화 가능

5. ✅ 전체 스펙트럼 모양
   → 어떤 소리가 날지 예측 가능

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

f_c = 440
f_m = 50

# 세 가지 modulation index
mod_indices = [1, 2.4, 5]
colors = ['b', 'r', 'g']

for idx, (I, color) in enumerate(zip(mod_indices, colors)):
    # n = -10 to +10 까지의 sideband
    n_vals = np.arange(-10, 11)
    frequencies = f_c + n_vals * f_m
    amplitudes = [abs(jv(n, I)) for n in n_vals]
    
    # 스펙트럼 그리기
    axes[idx].stem(frequencies, amplitudes, linefmt=f'{color}-', 
                   markerfmt=f'{color}o', basefmt='gray')
    axes[idx].axvline(f_c, color='black', linestyle='--', 
                      alpha=0.5, label='Carrier freq')
    axes[idx].set_ylabel('Amplitude')
    axes[idx].set_title(f'Modulation Index I = {I}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, 900)
    
    # 몇 개의 sideband가 의미있는지 카운트
    significant = sum(1 for amp in amplitudes if amp > 0.01)
    print(f"I = {I}: {significant}개의 의미있는 sideband")

axes[2].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

"""jacobi-Anger 전개를 쓰고 있는 부분 

sin(2π*f_c*t + I*sin(2π*f_m*t)) = Σ J_n(I) * sin(2π*(f_c + n*f_m)*t)
                                   ^^^^^^
                                   이게 각 sideband의 진폭!

    => amplitudes = [abs(jv(n, I)) for n in n_vals] 
    : 여기의 jv(n, I) 가 n번째 sideband 의 진폭을 계산하는 jacobi anger 전개부분 

* 아래의 전개식이 자코비 전개 . 그리고 그 각 항의 계수인 J_n(β)가 Bessel 함수

    sin(x + β*sin(y)) = J_0(β)*sin(x)
                    + J_1(β)*sin(x+y) + J_{-1}(β)*sin(x-y)
                    + J_2(β)*sin(x+2y) + J_{-2}(β)*sin(x-2y)
                    + J_3(β)*sin(x+3y) + J_{-3}(β)*sin(x-3y)
                    + ...

피자를 자르는 방법 (Jacobi-Anger)
vs
각 조각의 크기 (Bessel)

"""