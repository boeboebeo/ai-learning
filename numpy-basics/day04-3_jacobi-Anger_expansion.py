from scipy.special import jv
import numpy as np

# Jacobi-Anger 전개 직접 구현
def fm_via_jacobi_anger(t, A, f_c, f_m, I, max_n=10):
    """
    Jacobi-Anger 전개를 사용한 FM 합성
    """
    result = np.zeros_like(t)
    
    # Jacobi-Anger 공식 적용
    for n in range(-max_n, max_n+1):
        
        # 이 항의 계수 계산 (Bessel 함수!)
        coefficient = jv(n, I)  # ← 이게 Bessel 함수
        
        # 이 항의 주파수
        freq = f_c + n * f_m
        
        # 전개 결과에 이 항 추가
        # Jacobi-Anger: Σ J_n(I) * sin(2π*(f_c+n*f_m)*t)
        result += A * coefficient * np.sin(2*np.pi*freq*t)
    
    return result

# 사용
t = np.linspace(0, 1, 44100)
fm_signal = fm_via_jacobi_anger(t, A=1, f_c=440, f_m=100, I=3)