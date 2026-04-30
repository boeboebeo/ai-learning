from scipy.special import jv

def analyze_negative_frequencies(f_c, f_m, I):
    """음수 주파수 분석"""
    print(f"Carrier: {f_c} Hz, Modulator: {f_m} Hz, Index: {I}")
    print(f"Maximum safe index: {f_c/f_m:.2f}\n")
    
    negative_count = 0
    
    for n in range(-20, 21):
        freq = f_c + n * f_m
        amp = abs(jv(n, I))
        
        if amp > 0.01:  # 의미있는 sideband만
            if freq < 0:
                negative_count += 1
                reflected_freq = abs(freq)
                print(f"  n={n:3d}: {freq:7.1f} Hz → reflects to {reflected_freq:6.1f} Hz (amp={amp:.3f})")
    
    if negative_count > 0:
        print(f"\n⚠️  {negative_count}개의 음수 주파수 발생!")
    else:
        print("\n✓ 모든 주파수가 양수입니다.")

# 테스트
analyze_negative_frequencies(f_c=100, f_m=40, I=4)
print("\n" + "="*60 + "\n")
analyze_negative_frequencies(f_c=440, f_m=50, I=5)