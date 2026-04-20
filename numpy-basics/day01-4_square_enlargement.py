import numpy as np

t = 1/8  # 0.125초
omega = 2 * np.pi

print("=" * 70)
print("t = 1/8초 = 0.125초 (ωt = π/4 = 45도 지점)")
print("=" * 70)
print("n  |  각도(rad) |  각도(°) |  sin값  |  나누기 n  ")
print("---|-----------|---------|---------|------------")

sum_val = 0
for k in range(1, 10):
    n = 2*k - 1
    angle_rad = n * omega * t
    angle_deg = np.degrees(angle_rad)
    sin_val = np.sin(angle_rad)
    divided = sin_val / n
    sum_val += divided
    
    symbol = "✓" if divided > 0 else "✗"
    print(f"{n:2d} | {angle_rad:9.2f} | {angle_deg:7.0f} | {sin_val:7.4f} | {divided:10.4f}  {symbol}")

print(f"\n합계 (9개) = {sum_val:.4f}")
print(f"4/π 곱하면 = {(4/np.pi) * sum_val:.4f}")

# 무한대로 더하기
sum_infinite = sum(np.sin((2*k-1) * omega * t) / (2*k-1) for k in range(1, 10000))
scaled_infinite = (4/np.pi) * sum_infinite

print(f"\n합계 (무한대, 9999개) = {sum_infinite:.10f}")
print(f"4/π 곱하면 = {scaled_infinite:.10f} => 거의 1.0!")
print(f"    => 무한대로 더하면 1.0에 수렴!")
print(f"    => 9개만 더하면 {(4/np.pi) * sum_val:.4f}로 부족했던 것!")
print("=" * 70)