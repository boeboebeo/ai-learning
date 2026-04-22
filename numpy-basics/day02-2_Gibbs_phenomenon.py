import numpy as np
import matplotlib.pyplot as plt

FREQ = 1
t = np.linspace(0, 2, 10000)
omega = 2 * np.pi * FREQ

fig, axes = plt.subplots(4, 2, figsize=(8, 8))

# 다양한 배음 개수
num_harmonics_list = [3, 5, 10, 50]

for idx, num_harmonics in enumerate(num_harmonics_list):
    # Square wave 생성
    square = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        n = 2*k - 1  # 홀수
        square += (4/np.pi) * np.sin(n * omega * t) / n
    
    # 시간 도메인
    axes[idx, 0].plot(t, square, 'b-', linewidth=1)
    axes[idx, 0].axhline(y=1, color='red', linestyle='--', 
                         linewidth=0.8, alpha=0.5, label='Target: ±1')
    axes[idx, 0].axhline(y=-1, color='red', linestyle='--', 
                         linewidth=0.8, alpha=0.5)
    
    # Overshoot 찾기
    peak = np.max(square)
    valley = np.min(square)
    overshoot_percent = (peak - 1.0) * 100
    
    axes[idx, 0].axhline(y=peak, color='orange', linestyle=':', 
                         linewidth=1.5, alpha=0.7)
    axes[idx, 0].text(0.7, peak -0.3, 
                     f'Peak: {peak:.3f}\nOver: {overshoot_percent:.1f}%',
                     ha='center', fontsize=6,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    axes[idx, 0].set_title(f'{num_harmonics} harmonics - Gibbs', 
                          fontsize=8, fontweight='bold')
    axes[idx, 0].set_ylabel('Amp', fontsize=7)
    axes[idx, 0].set_ylim(-1.5, 1.5)
    axes[idx, 0].grid(True, alpha=0.3)
    axes[idx, 0].legend(fontsize=5, loc='lower right')
    axes[idx, 0].tick_params(labelsize=6)
    
    # 불연속점 확대
    mask = (t > 0.35) & (t < 0.55)
    t_zoom = t[mask]
    square_zoom = square[mask]
    
    axes[idx, 1].plot(t_zoom, square_zoom, 'b-', linewidth=1.5)
    axes[idx, 1].axhline(y=1, color='red', linestyle='--', 
                         linewidth=0.8, alpha=0.5)
    axes[idx, 1].axvline(x=0.5, color='green', linestyle='--', 
                         linewidth=0.8, alpha=0.5, label='Discontinuity')
    axes[idx, 1].axhline(y=peak, color='orange', linestyle=':', 
                         linewidth=1.5, alpha=0.7)
    
    axes[idx, 1].set_title(f'Zoom - Overshoot clear', 
                          fontsize=8, fontweight='bold')
    axes[idx, 1].set_ylabel('Amp', fontsize=7)
    axes[idx, 1].set_ylim(-0.5, 1.3)
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].legend(fontsize=5, loc='upper right')
    axes[idx, 1].tick_params(labelsize=6)

axes[-1, 0].set_xlabel('Time (s)', fontsize=7)
axes[-1, 1].set_xlabel('Time (s)', fontsize=7)

plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0) #파라미터 추가
# plt.savefig('/mnt/user-data/outputs/gibbs_phenomenon.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== Gibbs Phenomenon Analysis ===\n")

for num in [3, 5, 10, 50, 100, 500]:
    square = np.zeros_like(t)
    for k in range(1, num + 1):
        n = 2*k - 1
        square += (4/np.pi) * np.sin(n * omega * t) / n
    
    peak = np.max(square)
    overshoot = (peak - 1.0) * 100
    
    print(f"{num:3d} harmonics: Peak = {peak:.6f}, Overshoot = {overshoot:.3f}%")

print("\n→ Overshoot converges to ~9% even with more harmonics!")
print("→ Never disappears! (Gibbs discovery)")

