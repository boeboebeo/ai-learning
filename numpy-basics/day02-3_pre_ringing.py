# 이건 그냥 ringing 나와서 본건데, 너무 어려움 아직... 나중에 다시 보자

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

impulse = np.zeros(1000)
impulse[500] = 1.0

# (1) Linear Phase FIR
fir_linear = signal.firwin(201, 0.2, window='hamming')
filtered_linear = signal.lfilter(fir_linear, 1.0, impulse)

axes[0, 0].plot(range(300, 700), filtered_linear[300:700], 'b-', linewidth=1.5)
axes[0, 0].axvline(x=500, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].axvspan(400, 500, alpha=0.2, color='yellow', label='Pre')
axes[0, 0].axvspan(500, 600, alpha=0.2, color='cyan', label='Post')
axes[0, 0].set_title('Linear Phase FIR\n(Symmetric)', 
                    fontsize=9, fontweight='bold')
axes[0, 0].set_ylabel('Amp', fontsize=7)
axes[0, 0].legend(fontsize=6, loc='upper right')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(labelsize=6)

# (2) Minimum Phase (개념)
axes[0, 1].plot(range(300, 700), filtered_linear[300:700], 'g-', 
               linewidth=1.5, alpha=0.5)
axes[0, 1].axvline(x=500, color='red', linestyle='--', linewidth=1.5)
axes[0, 1].text(500, 0.003, 'Min phase:\nNo pre-ring\nCausal', 
               ha='center', fontsize=7, 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
axes[0, 1].set_title('Minimum Phase\n(Causal)', 
                    fontsize=9, fontweight='bold')
axes[0, 1].set_ylabel('Amp', fontsize=7)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(labelsize=6)

# (3) Gibbs
t = np.linspace(0.4, 0.6, 1000)
omega = 2 * np.pi * 5
square = np.zeros_like(t)
for k in range(1, 20):
    n = 2*k - 1
    square += (4/np.pi) * np.sin(n * omega * t) / n

axes[1, 0].plot(t, square, 'b-', linewidth=1.5)
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=1.5)
axes[1, 0].axhline(y=1, color='orange', linestyle=':', linewidth=1.5)
axes[1, 0].text(0.5, 1.08, 'Gibbs\novershoot', 
               ha='center', fontsize=7,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
axes[1, 0].set_title('Gibbs Phenomenon\n(Symmetric)', 
                    fontsize=9, fontweight='bold')
axes[1, 0].set_xlabel('Time', fontsize=7)
axes[1, 0].set_ylabel('Amp', fontsize=7)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(labelsize=6)

# (4) 비교 표
comparison_text = """Linear Phase:
✓ Pre-ring: YES
✓ Post-ring: YES
✓ Phase: NONE
✗ Delay: YES

Min Phase:
✓ Pre-ring: NO
✓ Post-ring: MORE
✗ Phase: YES
✓ Causal: TRUE

Gibbs:
✓ Both sides
✓ Symmetric
- Synthesis"""

axes[1, 1].text(0.1, 0.5, comparison_text, 
               fontsize=7, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Comparison', fontsize=9, fontweight='bold')

plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
# plt.savefig('/mnt/user-data/outputs/ringing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()