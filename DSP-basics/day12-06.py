import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Symmetric FIR
# b = np.array([0.2, 0.5, 1.0, 0.5, 0.2])
# Asymmetric
b = np.array([0.2, 0.7, 1.0, 0.5, 0.9])
    # 계수만 바꿔도 

w, h = signal.freqz(b, [1], worN=1024)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(w/np.pi, np.abs(h))
plt.title("Magnitude")

plt.subplot(1,2,2)
plt.plot(w/np.pi, np.unwrap(np.angle(h)))
plt.title("Phase")

plt.tight_layout()
plt.show()