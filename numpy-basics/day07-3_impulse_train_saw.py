# impulse 를 leaky 적분해야 sawtooth 가 됨

import numpy as np
import matplotlib.pyplot as plt

# Impulse train
impulse_train = np.zeros(200)
impulse_train[::40] = 1  # 40개마다 impulse

# 적분
integral = np.cumsum(impulse_train)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Impulse train
axes[0].stem(impulse_train, basefmt=' ')
axes[0].set_ylabel('Impulse Train')
axes[0].set_title('Impulse')
axes[0].grid(True)

# 적분
axes[1].plot(integral, 'o-', markersize=2)
axes[1].set_xlabel('sample')
axes[1].set_ylabel('integration')
axes[1].set_title('integration → steps!')
axes[1].grid(True)

plt.tight_layout()
plt.show()



"""
1. Band-Limited Impulse 생성 (양수만)
2. Leaky Integrator로 적분:
   
   y[n] = leak * y[n-1] + impulse[n]
   
3. Leak이 자동으로:
   - 올림: impulse로
   - 내림: leak으로 감소
   
4. 결과: Sawtooth!
"""