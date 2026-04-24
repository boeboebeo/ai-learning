import numpy as np
import matplotlib.pyplot as plt

sr = 22050
t = np.linspace(0, 1, sr)

fc = 440   # carrier
fm = 50     # modulator
m = 0.8    # modulation index

carrier = np.sin(2 * np.pi * fc * t)
modulator = 1 + m * np.sin(2 * np.pi * fm * t)

y = carrier * modulator

# 시간 영역
plt.figure()
plt.plot(t[:2000], y[:2000])
plt.title("AM waveform (time domain)")
plt.show()

# FFT
Y = np.fft.fft(y)
freq = np.fft.fftfreq(len(Y), 1/sr)

plt.figure()
plt.plot(freq[:2000], np.abs(Y[:2000]))
plt.title("AM spectrum")
plt.show()