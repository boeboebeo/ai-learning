#day12-2 numpy + matplotlib.pyplot

import numpy as np
import matplotlib.pyplot as plt

fs = 44100
t = np.linspace(0, 0.01, fs) 
    # 여기 duration (가운데 자리)가 1초라면 너무 빽빽해서 시각적으로 잘 안보임
    # 시간 단위 줄여서 파형 잘보이게 duration 줄임

audio_signal = np.sin(2*np.pi*440*t)

c = np.sin(2*np.pi*261.63*t)
e = np.sin(2*np.pi*329.63*t)
g = np.sin(2*np.pi*392.00*t)
chord = (c+e+g) / 3

    # plt.plot(x, y): x->가로축 / y->세로축
    # 여기서는 각 순간의 진폭이 audio_signal 이기 때문에!!!! y 축 자리에 audio_signal을 넣고 x축 자리에는 시간을 넣는것
    # if, plt.plot(audio_signal) 만 넣는경우 x축이 샘플 인덱스가 되어서 약간 덜 직관적이다 -> 걍 샘플 순서대로 그리는것 . 
        # 그래서 각 t 이 일정하지 않으면 파형의 모양이 달라짐
plt.plot(t, audio_signal)
plt.plot(t, chord)
    # => plt.plot(audio_signal), plt.plot(t, chord) -> 이렇게 다르게 넣게되면 둘의 x축 기준이 다르기 때문에 하나가 0근처의 얇은 선처럼 보임

plt.plot(t, audio_signal, label="Sine")
plt.plot(t, chord, label="Chord")

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("440Hz Sine wave + C Major chord waveform")
plt.show()
plt.legend() #그래프 한쪽에 작은 박스가 생겨서 이름표 붙이는 기능

# -1 ~ +1 까지의 진폭은 최대 음압 범위를 정규화(normalization) 한 것
# -1.0 ~ 1.0   (float) / -32768 ~ 32767   (int16)