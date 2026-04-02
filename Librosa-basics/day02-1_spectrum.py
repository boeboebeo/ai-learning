# day 02 spectrum

import librosa 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

y, sr = librosa.load("audio_sample/noise.wav")

# ── 1. 평균 스펙트럼 (배음 구조 보기) ────────────────────────
D = np.abs(librosa.stft(y))  #그럼 여기서는 전체 시간구간에서의 stft한 것(복소수)을 진폭 스펙트럼(절대값)으로 나타냄
                             #결과는 shape(n_fft//2+1, n_frames)
mean_spectrum = np.mean(D, axis=1)  #전체시간의 평균 스펙트럼을 보여줌 => 음원 내의 평균적인 배음구조 를 보는데 유용
                                    #axis = 1 -> 시간 프레임 방향 평균
freqs = librosa.fft_frequencies(sr=sr)  #
print(freqs)
""".fft_frequencies(sr=sr)
: stft(이산 푸리에 변환)에서 계산된 주파수 bin 이 실제 Hz단위로 어떤 주파수를 의미하는지 알려주는 함수
=> 출력 배열은 [0, 1, 2, ..., N/2] 이런식의 audio bin index 형태로 나옴
    => fft_frequencies 가 bin index -> Hz 변환해주는 역할

+ print(freqs)의 결과는 [0.00000000e+00 1.07666016e+01 2.15332031e+01 ... 
    1.10034668e+04 ... 1.10142334e+04 1.10250000e+04]
    + 여기서 뒤의+04는 지수표기(*10^4)를 하면 11025.0000이 되게 되는것임
    => 이런식으로 0Hz 부터 1102Hz 까지를 전체 audio bin 1024개 까지로 나눔(Nyquist)
"""

plt.figure(figsize=(8, 8))
plt.plot(freqs, mean_spectrum, color="#4A90D9", linewidth=1)   #(x축, y축)
plt.xscale("log") #얘때문에 x축 freq 가 10^2, 10^3, 10^4 (이렇게 보이게 됨 -> logarithmic)
                  #linear하게 바꾸고 싶다면 "linear"
plt.xlabel("freq(Hz)")
plt.ylabel("amp")
plt.title("average spectrum - harmonics structure")
plt.xlim(20, sr//2) #x축 범위를 설정 (20Hz ~ sr의 1/2 지점까지)
plt.grid(alpha=0.3) #그래프에 격자 추가 (alpha=0.3) : 투명도(0~1)
plt.tight_layout()
plt.savefig("day2_spectrum.png")
plt.show()


