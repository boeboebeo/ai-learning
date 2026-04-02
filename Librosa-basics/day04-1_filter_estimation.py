#day 04 filter estimation  
    # 이 코드 다시 고쳐서 깃헙 올리기 

import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

y, sr = librosa.load("audio_sample/saw+LPF.wav")

# 필터 추정

centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

print(f"\n스펙트럼 무게중심 : {centroid:.0f}Hz <- 필터 밝기")
print(f"롤오프 주파수: {rolloff:.0f}Hz <- 에너지 85% 기준")
print(f"대역폭: {bandwidth:.0f}Hz <- 넓을수록 bright")


if centroid < 800:
    filter_hint = "LPF 많이 닫힘"
elif centroid < 2000:
    filter_hint = "LPF 중간"
elif centroid < 5000:
    filter_hint = "LPF 거의 열림"
else :
    filter_hint = "거의 열림 or HPF"

print(f"filter estimation : {filter_hint}")


"""
librosa.feature.spectral_ *** : STFT -> magnitude spectrum -> 거기에서 특징 추출하는 함수들

1. spectral_centroid 
 : 소리의 무게중심을 측정 - 더 큰 magnitude 에 가중치를 주어 계산함
 More precisely
  : centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t]). 시간마다 centroid 를 계산 => centroid = [t1, t2, t3 ...]를 np.mean(...)으로 평균 낸 것
  (where S is a magnitude spectrogram, 
  and freq is the array of frequencies (e.g., FFT frequencies in Hz) of the rows of S.)

    - 필터 전(saw) : 전 주파수 대역 골고루 퍼짐 : centroid = 중간쯤
    - LPF 후(saw) : 에너지가 아래쪽에 몰림 : centroid = 내려감
    => 근데 원래 centroid 가 높거나, 내려가 있는 파형도 있어서 (ex. sine, noise)이것만으로는 구분 어려움
        => 그래서 사용하는게 rolloff, bandwith 

2. spectral_rolloff (0.85 by default)
 : 전체 에너지의 X%가 쌓이는 지점의 주파수 (보통 x= 0.85. 85%)
    s[k] = 각 주파수 bin 의 에너지
    freq[k] = 그 주파수
    total = Σ S[k] , cumsum[k] = S[0] + S[1] + ... + S[k]
    cumsum[k] >= 0.85 x total (이걸 처음 만족하는 k 찾기)
        =>rolloff = freq[k]

        ex. freq:   100  200  300  400
            energy:  10   8    2    1   
            => 여기서 낮은 주파수부터 누적해서 합을 봄 . 전체의 85%가 처음 넘는 지점 -> 그 주파수가 roll off
            total = 10 + 8 + 2 + 1 = 21 
            85% = 21 * 0.85 = 17.85 인데, 그건 100Hz + 200Hz 만 해도 넘어감 
            => roll off = 200Hz 
            => 대부분의 소리에너지는 200hz 이하에 있다는 뜻 => LPF 느낌 = 고주파가 얼마나 살아있냐를 측정하는게 rolloff의 역할

=> 해당 프레임의 스펙트럼에서 전체 에너지의 최소 roll_percent (기본값 0.85, 즉 85%)가
그 주파수 bin과 그보다 낮은 주파수 bin들에 포함되도록 하는 그 bin의 중심 주파수를 의미한다.
    - roll_percent -> 1.0에 가까움 : 거의 전체에너지 , 최대 주파수 근사
    - roll_percent -> 0.0에 가까움 : 아주 초반에너지만, 최소 주파수 근사

    print(librosa.feature.spectral_rolloff(y=y, sr=sr))
     : 시간마다 roll off 를 계산하기 때문에 배열이 출력됨
      + spectral_*의 모든 함수는 프레임 단위로 계산하기때문에 -> 시간에 따라 여러값 표현


3. spectral_bandwidth 
: 에너지가 얼마나 퍼져있는가 (width). 분산의 루트 = 표준편차
    => 집중됨 : bandwidth 작음
    => 퍼짐 : bandwidth 넓음
     ex. saw : bandwidth 큼
         noise : bandwidth 매우 큼
         sine : bandwidth 매우 작음


    bandwidth(t) = (sum_k S[k, t] * (freq[k, t] - centroid[t])**p)**(1/p)
    bandwidth(t) = sqrt( Σ_k S[k,t] * (freq[k] - centroid(t))² / Σ_k S[k,t] )
    + freq[k] - centroid : 중심에서 얼마나 떨어졌냐
    + ^2 = 멀수록 더 크게 반영
    + S[k, t] : 에너지 큰 애가 더 영향이 큼
    => 각 주파수가 중심(centroid)에서 얼마나 떨어져 있는지 평균 내는 것

        centroid = 중심주파수
        freq = centroid : 각 주파수 마다 (freq) 중심으로 (centroid)얼마나 떨어져 있는지 거리 계산
        (distance)^2 = 멀리있는 건 더 크게 반영하려고 제곱 
        s[k] * distance^2 = 소리 큰 주파수 일 수록 더 중요하도록 -> 제곱한거 곱해버림
        Σ S[k] * distance^2 = 다 더하기
        / Σ S[k] => 전체 에너지 로 나눠서 scale 제거
        sqrt(...) : 다시 거리단위로 돌려놓기 
            +제곱하는 이유: 음수제어, 멀리 있는 값 강조
                ex. 10차이나면 100, 50차이나면 2500 으로 훨씬 더 크게 반영된다
 
"""

# centroid, rolloff, bandwidth 뭔지 공부하고 
# low pass -freq , resonance 계산
# LPF, HPF, BPF 구분 프로젝트