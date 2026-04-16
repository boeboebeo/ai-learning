# centroid . 
    # 1. 음색 분류, 
    # 2. Vibrato 감지 (Centroid 가 주기적으로 변하면 Vibrato)
    # 3. Filter 감지 (Centroid 가 시간에 따라 올라가면 -> High pass filter)


import librosa
import numpy as np
import matplotlib.pyplot as plt

# 오디오 로드
y, sr = librosa.load("Librosa-basics/audio_sample_percvswave/snare.wav")

# Spectral Centroid 계산
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
times = librosa.times_like(centroid, sr=sr, hop_length=256)

# 그래프
plt.figure(figsize=(14, 5))
plt.plot(times, centroid, linewidth=2, color='green')
plt.title("Spectral Centroid")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.grid(alpha=0.3)
plt.show()

# 통계
print(f"평균 Centroid: {np.mean(centroid):.1f} Hz")
print(f"최소 Centroid: {np.min(centroid):.1f} Hz")
print(f"최대 Centroid: {np.max(centroid):.1f} Hz")


# Zero Crossing Rate (ZCR)
    # : 0의 교차하는 빈도수
    # 1. 노이즈 감지
    # 2. 음성 / 음악 구분 
    # 부호가 바뀌는 횟수 / 전체 샘플 수 


# ZCR 계산
zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=512)[0]
times = librosa.times_like(zcr, sr=sr, hop_length=512)

# 그래프
plt.figure(figsize=(14, 5))
plt.plot(times, zcr, linewidth=2, color='purple')
plt.title("Zero Crossing Rate (0 cross)")
plt.ylabel("ZCR")
plt.xlabel("Time (s)")
plt.grid(alpha=0.3)
plt.show()

# 통계
print(f"평균 ZCR: {np.mean(zcr):.4f}")
print(f"최소 ZCR: {np.min(zcr):.4f}")
print(f"최대 ZCR: {np.max(zcr):.4f}")


# 1. 소리 분류
if np.mean(zcr) < 0.05:
    print("톤 악기 (피아노, 기타)")
elif np.mean(zcr) < 0.15:
    print("타악기 (드럼)")
else:
    print("노이즈 (심벌, 하이햇, 화이트 노이즈)")

# 2. 유성음/무성음 구분 (음성 인식)
if np.mean(zcr) < 0.1:
    print("유성음 (모음: 아, 에, 이)")
else:
    print("무성음 (자음: ㅅ, ㅎ, ㅊ)")

# 3. 음악 vs 대화 구분
if np.mean(zcr) < 0.08:
    print("음악")
else:
    print("대화 (말소리)")