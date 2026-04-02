import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ── 한글 폰트 설정 (Mac) ──────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ── 1. 음악 파일 로드 ──────────────────────────────────────────
audio_path = "/Users/goeun/Desktop/260303 airport 1.wav"   # mp3, wav, flac 모두 가능
y, sr = librosa.load(audio_path, sr=None) #파이썬의 다중 반환값, 받는 쪽에서 두 변수로 언패킹
# y  = 오디오 신호 (numpy 배열)
# sr = 샘플레이트 (기본 22050 Hz)

print(f"샘플레이트: {sr} Hz")
print(f"재생 길이: {len(y) / sr:.1f}초") #y = 샘플 배열

#ex. 30초 짜리 곡이면 22050 x 30 = 샘플 661,500개 들어있음


# ── 2. BPM / 템포 감지 ────────────────────────────────────────
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(f"\n🥁 BPM: {tempo[0]:.1f}")
print(f"   감지된 비트 수: {len(beats)}")


# ── 3. 음정 분석 (pyin) ────────────────────────────────────────
f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),   # 최저 감지 음 (C2 = 65Hz)
    fmax=librosa.note_to_hz("C7"),   # 최고 감지 음 (C7 = 2093Hz)
)

# Hz → 음이름 변환 (소리가 감지된 구간만)
notes = []
for freq in f0:
    if freq is not None and not np.isnan(freq):
        note = librosa.hz_to_note(freq)
        notes.append(note)

if notes:
    from collections import Counter
    top_notes = Counter(notes).most_common(5)
    print("\n🎵 가장 많이 나온 음정 Top 5:")
    for note, count in top_notes:
        print(f"   {note}: {count}번")


# ── 4. 파형 + 음정 + 스펙트로그램 시각화 ──────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("음악 분석 결과", fontsize=14, fontweight="bold")

# 파형
librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4A90D9")
axes[0].set_title(f"파형 (BPM: {tempo[0]:.1f})", fontsize=12)
axes[0].set_xlabel("시간 (초)")
axes[0].set_ylabel("진폭")
beat_times = librosa.frames_to_time(beats, sr=sr)
for bt in beat_times:
    axes[0].axvline(x=bt, color="orange", alpha=0.3, linewidth=0.8)

# 음정
times = librosa.times_like(f0, sr=sr)
axes[1].plot(times, f0, color="#E85D8A", linewidth=1, label="감지된 음정 (Hz)")
axes[1].set_title("음정 변화 (F0)", fontsize=12)
axes[1].set_xlabel("시간 (초)")
axes[1].set_ylabel("주파수 (Hz)")
axes[1].legend()
axes[1].set_ylim(0, 1000)

# 스펙트로그램 ← 새로 추가!
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=axes[2], cmap="magma")
axes[2].set_title("스펙트로그램", fontsize=12)
axes[2].set_xlabel("시간 (초)")
axes[2].set_ylabel("주파수 (Hz)")
fig.colorbar(img, ax=axes[2], format="%+2.0f dB")

plt.tight_layout()
plt.savefig("analysis_result.png", dpi=150)
plt.show()
print("\n✅ 그래프 저장 완료: analysis_result.png")


"""
함수                  -> 내부에서 하는 일 
librosa.load()       → 파일 디코딩 + 리샘플링 + 정규화
librosa.beat.beat_track() → onset detection + tempogram
librosa.pyin()       → probabilistic YIN 알고리즘
librosa.stft()       → Short-Time Fourier Transform
librosa.feature.mfcc() → Mel-Frequency Cepstral Coefficients
"""