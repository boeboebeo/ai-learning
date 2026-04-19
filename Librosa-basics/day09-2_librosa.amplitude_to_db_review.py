# day09-2_librosa.amplitude_to_db_review

import librosa
import numpy as np

y, sr = librosa.load("Librosa-basics/audio_sample/noise.wav")

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

print(np.max(D)) # 0 -> ref 가 0으로 자동 정규화 됨
print(D[0, :3])
    # [-5.532219 -3.779564 -9.972975]

