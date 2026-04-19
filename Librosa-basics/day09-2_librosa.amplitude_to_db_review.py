# day09-2_librosa.amplitude_to_db_review

import librosa
import numpy as np

y, sr = librosa.load("Librosa-basics/audio_sample/noise.wav")

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    #np.abs(librosa.stft(y)) : magnitude 값만 불러오기

# print(np.max(D)) # 0 -> ref 가 0으로 자동 정규화 됨
# print(D[0, :3])
    # [-5.532219 -3.779564 -9.972975]


# mel scale

mel_freqs = librosa.mel_frequencies(
    n_mels=20, #몇개의 밴드로 표현할건지
    fmin=0,
    fmax=22050
)

print(mel_freqs)
    #sr, n_mels, fmin, fmax 가 같으면 mel bin 은 동일하게 생성됨
    #[    0.           210.49991101   420.99982202   631.49973303
    #mel_freqs 는 배열이라서 꼭 np.round(mel_freqs, 2) 이렇게 소수점 뽑아야함 