# day04-3 lowpass filter version2

# 1. 평균스펙트럼 -> 2. 부드럽게 만들기 -> 3. 기준 dB이하 지점 찾기

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

def estimate_lpf_pro(y, sr):

    # 1. STFT
    D = np.abs(librosa.stft(y, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr)

    # 2. 평균 스펙트럼
    spectrum = np.mean(D, axis=1)

    # 3. dB 변환
    spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)

    # 4. 스무딩 (핵심)
    smooth = gaussian_filter1d(spectrum_db, sigma=3)

    # 5. cutoff 찾기 (-20 dB 기준)
    threshold = -20

    idx_candidates = np.where(smooth < threshold)[0]

    if len(idx_candidates) == 0:
        cutoff = freqs[-1]
    else:
        cutoff = freqs[idx_candidates[0]]

    # 6. resonance (peak 검사)
    peak_region = smooth[max(0, idx_candidates[0]-10):idx_candidates[0]]
    resonance = np.max(peak_region) - smooth[idx_candidates[0]]

    if resonance > 6:
        res_label = "High Resonance"
    elif resonance > 3:
        res_label = "Medium Resonance"
    else:
        res_label = "Low Resonance"

    return cutoff, res_label


# 사용
y, sr = librosa.load("your_audio.wav")

cutoff, res = estimate_lpf_pro(y, sr)

print(f"LPF cutoff: {cutoff:.0f} Hz")
print(f"Resonance: {res}")