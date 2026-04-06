#day05-1 lowpassfilter _ saw vs noise 
    #day04-2_lowpassfilter.py 에서 코드 너무 지저분해져서 넘어옴

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os   #? 
from scipy.signal import savgol_filter 


def estimate_lpf(y, sr):

    # 1. STFT -> magnitude
    D = np.abs(librosa.stft(y, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    # 2. 시간 평균 -> spectrum
    spectrum = np.mean(D, axis=1)
    print(len(spectrum)) 
        #2049 개 나옴. axis=1 방향으로 평균. -> 가로방향(시간축)각각의 주파수빈의 평균 낸것 
        #만약 axis = 0 이라면 각 시간의 평균. -> 세로방향으로 평균

    # 3. log scale + 정규화
    spectrum = np.log1p(spectrum)
    spectrum /= np.max(spectrum)

    # 4. smoothing
    spectrum_smooth = savgol_filter(spectrum, window_length=21, polyorder=3)

    # 5. 탐색 범위 제한
    valid_mask = freqs >= 200
    freqs_v = freqs[valid_mask]
    spectrum_v = spectrum_smooth[valid_mask]

    # 6. 평탄구간 평균 (base, threshold 계산용)
    flat_mean = np.mean(spectrum_v[:len(spectrum_v)//10])
    threshold = flat_mean * (10 ** (-3/20)) #flat_mean에서 -3dB된 레벨이 threshold

    # 7. LPF 없는 신호 감지
    high_mean = np.mean(spectrum_v[int(len(spectrum_v)*0.8):])
    overall_mean = np.mean(spectrum_v)
    if high_mean / (overall_mean + 1e-8) > 0.5: 
            #overall_mean이 0인 상태는 거의 불가능하지만, 완전한 무음파일/파일로딩실패로 빈 배열/ 극도로 작은 값들이 반올림 되어 0으로 수렴할 수 있음
            #0.5 는 경험적 임계치
        return None, "No LPF", "No LPF"

    # 8. slope 계산 (A방식 준비)
    slope = np.gradient(spectrum_v) #변화율
    slope_smooth = savgol_filter(slope, window_length=21, polyorder=3)
    cutoff_idx_A = np.argmin(slope_smooth) 
        #변화율이 제일 -쪽으로 작은 값을 cutoff_idx_A로 넣음
        #변화율이 제일 큰 것! 

    # 9. -3dB 계산 (B방식 준비)
    cutoff_idx_B = None
    for i in range(len(spectrum_v)):
        if spectrum_v[i] < threshold: 
            cutoff_idx_B = i
            break
    cutoff_freq_B = freqs_v[cutoff_idx_B] if cutoff_idx_B is not None else None

    # 10. resonance 유무 판단 -> 방식 선택
    peak_candidate = np.max(spectrum_v) #제일 높은 magnitude 를 가진 주파수빈
    flat_mean_ratio = peak_candidate / flat_mean 
        #제일 높은 magnitude / 평평한 곳의 평균치 => 이게 높으면 resonance가 있다고 판단
        #근데 만약, sawtooth 이고, resonance 가 그렇게 높지 않다면? 
        # -> 그래서 sawtooth 의 기음이 제일 높다면? 

    if flat_mean_ratio > 1.3 and cutoff_idx_A > 0: #근데 cut_idx_A는 항상 0보다는 큰거 아니야? (완전 맨앞 index만 뺀다면?)
        # resonance 있음 -> A방식 (peak 꼭대기)
        search_start = max(0, cutoff_idx_A - 30)
        search_end = cutoff_idx_A
        if search_end > search_start:  # 빈 배열 방지
            peak_idx = search_start + np.argmax(spectrum_v[search_start:search_end])
            cutoff_freq = freqs_v[peak_idx]
        else:
            cutoff_freq = freqs_v[cutoff_idx_A]
        method_used = "slope(resonance)"
    else:
        # resonance 없음 -> B방식 (-3dB)
        cutoff_freq = cutoff_freq_B
        method_used = "-3dB"

    # 11. resonance 수치 계산
    cutoff_idx_final = np.argmin(np.abs(freqs_v - cutoff_freq))
    region_start = max(0, cutoff_idx_final - 15)
    region_end = min(len(spectrum_v), cutoff_idx_final + 15)
    region = spectrum_v[region_start:region_end]
    peak = np.max(region)
    base = flat_mean
    resonance_ratio = peak / (base + 1e-8)

    if resonance_ratio > 1.5:
        resonance_label = "High Resonance"
    elif resonance_ratio > 1.2:
        resonance_label = "Medium Resonance"
    else:
        resonance_label = "Low Resonance"

    return cutoff_freq, resonance_label, method_used


# 테스트할 오디오 파일 목록 (한번에 테스트 되게 하기)
audio_files = [
    "Librosa-basics/audio_sample/saw+LPF(700).wav",
    "Librosa-basics/audio_sample/saw+LPF(5000hires).wav",
    "Librosa-basics/audio_sample/saw+LPF(300).wav",
    "Librosa-basics/audio_sample/saw+LPF(nofilter).wav",
    "Librosa-basics/audio_sample/noise+LPF(300).wav",
    "Librosa-basics/audio_sample/noise+LPF(1000).wav",
    "Librosa-basics/audio_sample/noise+LPF(5000hires).wav",
    "Librosa-basics/audio_sample/noise+LPF(5000res).wav",
]

print("=" * 50)
for path in audio_files:
    filename = os.path.basename(path)  # 경로 빼고 파일명만
    y, sr = librosa.load(path)
    cutoff, res, meth_used = estimate_lpf(y, sr)
    
    if cutoff is not None:
        print(f"[{filename}]")
        print(f"  COF     : {cutoff:.0f}Hz ({meth_used})")
        print(f"  Resonance: {res}")
    else:
        print(f"[{filename}] No LPF detected")
    print("-" * 50)