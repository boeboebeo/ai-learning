#day05-1 lowpassfilter _ saw vs noise 
    #day04-2_lowpassfilter.py 에서 코드 너무 지저분해져서 넘어옴

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os   #python 기본 내장모듈. Operating System . 운영체제와 상호작용하는 기능 제공
from scipy.signal import savgol_filter 


def estimate_lpf(y, sr):

    # 1. STFT -> magnitude
    D = np.abs(librosa.stft(y, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    # 2. 시간 평균 -> spectrum
    spectrum = np.mean(D, axis=1)
    # print(len(spectrum)) 
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
        #200Hz이상인 대역만 유효한 대역으로 자름 

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

    if flat_mean_ratio > 1.3 and cutoff_idx_A > 0: 
        # cutoff_idx_A 는 위에서 gradient 를 계산한거이기때문에 만약 첫번째 인덱스에서 slope이 가장 작을수도 있음
        # ex. slope_smooth = [-100, -50, -30, -20, ...] => cutoff_idx_A = 0 도 가능
        # 근데 왜 cutoff_idx_A 가 0인 상황은 제외하는거지?
        # resonance 있음 -> A방식 (peak 꼭대기)
        search_start = max(0, cutoff_idx_A - 30) #cutoff idx 에서 30앞 과 0 중 더 큰 값이 search start 지점 : cutoff idx 에서 30뺐을때 0보다 더 마이너스일 수도 있으니까 
        search_end = cutoff_idx_A #컷 오프 지점 까지
        #컷오프가 첫번째 idx 일수도 있기때문에 나뉘어진 if/else 
        if search_end > search_start: #peak를 찾을 범위가 존재하는가? 
                # 빈 배열 방지. 근데 빈배열이 되는 경우 : cutoff_idx_A = 0, search_start 부분의 max 가 0이 되고, search end 도 0 이라면 -> spectrum_v[0:0] : 빈배열이 출력됨 !
                # 안그럼 np.argmax( ) -> 여기서 ValueError 발생함
            peak_idx = search_start + np.argmax(spectrum_v[search_start:search_end]) # cut off freq 지점은 떨어지는 시작점이기때문에 peak 찾는 범위 안에 포함되지 않아도 됨
                # 상대적인 인덱스가 나오는 걸 방지 
                # 그냥 peak_idx = np.argmax(spectrum_v[20:50])을 하게되면 원본 배열의 위치가 아님(그냥 0 ~ 29 사이의 값 반환함 . 슬라이스 내 상대 인덱스)
            cutoff_freq = freqs_v[peak_idx]
        else:   #검색할 범위가 없다면 (빈 배열) -> peak를 못찾겠으니 그냥 cutoff 지점을 사용 
            cutoff_freq = freqs_v[cutoff_idx_A]
            #peak를 찾을 수 없음. -> 원래 지점을 cutoff 지점으로 사용 (resonance 있을때는 그게 cutoff지점이니까)
        method_used = "slope(resonance)"
    else:
        # flatmean_ratio 가 1.3보다 낮으면 resonance 없다고 간주 -> B방식 (-3dB)
        cutoff_freq = cutoff_freq_B
        method_used = "-3dB"

    # 11. resonance 수치 계산
    cutoff_idx_final = np.argmin(np.abs(freqs_v - cutoff_freq)) #cutoff 와 제일 간격이 작은 index 찾기 = 그 index 가 cutoff idx
    region_start = max(0, cutoff_idx_final - 15) #cutoff idx 에서 15뺀 지점 부터 or 0 (이것도 15뺐을때 -인거 대비해서 0)
    region_end = min(len(spectrum_v), cutoff_idx_final + 15) #이것도 전체 spectrum_v 인덱스 넘어가지 않게 min 
    region = spectrum_v[region_start:region_end]
    peak = np.max(region) #region 내에서의 최고값 
    base = flat_mean #평평한 곳의 평균값 
    resonance_ratio = peak / (base + 1e-8) #

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
    "Librosa-basics/audio_sample/square+LPF(nofilter).wav",
    "Librosa-basics/audio_sample/square+LPF(2000).wav",
    "Librosa-basics/audio_sample/square+LPF(1100hires).wav",    
]

print("=" * 50)
for i, path in enumerate(audio_files): #enumerate 로 인덱스 추가
    filename = os.path.basename(path)  # os.path.basename() : 경로에서 파일명만 추출하는 함수
    y, sr = librosa.load(path)
    cutoff, res, meth_used = estimate_lpf(y, sr) #위 estimate_lpf(y, sr) 수행했을때의 return 값
    
    if cutoff is not None:
        print(f"[{filename}]")
        print(f"  COF      : {cutoff:.0f}Hz ({meth_used})")
        print(f"  Resonance: {res}")
    else:
        print(f"[{filename}] No LPF detected") #cutoff 값이 None 이라면 
    
    #
    if i < len(audio_files) - 1: #전체 audio_files의 개수의 -1 한 것보다 (index 0부터 시작하니까) 작을때만 ---표기 하고 아니면 ===하게끔 
        print("-" * 50)

print("=" *50)  
    
