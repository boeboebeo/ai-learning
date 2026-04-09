#day05-1 lowpassfilter _ saw vs noise 
    #day04-2_lowpassfilter.py 에서 코드 너무 지저분해져서 넘어옴

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os   #python 기본 내장모듈. Operating System . 운영체제와 상호작용하는 기능 제공
from scipy.signal import savgol_filter 
from scipy.signal import find_peaks #peak를 찾고 그 뒤의 급락 지점을 cutoff 로 적용하기 위함.




#스퀘어 최적화 해보려다가 결국 안 쓰게 된 함수
def analyze_peak_width(spectrum, threshold_ratio = 0.7): 
    #peak 폭 측정 
    max_val = np.max(spectrum)
    threshold = max_val * threshold_ratio

    above_threshold = spectrum > threshold
    diff = np.diff(np.concatenate([[0], above_threshold, [0]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return 0
    
    max_width = np.max(ends - starts)
    return max_width


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
    # flat_mean = np.median(spectrum_v[:50]) -> 역효과 나서 버림
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


    peaks, properties = find_peaks(spectrum_v, prominence=0.01)
        # prominence에 대한 기준이 없으면, 그냥 노이즈도 peak 로 인식함. (작은 요철도 인식해버려서)
        # prominence = 0.01 은 주변보다 최소 0.01 이상 높은 peak 만 
    
    # 디버깅 
    # print(len(peaks))
    # print(f"spectrum 최댓값: {freqs_v[np.argmax(spectrum_v)]:.1f}Hz")
    # print(f"peak 위치: {freqs_v[peaks[-1]]:.1f}Hz")
    # print(f"peak 높이: {spectrum_v[peaks[-1]]:.4f}")
    # print(f"기음 높이: {spectrum_v[0]:.4f}")

    if len(peaks) > 0:
        # 가장 오른쪽(고주파) peak
        last_peak = peaks[-1] #마지막 피크가 last peak (맨 뒤 피크 고름)
        
        # peak 뒤에서 급락 지점 찾기
        search_range = min(50, len(slope_smooth) - last_peak - 1)   
            #last peak뒤에서 50개 구간을 탐색하고 싶은데 배열 끝을 넘어갈수 있기 때문에
            #전체 index에서 last_peak의 인덱스를 빼고 -1 함. 인덱스 기준으로는 인덱스 오버플로우 방지하기 위해서 -1함
            #근데 search_range 가 0 나오게 되면 else 로 감
        if search_range > 0:
            cutoff_idx_A_peak = last_peak + np.argmin(slope_smooth[last_peak:last_peak+search_range])
                #그리고 그 last_peak 의 뒤 50안에서 제일 변화율이 큰 (-쪽이니까 argmin 으로 처리) 애의 인덱스를 더하면 cutoff idx 나옴
                #slope_smooth =     [10, 20, 30, 40] 이였어도 위에서 [last_peak:last_peak+..] 이렇게 처리하면
                #새로운 배열이 만들어져서   0   1   2   3  => 이런식의 새로운 인덱스 만들어짐! 
        else:
            cutoff_idx_A_peak = last_peak
    else:
        cutoff_idx_A_peak = None
    
    print(f"cutoff_idx: {cutoff_idx_A_peak}") # 얘가 982 이렇게 나오므로, 

    skip_start = 20 #window length 와 비슷하게 / 어짜피 그 앞에서는 savgol filter 경계효과 나타나니까 

    # cutoff_idx_A = skip_start + np.argmin(slope_smooth[skip_start:]) # 스킵 한 그 이후의 인덱스에서만 최소값 구하기
        #변화율이 제일 -쪽으로 작은 값을 cutoff_idx_A로 넣음
        #변화율이 제일 큰 것! 
    actual_freq = freqs_v[8]
    # print(actual_freq) #실제 주파수 확인 

    cutoff_idx_A_slope = skip_start + np.argmin(slope_smooth[skip_start:])

    # 8-3. 둘 중 선택
    if cutoff_idx_A_peak is not None:
        cutoff_idx_A = cutoff_idx_A_peak  # peak 방식 우선
        print(f"Using peak method: {freqs_v[cutoff_idx_A]:.1f}Hz")
    else:
        cutoff_idx_A = cutoff_idx_A_slope  # 없으면 기존 방식
        print(f"Using slope method: {freqs_v[cutoff_idx_A]:.1f}Hz")

    # ✅ 디버깅 추가:
    # print(f"slope_smooth 앞부분: {slope_smooth[:20]}")
    # print(f"slope_smooth 최솟값: {np.min(slope_smooth)}")
    # print(f"cutoff_idx_A: {cutoff_idx_A}")
    # print(f"slope_smooth[cutoff_idx_A]: {slope_smooth[cutoff_idx_A]}")

    # 9. -3dB 계산 (B방식 준비)
    cutoff_idx_B = None
    for i in range(len(spectrum_v)):
        if spectrum_v[i] < threshold: 
            cutoff_idx_B = i
            break
    cutoff_freq_B = freqs_v[cutoff_idx_B] if cutoff_idx_B is not None else None
    # print(cutoff_freq_B)

    # 10. resonance 유무 판단 -> 방식 선택
    peak_candidate = np.max(spectrum_v) #제일 높은 magnitude 를 가진 주파수빈
    flat_mean_ratio = peak_candidate / flat_mean 
        #제일 높은 magnitude / 평평한 곳의 평균치 => 이게 높으면 resonance가 있다고 판단
        #근데 만약, sawtooth 이고, resonance 가 그렇게 높지 않다면? 
        # -> 그래서 sawtooth 의 기음이 제일 높다면? 
    # print(f"peak_candidate : {peak_candidate}")
    # print(f"flat_mean : {flat_mean}")
    # print(f"flat_mean_ratio : {flat_mean_ratio}")
    # resonance는 보통 특정 범위에 몰려있음
    # print(f"cutoff_idx_A: {cutoff_idx_A}")

    # -- 새 함수 호출 ------------------------------------------------------
    peak_width = analyze_peak_width(spectrum_v, threshold_ratio = 0.7)
    # print(f"peak_width : {peak_width}")
    
    # if flat_mean_ratio > 1.3 and cutoff_idx_A >0 and peak_width > 15:  # 좁은 peak만
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
            # print(freqs_v[peak_idx]) #얘 때문에 cutoff freq 가 그냥 peak 값으로 찍힘
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
        #fundamental = 220 이라서 그런지 자꾸 COF 가 221로 나옴.
        #아마 peak_idx 때문일 확률 높음

    "Librosa-basics/audio_sample/square+LPF(1100hires).wav", 
    "Librosa-basics/audio_sample/square+LPF(645mires).wav",  #fundamental > resonance Hz  
]

print("=" * 50)
for i, path in enumerate(audio_files): #enumerate 로 인덱스 추가
    filename = os.path.basename(path)  # os.path.basename() : 경로에서 파일명만 추출하는 함수
    y, sr = librosa.load(path)
    cutoff, res, meth_used = estimate_lpf(y, sr) #위 estimate_lpf(y, sr) 수행했을때의 return 값
    
    if cutoff is not None:
        print(f"[{filename}]")
        print(f"  COF      : {cutoff:.0f}Hz ({meth_used})") #여기서 cutoff freq 를 반올림하고 있기때문에 220.7153.. 인 peak 값에서 반올림되어서 얘기 COF로 나옴
        print(f"  Resonance: {res}")
    else:
        print(f"[{filename}] No LPF detected") #cutoff 값이 None 이라면 
    
    #
    if i < len(audio_files) - 1: #전체 audio_files의 개수의 -1 한 것보다 (index 0부터 시작하니까) 작을때만 ---표기 하고 아니면 ===하게끔 
        print("-" * 50)

print("=" *50)  


"""
ex. 

sr = 44100
n_fft = 2048

#주파수 bins 개수
n_bins = n_fft // 2 + 1 = 1025개 #0~1024 까지의 주파수 빈 존재

#최대 주파수 : 나이퀴스트
max_freq = sr / 2 = 22050Hz 

#주파수 간격
freq_step 
    = max_freq(=sr/2) / (n_fft/2)
    = 22050 / 1024
    = 21.53Hz
            

#주파수 배열 :
freq[0] : 0Hz
freq[1] : 21.53Hz
freq[2] : 43.07Hz

...

freqs[1024] : 21.53 * 1024 = 22050Hz

cutoff_idx_A가 8이라고 나와서 계산했더니 243Hz 정도로 나옴 .

"""

"""
"Librosa-basics/audio_sample/square+LPF(2000).wav", 의 COF 가 맞지 않는 issue 발생 !

=> 우선 flat_mean_ratio 값이 거의 6정도로 매우 커서
    그 안에 있는 if 문의 peak_idx로 cutoff freq 가 잡혀버리고 만다 
    우선 flat_mean_ratio 는 peak_candidate / flat_mean 한 값인데, 
    아마 peak_candidate 은 220Hz 일거고 (기음) 
    -> flat_mean 은 200Hz 부터 10indx 뒤 까지인 200 + 21.53*10 인 약 430Hz 까지이고, 
        거기 까지는 220Hz 말고는 뭐가 없기 때문에 평균을 구하면 매우 작은 수가 나와 저렇게 flat_mean_ratio 가 엄청 커져버림

        => 해결방안(1) : 
            flat_mean 구할때 평균을 내지말고, 극단값에 덜 민감한(위의 사례에서의 기음 : 220Hz)
            np.median 사용해보기 -> 평균이 아닌 중앙값을 냄 
            flat_mean = np.median(spectrum_v[:50])
                => 결과 : 
                    flat_mean이 더 올라가버림 .. 역효과 ! median 이 더 작아져서 비율이 더 커짐 

        => 해결방안(2) : 
            flat_mean = np.mean(spectrum_v[100:150]) 
            이렇게 맨 앞값이 아닌 중간값을 사용해보기 

        => 해결방안(3) : 
            if flat_mean_ratio > 8.0 and cutoff_idx_A > 0:  # 매우 엄격
            이렇게 flat_mean_ratio 의 기준을 좀 많이 올려보기 => 근데 그랬더니 다른 파형에서 오류 생김

        => 해결방안(4) : 
            analyze_peak_width(spectrum, threshold_ratio=0.7) 이라는 새 함수 추가
            => 했는데 우선 이건 Noise, saw 에 최적화되어있던 알고리즘이라 그런지 square 의 정답률이 더 멀어짐 . 
            => 다시 noise, sawtooth 최적화 알고리즘으로 돌아가자...! 
        

"""

"""    "Librosa-basics/audio_sample/saw+LPF(5000hires).wav", 
=> 위 sawtooth 파형 resonance 못 잡고, 컷오프 한참 낮게 나오는것 확인 

    => 우선 if 문에서 cutoff_idx_A = 0 이기 때문에 else 문으로 가는 것 확인
    => cutoff_idx_A = 0 이 나오는 이유 
    우선 slope_smooth 앞부분
    : [-0.01230647 -0.01000998 -0.00799438 -0.00624936 -0.00476461 -0.00352981
        -0.00253463 -0.00176877 -0.00122191 -0.00088372 -0.0007439  -0.00328727
        -0.00479215 -0.0030079   0.00019185  0.00168616  0.00068458 -0.00164151
        -0.00338468 -0.00321784]
    인데 여기서 slope_smooth 최솟값: -0.012306474149227142. -> 맨 앞 index 가 제일 작게 나옴
    => 따라서 cutoff_idx 가 0으로 나왔던 것 . 

-why? 
window_length = 21 인데 앞뒤 10개씩 필요하지만, index 0~10까지는 충분한 데이터가 없어서 경계에서 이상한 값이 나옴.
=> savgol filter 경계효과 
(각 지점마다 앞뒤 10개 (총 21개)를 보고 다항식으로 smoothing 하는거라서)
    => 근데 index 0은 앞에 10개가 없음
        => 해결방법 (1): 
        그 앞 20개의 인덱스를 무시하고 계산하기 위해서 아래와 같이 처리

        skip_start = 20 #window length 와 비슷하게
        cutoff_idx_A = skip_start + np.argmin(slope_smooth[skip_start:])

        => (2) 근데 이렇게 skip을 해도 idx_A 는 24가 나와서 실제 COF(5000Hz)와는 턱없이 멀음
            : 60Hz ~ 5000Hz 까지 전체적으로 계속 감소하기 때문. slope 가 계속 음수여서 이러한 에러가 발생한다.
            => 또 기음이 60Hz 이기때문에 거기서 다음 2배음 까지가 제일 많이 급락함.

"""
    
