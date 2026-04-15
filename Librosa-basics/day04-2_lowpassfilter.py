#day 04-2 low pass filter - COF , RES 구하기
    #오케이 우선은.. noise 에서는 200Hz 안으로 오차가 없이 resonance 까지 잘 추출하다가
    #saw 가 들어오니 평탄한 파형이 아니라서.... 에러생기고 위 아래 왔다갔다 수정하다가 우선 
    #코드 정리 먼저 하기로.. => "day05-1_lowpassfilter_sawvsnoise" 여기에 정리해놓음

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # scipy.signal : signal processing 관련 함수들 모여있는 파이썬 패키지
                                #savgol_filter : Savitzky-Golay 필터 구현한 함수. 신호의 노이즈를 줄이면서+피크/곡선/shape 최대한 유지. 곡률(curvature)을 보존

def estimate_lpf(y, sr):

    #1. STFT -> magnitude
    D = np.abs(librosa.stft(y, n_fft=4096)) #0Hz ~ Nyquist freq (sr/2) 를 균등 간격으로 나눔. 
                                            #ex. bin width = 22050 / 4096 => 하나의 주파수 bin당 약 5.38Hz 
                                            #bin 개수 = 4096/2 + 1 = 2049 (복소수, magnitude는 같고, phase 는 반대. 중간 지점 이후부터는 대칭이기 때문에)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096) #아래에서 slope, spectrum 보려고 freqs 도 n_fft=4096으로 통일함

    #2. time average (spectrum)
    spectrum = np.mean(D, axis=1)   
        #각 주파수 bin 별로 "시간에 따른 진폭 변화"를 평균 낸것 -> 이 소리에서 어떤 주파수가 전체적으로 많이 존재했는지 보려함
        #axis = 1 : 어느 방향으로 계산할건지 정함
    # print(spectrum) # 확인용. => 여기서의 specrum은 주파수별 진폭평균값이 들어있는 1차원 배열

    #3. log scale (사람귀+필터 형태 반영)
    spectrum = np.log1p(spectrum)
    # print("\n",spectrum) # 확인용. => 여기서의 spectrum

    #4. 정규화 (shape 만 보고)
    spectrum /= np.max(spectrum) 
        # /= : spectrum 의 각 값을 나누고 바로 a에 대입
        # 배열 전체의 최대값을 골라서 그걸로 모든 배열 전체를 나눠 0~1 범위로 정규화(normalization) 함 
        # spectrum[i] = spectrum[i] / np.max(spectrum) #모든 i에 대해 적용 하는 이 식과 같은 코드
    # print(spectrum) # 확인용 . 
    # print(np.max(spectrum)) #정규화 되었으므로 np.max(spectrum) 해보면 1 나옴.

    # print(spectrum)
    #new smoother (튀는 값 제거)
    spectrum_smooth = savgol_filter(spectrum, window_length=21, polyorder=3)
    # print(spectrum_smooth)
        #데이터를 매끄럽게 다듬는 함수
        #window_length = 한번에 21개의 포인트를 보면서 다듬음
        #polyorder = 3 -> 3차 다항식으로 맞춤 
        #polyorder = 1 : 직선
        #polyorder = 2 : U자 곡선 (2차 다항식)
        #polyorder = 3 : S자 곡선 (3차 다항식) -> 차수가 높을수록 원본데이터와 비슷해짐

    

    #new 탐색범위 제한 (200Hz ~ sr/2)
    freq_min = 70
    valid_mask = freqs >= freq_min # freq_min(200Hz)보다 높은 곳만 True 로 [False, False, False, True, True .. ]
                                    # 이렇게 Boolean index 함
    freqs_v = freqs[valid_mask]     # valid_mask 내의 True 인 것만 남게 됨 => freqs, spectrum이 같은 mask 로 잘리니까 인덱스가 계속 대응되어서 좋음
    spectrum_v = spectrum_smooth[valid_mask]

    flat_region = spectrum_v[:len(spectrum_v)//10] #유효한 구간의 앞 10% 구간
        #dB 변환
    spectrum_db = 20*np.log10(spectrum_v + 1e-10) #log(0) 방지
        # y축 dB로 할랬더니, x y 가 다 같게 매칭되는 값이여야 된다 해서 spectrum_v + 1e-10으로 바꿈 
    flat_mean = np.mean(flat_region)
    threshold = flat_mean * (10 ** (-3/20))  # flat_mean의 70.8%



    # method A : slope 기반 
    slope = np.gradient(spectrum_v) #변화율(기울기) 배열
    slope_smooth = savgol_filter(slope, window_length=21, polyorder=3) #그걸 smoothing 
    cutoff_idx_A = np.argmin(slope_smooth) #그 smoothing 된 값 중에 제일 작은 거 고르기(왜지?) 
                                        #-> 변화량이 제일 작은 걸 고르는데 왜 cutoff 가 되는지 : slope가 음수(-)인 상태가 -> 내려가는 중임 (0은 변화거의 없음)

    
    cutoff_idx_B = None
    for i in range(len(spectrum_v)):
        if spectrum_v[i] < threshold: #여러 인덱스중 threshold 보다 작은 지점이 있었는지?
            cutoff_idx_B = i #있었다면 그 인덱스를 cutoff_idx_B 에 저장하고, break
            break

    
    # -3dB 신뢰도 : 떨어진 정도가 뚜렷한가
    if cutoff_idx_B is not None: #만약 cutoff_idx_B 가 None 이 아니라면 (다른 인덱스가 들어왔다면)
        cutoff_freq_B = freqs_v[cutoff_idx_B] #그 인덱스의 값을 cut off freq B 로 저장
        drop = flat_mean - spectrum_v[cutoff_idx_B] 
            # 평탄한 부분의 평균에서 그 인덱스의 magnitude 값 뺌 (?) 
        confidence_B = drop

    else:
        cutoff_freq_B = None
        confidence_B = 0


    # resonance 있는지 먼저 판단하고 A or B 결정 => (1)번의 방식으로 한번 resonance 유무 부터 판단해보기로 
    peak_candidate = np.max(spectrum_v)
    flat_mean_ratio = peak_candidate / flat_mean

    if flat_mean_ratio > 1.3:
        #A방식 : peak 지점을 컷오프로
        search_start = max(0, cutoff_idx_A - 30)
        search_end = cutoff_idx_A
        peak_idx = search_start + np.argmax(spectrum_v[search_start:search_end])
        cutoff_freq = freqs_v[peak_idx]
        method_used = "slope(resonance)"

    else :
        #B방식 : -3dB 지점을 컷오프로
        cutoff_freq = cutoff_freq_B
        method_used = "-3dB"


    # # method A : slope 기반 
    # slope = np.gradient(spectrum_v) #변화율(기울기) 배열
    # slope_smooth = savgol_filter(slope, window_length=21, polyorder=3) #그걸 smoothing 
    # cutoff_idx_A = np.argmin(slope_smooth) #그 smoothing 된 값 중에 제일 작은 거 고르기(왜지?) 
    #                                     #-> 변화량이 제일 작은 걸 고르는데 왜 cutoff 가 되는지 : slope가 음수(-)인 상태가 -> 내려가는 중임 (0은 변화거의 없음)
    # cutoff_freq_A = freqs_v[cutoff_idx_A]



    # saw 를 넣었을때 ValueError: attempt to get argmax of an empty sequence 잡는중
    print(spectrum_v[search_start:search_end]) #아하 출력해보니까 여기의 이 리스트가 비어있음 

    print(cutoff_idx_A) #왜 컷오프 인덱스가 0으로 나오지???
    print(freqs_v)
    print(spectrum_v)
    print(slope_smooth)

    # peak_idx = search_start + np.argmax(spectrum_v[search_start:search_end])
        #white 노이즈의 경우 전 주파수 레벨이 평탄하기 때문에 peak 로 cutoff freq를 구하는것이 가능했음. 
        #근데 sawtooth 라면? -> 현재 error : 특히 argmax()쓸때, array 나 list 안에 아무런 요소도 안들어 있음
        #=> saw 의 경우는 기음에서 바로 뚝떨어지니까 slope 가 맨앞 index 0 근처에서 제일 가파르게 내려감. 그래서 argmin : 0 을 반환 (index 0 : 처음 주파수 빈)
        # 가장 급격히 떨어지는 곳을 찾아야 하는데, saw 는 기음의 drop 에 낚이고 있음
    
    """ 여기서.. 처리 방법
    (1) resonance 가 없는 saw 면 그냥 바로 confidence B의 방법으로 가게끔 처리하면 되는데, 
        => 그러면 결국 resonance 가 있는 saw 에서는 대응이 안되긴 함 
    
    (2) 기음을 무시하고 일정 주파수 이상에서만 탐색하게 함
        => 근데 그럼 컷오프가 ex. 1000Hz (이걸 min 으로 설정했을때) 이하라면 못잡는 일이 발생함.
    
    """

    

    cutoff_freq_A = freqs_v[peak_idx] 
        # 아 근데 A method 는 그 피크 지점을 cutoff freq 로 지정하면 안됨!!! 
        # 그래서 만약 레조넌스 있다면 A, 그리고 없다면 B의 -3dB지점을 컷오프로 지정하게끔 만들어줘야 할듯


    # slope 신뢰도 : 그 지점의 기울기가 얼마나 급격한지
    # confidence_A = abs(slope_smooth[cutoff_idx_A]) 
        #아 slope_smooth 배열 안에 들어있는 것중에서 cuttoff_indx_A 인덱스 값을 꺼내고, 절대값 취함
        # 변화율에 절대값 취한게 confidence_A

    #method B : -3dB 기반
    #저주파 평탄 구간 평균을 기준(0dB)로 삼음
    # flat_region = spectrum_v[:len(spectrum_v)//10] #유효한 구간의 앞 10% 구간
    #     #dB 변환
    # spectrum_db = 20*np.log10(spectrum_v + 1e-10) #log(0) 방지
    #     # y축 dB로 할랬더니, x y 가 다 같게 매칭되는 값이여야 된다 해서 spectrum_v + 1e-10으로 바꿈 
    # flat_mean = np.mean(flat_region)
    # flat_mean_db = np.mean(spectrum_db[:len(spectrum_db)//10])
    # threshold = flat_mean_db - 3
    # threshold = flat_mean * (10 ** (-3/20))  # flat_mean의 70.8%
    

    # cutoff_idx_B = None
    # for i in range(len(spectrum_v)):
    #     if spectrum_v[i] < threshold: #여러 인덱스중 threshold 보다 작은 지점이 있었는지?
    #         cutoff_idx_B = i #있었다면 그 인덱스를 cutoff_idx_B 에 저장하고, break
    #         break

    # # -3dB 신뢰도 : 떨어진 정도가 뚜렷한가
    # if cutoff_idx_B is not None: #만약 cutoff_idx_B 가 None 이 아니라면 (다른 인덱스가 들어왔다면)
    #     cutoff_freq_B = freqs_v[cutoff_idx_B] #그 인덱스의 값을 cut off freq B 로 저장
    #     drop = flat_mean - spectrum_v[cutoff_idx_B] 
    #         # 평탄한 부분의 평균에서 그 인덱스의 magnitude 값 뺌 (?) 
    #     confidence_B = drop

    # else:
    #     cutoff_freq_B = None
    #     confidence_B = 0

    # ── LPF 없는 신호 감지 ───────────────────────────────
    #고주파 영역(상위 20%) 평균이 전체 평균의 50% 이상이면 LPF 없다고 판단 
    high_region = spectrum_v[int(len(spectrum_v)*0.8):] # 전체 주파수의 0.8 부근에서 부터 끝까지 (1까지)
    high_mean = np.mean(high_region)
    overall_mean = np.mean(spectrum_v)

    if high_mean / (overall_mean + 1e-8) > 0.5:
        print("No LPF detected")
        return None, "No LPF"
    



    # ── 최종 선택 ────────────────────────────────────────
    # cutoff_freq = cutoff_freq_A
    # method_used = "slope"

    if cutoff_freq_B and abs(cutoff_freq_A - cutoff_freq_B) > 1000:
        print(f"A({cutoff_freq_A:.0f}Hz) 와 B({cutoff_freq_B:.0f}Hz)차이가 큼 -> 결과 불확실")

    
    # if cutoff_freq_B is not None and confidence_B > confidence_A:
    #     cutoff_freq = cutoff_freq_B
    #     그method_used = "-3dB"
    # else:
    #     cutoff_freq = cutoff_freq_A
    #     method_used = "slope"

    # print(f"\n[slope 기반] {cutoff_freq_A:.0f}Hz (신뢰도: {confidence_A:.4f})")
    # if cutoff_freq_B:
    #     print(f"[-3dB 기반] {cutoff_freq_B:.0f}Hz (신뢰도 : {confidence_B:.4f})")
    # print(f"-> 최종선택 : {method_used} 방식")



    # #5. 기울기 (배열의 변화율(기울기)를 구하는 함수) . 미분의 이산버전 
    # slope = np.gradient(spectrum)
    # # print(slope)
    # with open("slope.txt","w") as f: # (1) 이렇게 해서 그래프 txt로 보거나
    #     for i, val in enumerate(slope):
    #         f.write(f"{i} {val}\n")




    # #6. cutoff 찾기 . 가장 급격히 떨어지는 지점 -> steepest drop 찾기
    # cutoff_idx = np.argmin(slope) #가장 급격하게 하강하는 지점의 index 알려줌 
    # cutoff_freq = freqs[cutoff_idx]

    # #7. resonanace estimation . cutoff 주변 peak 존재 여부
    # region = spectrum[max(0, cutoff_idx-5):cutoff_idx+5] #마지막 인덱스 포함안돼서 10개의 값 불러와짐
    # print(region)

    # peak = np.max(region)
    # base = spectrum[cutoff_idx]  #여기를 (cutoff_idx)로 불러오려해서 에러가 났었음. 함수가 아닌데 왜 ()를 붙여서 실행하려고 하냐.. 해서 난 에러 

    # # resonance = peak - base #피크 강조 정도. peak(컷오프 주변 값 중) 가 base(컷오프 지점 magnitude)보다 높으면 resonance = 공진
    # resonance_ratio = peak / (base + 1e-8)  #비율로 보기

    # ── Resonance 추정 -> 범위가 좁아서 더 확장 ───────────────────────────────────
    # 레조넌스가 있는경우 cut off 지점이 base 가 될 수 없음 -> 변경 필요
    cutoff_idx_final = np.argmin(np.abs(freqs_v - cutoff_freq)) #freq_v내의 index에서 cutoff_freq 랑 제일 가까운 곳을 찾고 (제일 간격이 작은 값)
    region_start = max(0, cutoff_idx_final - 15)   # 위에서 찾은 cut off freq index 의 -15 전의 인덱스 부터의 값에서 부터 제일 큰 값 찾기
    region_end = min(len(spectrum_v), cutoff_idx_final + 15) 
        #spectrum_v의 길이는 왜 세는거지..? 암튼 그 cut off index 값 +15 한 인덱스까지에서의 제일 작은 값 찾기
        #유효 스펙트럼의 길이(index 개수) 보다 더 큰 수가 나오면 에러가 나니가 그 것보다 더 작은 수를 선택하게 함 -> 전체 인덱스 길이 넘어가면 그냥 총 Index 개수로 결정
    region = spectrum_v[region_start:region_end]  #region 은 제일 큰 값부터, 제일 작은 값 사이의 인덱스들 
    peak = np.max(region)  # cut off 주변 region 내의 제일 큰 진폭을 가진 값이 Peak 

    # base = spectrum_v[cutoff_idx_final] #cut off 의 인덱스를 가진 그 magnitude 가 base (base 가 여기서는 뭐말하는거야)
                                        #cut off 지점의 값 
                                        #근데 지금은 cut off 지점을 base 로 처리하게 되면 안됨. cut off 가 곧 peak 지점 이기 때문에
    # flat_before_cutoff = spectrum_v[max(0, cutoff_idx_final - 1000):cutoff_idx_final - 20]
    base = np.mean(flat_mean)
    resonance_ratio = peak / (base + 1e-8) #내가 보기에 지금 peak 랑 base 가 별로 차이가 안난다는게 문제야


    print(f"\npeak: {peak:.6f}")
    print(f"base: {base:.6f}")
    print(f"resonance_ratio: {resonance_ratio:.6f}") 


    #8. 결과정리
    if resonance_ratio > 1.5 :
        resonance_label = "High Resonance"
    elif resonance_ratio > 1.2 :
        resonance_label = "Mediun Resonance"
    else:
        resonance_label = "Low Resonance"




    # plt.figure(figsize=(8,8))
    # plt.plot(freqs_v[1:], spectrum_v[1:], label="spectrum") #(2) 이렇게 해서 spectrum 확인
    # plt.plot(freqs_v[1:], slope_smooth[1:], label="slope")   #error : x , y 가 같은 n_fft 기준으로 만들어야 함 
    # plt.axvline(x=cutoff_freq, color='r', linestyle='--', label=f"cutoff:{cutoff_freq:.0f}Hz ({method_used})")
    # plt.xscale("log")
    # plt.legend() # 각 그래프 뭔지 알려줌 
    # plt.grid(which="both")
    # plt.title("LPF Estimation")
    # plt.show()
    
    # ── 디버깅 용 그래프 ──────────────────────────────────────────



    print(len(freqs_v), len(spectrum_v), len(slope_smooth), len(spectrum_db))
        #다 같게 나옴 이제

    plt.figure(figsize=(14, 6))

    # spectrum
    plt.plot(freqs_v, spectrum_db, label="spectrum", alpha=0.4)

    # slope
    plt.plot(freqs_v, slope_smooth, label="slope (smoothed)", alpha=0.7)

    # method A 결과
    plt.axvline(x=cutoff_freq_A, color='blue', linestyle='--', label=f"A(slope): {cutoff_freq_A:.0f}Hz")

    # method B 결과
    if cutoff_freq_B:
        plt.axvline(x=cutoff_freq_B, color='green', linestyle='--', label=f"B(-3dB): {cutoff_freq_B:.0f}Hz")

    # 진짜 정답 (알고 있으면)
    # plt.axvline(x=5000, color='red', linestyle='-', label="COF: 5000Hz")

    # threshold 선
    plt.axhline(y=threshold, color='orange', linestyle=':', label=f"threshold: {threshold:.3f}")

    plt.axvline(x=cutoff_freq, color='r', linestyle='--')
    plt.axvspan(freqs_v[region_start], freqs_v[region_end], alpha=0.2, color='blue', label="region")
    #어디가 문제인지 보기위해서 점 찍어보기 
    plt.axvline(x=freqs_v[cutoff_idx_final], color='purple', linestyle='--', label=f"cut off: {base:.3f}")
    plt.axhline(y=peak, color='pink', linestyle='--', label=f"peak: {peak:.3f}")
    plt.axhline(y=base, color='brown', linestyle='--', label=f"base value: {base:.3f}")

    plt.xscale("log")
    plt.ylabel("Amplitude(dB)")
    plt.legend()
    plt.grid(which="both")
    plt.title("LPF Debugging")
    plt.show()

    #test 용..
    # print(f"confidence_A : {confidence_A:.6f}")
    # print(f"confidence_B : {confidence_B:.6f}")
    # print(f"cutoff_freq_A : {cutoff_freq_A:.0f}Hz")
    # print(f"cutoff_freq_B : {cutoff_freq_B:.0f}Hz")



    return cutoff_freq, resonance_label, method_used

# 사용
y, sr = librosa.load("Librosa-basics/audio_sample/saw+LPF(700).wav")

cutoff, res, meth_used = estimate_lpf(y, sr)

if cutoff is not None:
    print(f"\nLPF cutoff : {cutoff:.0f}Hz ({meth_used})")
    print(f"Resonance : {res}")





"""1. STFT -> magnitude

librosa.stft(y, n_fft=4096)
    시간 ----------------> (각 주파수 성분의 세기(amplitude)+위상)가 어떻게 변하는지 보여줌
    freq 1 [-1.4935942e+00-6.0843825e-01j  1.0760520e+00-1.1441499e+00j ...
    freq 2 [-2.0126161e-05-3.6510662e-07j  1.0405716e-05-6.0332816e-08j ...
    ...
    freq n [-2.0316222e-05+0.0000000e+00j -1.0719299e-05+0.0000000e+00j ...
    세로 ⬇️ 주파수 낮은 순서에서 높은 순서 순  

인데 np.abs(librosa.stft(y, n_fft=4096)) 하게 되면 내부에 있던 진폭+위상 정보중에 진폭만 빠지고, 그게 다 양수로 바뀜 
    freq 1 [1.6127681e+00 1.5706580e+00 9.3536109e-01 ... 0.0000000e+00 ...
    freq 2 [2.0129472e-05 1.0405891e-05 5.9025228e-07 ... 0.0000000e+00 ...
    ...
    freq n [2.0129472e-05 1.0405891e-05 5.9025228e-07 ... 0.0000000e+00 ...

    ❌ “앞부분이 진폭, 뒷부분이 위상” => 뭐 이런거 아님! 
    (복소수 하나안에 두 정보가 얽혀서 들어있는 상태) 그냥 복소수 : a + bj = r* e^(jθ) => r(amplitude), θ(phase)
    => np.abs 는 벡터 길이만 남기고 방향(각도)는 버림
    & 그리고 길이는 항상 0이상이니까 전부 양수로 변하는 것 

    a + bj => (a, b) : 하나의 점 (벡터)로 해석해야 함. 화살표 길이(진폭), 화살표 방향(위상)
    => 따로 normalize 되지는 않음

    k번째 bin 주파수 : k * (fs/n_fft)
"""

"""2. time average (spectrum)

D =
[[1, 2, 3],
 [4, 5, 6]] 라면 2차원 배열. D.shape = (freq_bins, time_frames). 
    => 행(row) = freq / 열(column) = time
    - axis = 0 : 세로방향으로 내려가면서 계산
    - axis = 1 : 가로방향으로 옆으로 계산 => 각 주파수 bin(행) 에서 시간방향으로 평균냄

    freq1 : 시간 따라 값들 - 평균
    freq2 : 시간 따라 값들 - 평균
     ...
     => [4.4408590e-01 2.7636945e-01 7.9988092e-02 ... 9.5172271e-07 8.4451500e-07
 8.3812688e-07] 1차원 결과 나옴
     => [0.440  0.276  0.079 ... 0.000000951  0.000000844  0.000000838]

지수표기 : exponent notation 
ex. 1.23e-01 = 1.23 * 10^(-1) = 0.123
    1.23e+02 = 1.23 * 10^2 = 123

"""

"""3. log
np.log1p()
: 로그 그래프를 보면 x=0인경우, y 가 -무한대를 가지는데, 따라서 x값이 0일경우 -infinite 가 되게됨.
    => error 발생
    so, x+1을 해줘서 0->1로 바꾸는 함수가 np.log1p()

amp = np.array([0.01, 0.1, 1.0, 10.0]) -> 원래는 제일 작은 값과 큰 값의 차이가 0.01 <-> 10.0 1000배 차이났었지만
print(np.log1p(amp))
# 출력: [0.00995 0.0953 0.6931 2.3979]  -> 2.39와 0.009 의 차이로 2.4로 차이가 압축됨
        => 시각화, 파형추정, 배음 비율 계산시 극단적인 값에 덜 민감하다 !
        +동적 범위 압축 / 작은 배음/잔향도 비교가능 / 시각화와 인간 청각 지각에 맞춤


ex. librosa.stft(y, n_fft=16) 이렇게 전체를 16개의 밴드로만 나눠서 확인했는데 

    #2. time average (spectrum)
    spectrum = np.mean(D, axis=1)   
        #각 주파수 bin 별로 "시간에 따른 진폭 변화"를 평균 낸것 -> 이 소리에서 어떤 주파수가 전체적으로 많이 존재했는지 보려함
        #axis = 1 : 어느 방향으로 계산할건지 정함
    print(spectrum) # 확인용. => 여기서의 specrum은 주파수별 진폭평균값이 들어있는 1차원 배열

    #3. log scale (사람귀+필터 형태 반영)
    spectrum = np.log1p(spectrum)
    print("\n",spectrum) # 확인용. => 여기서의 spectrum

    #2 걍 spectrum = np.mean(D, axis=1)   
[2.2840254e-01 1.2754485e-01 1.0166810e-02 1.9733571e-03 7.2121876e-04
 3.3376538e-04 1.6506093e-04 7.2410432e-05 1.3176720e-05]

    #3 spectrum = log scale (사람귀+필터 형태 반영)
[2.0571458e-01 1.2004257e-01 1.0115475e-02 1.9714125e-03 7.2095881e-04
 3.3370970e-04 1.6504731e-04 7.2407813e-05 1.3176634e-05]
 => log 취한것 혹은 안 취한것 별 차이가 없음
   (소리 에너지가 주파수대역별로 고르게 분포되어 있거나, log 처리로 작은 차이가 압축된것)

"""

"""#4. 정규화

x = 10
x /= 2   # x = x / 2
print(x) # 5.0

[9.6256226e-02 6.3918136e-02 2.0156167e-02 ... 2.4929264e-07 2.2121085e-07
 2.1953755e-07] 
 => 정규화 되어서 0.096 0.063 0.0201 ... 등의 숫자로 바뀜
 => print(np.max(spectrum)) 이거 출력 시 1 이 나옴. 정규화 된것 .

"""

""" 5. np.gradient(spectrum)

+ 주파수에 따라 에너지가 어떻게 변하는지
    값이 큼 (+) : 갑자기 커짐 (에너지 상승)
    값이 작음 (-) : 급격히 떨어짐
    값이 0 근처 : 평평함 

+ 내부 계산 방식
  
1) 가운제 값들 (central difference. 중앙 차분)
    = (x[i+1] - x[i-1]) / 2

2) 양 끝 값들 
    = 맨 앞 : x[1] - x[0]
    = 맨 뒤 : x[-1] - x[-2]

ex. spectrum = [10, 20, 15, 5] => gradient = [+10 → +2.5 → -7.5 → -10]

index 0 (맨앞): 20-10     = 10
index 1 (중앙): (15-10)/2 = 2.5  => 양옆을 같이 보고 양쪽 평균 기울기 냄 
index 2 (중앙): (5-20)/2  = -7.5
index 3 (맨끝): 5-15 = -10

맨앞 : 급격히 증가(+10) 
중간 : 조금 증가(+2.5)
다음 : 감소 (-7.5)
맨끝 : 급격히 감소 (-10)


=> grandient 가 0이 되는 지점 peak 
 +수학에서는 (연속함수) f'(x) = 0 일때 극값 . + -> - 면 peak / - -> +면 valley(최소값)
 +우리는 discrete(배열) 미분의 근사값 이기 때문에 부호가 + -> - 로 바뀌고, 그 지점이 주변보다 커야 peak. 

[1:] : index 1 부터 끝까지. 0번째 값 제거하려고, log때문에 log(0) - undefined 

S[:, 10] : 10번째 시간 프레임
S[100, :] : 특정 주파수의 시간변화


"""

"""savgol_filter

전체 데이터: [a, b, c, d, e, f, g, h, i, j, k ...]

window_length=5 라고 하면 (설명 편하게 5로)

1번째 윈도우: [a, b, c, d, e] → 이 5개에 맞는 3차 곡선 그리기 → 가운데 c 값을 곡선값으로 교체
2번째 윈도우:    [b, c, d, e, f] → 이 5개에 맞는 3차 곡선 그리기 → 가운데 d 값을 곡선값으로 교체
3번째 윈도우:       [c, d, e, f, g] → ...

=> 
원본: [100, 1, 1, 1, 1]  ← 첫번째가 엄청 큼
           ↓ 이 5개에 3차 곡선 피팅
곡선: [80,  20, 5, 2, 1]  ← 가운데 c(=1) 가 5로 바뀜 => 곡선으로 피팅해서 자연스럽게 완만해짐. 튀는 값 완화

"""


""" -3dB

dB = 20*log1O(threshold / flat_mean) 

flat_mean = A_ref 역할
spectrum_v[i] = A  => dB = 20*log( A / A_ref )


-3dB = 20*log10(threshold / flat_mean)
-3/20 = log10(threshold / flat_mean)
10^(-3/20) = threshold / flat_mean
threshold = flat_mean * 10^(-3/20)  = flat_mean * (10**(-3/20))

= 약 0.708

"""

""" LPF 의 상위 20%, 평탄한 부분으 50%

LPF 가 걸린 신호 : 고주파 에너지 거의 없음
high_mean / overall_mean = 0.1 ~ 0.2 -> 0.5 미만 -> LPF 있다고 판단

LPF 없는 신호 : 고주파 에너지 충분히 살아있음
high_mean / overall_mean  = 0.6 ~ 0.9 -> 0.5 이상 -> LPF 없다고 판단 

"""



"""
✅ Things to study
- 복소수
- 미분의 이산버전(?)
- 중앙 차분 

"""

"""
git add .
git commit -m "feat : percussive vs harmonics 사운드 분류 기능 제작"
git push origin main

wip : work in progress. 아직 작업중(미완인 프로젝트 올릴때 쓰는 커밋 컨벤션(관습))

-commit type-
feat:     새로운 기능 추가
fix:      버그 수정
docs:     문서/주석 수정 (코드 동작 변경 없음)
style:    코드 포맷팅, 세미콜론 등 (로직 변경 없음)
refactor: 코드 리팩토링 (기능 변경 없음)
test:     테스트 코드 추가/수정
chore:    빌드, 설정 파일 수정

"""