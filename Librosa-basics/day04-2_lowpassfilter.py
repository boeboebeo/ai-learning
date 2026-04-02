#day 04-2 low pass filter - COF , RES 구하기

import librosa
import numpy as np
import matplotlib.pyplot as plt

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

    #5. 기울기 (배열의 변화율(기울기)를 구하는 함수) . 미분의 이산버전 
    slope = np.gradient(spectrum)
    # print(slope)
    with open("slope.txt","w") as f: # (1) 이렇게 해서 그래프 txt로 보거나
        for i, val in enumerate(slope):
            f.write(f"{i} {val}\n")

    plt.plot(freqs[1:], spectrum[1:], label="spectrum") #(2) 이렇게 해서 spectrum 확인
    plt.plot(freqs[1:], slope[1:], label="slope")   #error : x , y 가 같은 n_fft 기준으로 만들어야 함 
    plt.xscale("log")
    plt.legend() # 각 그래프 뭔지 알려줌 
    plt.grid(which="both")
    plt.show()



    #6. cutoff 찾기 . 가장 급격히 떨어지는 지점 -> steepest drop 찾기
    cutoff_idx = np.argmin(slope) #가장 급격하게 하강하는 지점의 index 알려줌 
    cutoff_freq = freqs[cutoff_idx]

    #7. resonanace estimation . cutoff 주변 peak 존재 여부
    region = spectrum[max(0, cutoff_idx-5):cutoff_idx+5] #마지막 인덱스 포함안돼서 10개의 값 불러와짐
    print(region)

    peak = np.max(region)
    base = spectrum[cutoff_idx]  #여기를 (cutoff_idx)로 불러오려해서 에러가 났었음. 함수가 아닌데 왜 ()를 붙여서 실행하려고 하냐.. 해서 난 에러 

    # resonance = peak - base #피크 강조 정도. peak(컷오프 주변 값 중) 가 base(컷오프 지점 magnitude)보다 높으면 resonance = 공진
    resonance_ratio = peak / (base + 1e-8)  #비율로 보기

    #8. 결과정리
    if resonance_ratio > 1.5 :
        resonance_label = "High Resonance"
    elif resonance_ratio > 1.2 :
        resonance_label = "Mediun Resonance"
    else:
        resonance_label = "Low Resonance"
    
    return cutoff_freq, resonance_label

# 사용
y, sr = librosa.load("audio_sample/saw+LPF(5000hires).wav")

cutoff, res = estimate_lpf(y, sr)

print(f"LPF cutoff : {cutoff:.0f}Hz")
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



"""
✅ Things to study
- 복소수
- 미분의 이산버전(?)
- 중앙 차분 

"""