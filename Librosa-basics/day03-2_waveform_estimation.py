# day 03 waveform estimation

import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

y, sr = librosa.load("audio_sample/square_2.wav")   
    #frame lengh = 2048 (default)
    #22050 / 2048 = 약 10.77Hz (fmin <= sr / frame_length)
    #y = normalized 된 amp
    #sr = sample rate. default = 22050

def get_harmonic_energy(D, freqs, fundamental, n):
    target = fundamental * n    
    idx = np.argmin(np.abs(freqs - target)) 
                        #index
                        #np.argmin() : index of minimum -> 배열에서의 가장 작은 값의 위치(index)를 반환. 값 자체가 아니라 인덱스
                        #freqs 의 배열중에서 target freq 와 제일 간격이 작은 index 찾음 (아래의 argmin()확인)
                        #실제 주파수 는 bin center 가 아님 => 에너지 퍼짐(leakage)
    return np.mean(D[max(0,idx-2):idx+3, :])

def classify_Waveform(y, sr):
    # ── 1. Average spectrum (Harmonic) ────────────────────────
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)  #개수는 2048 //2 + 1 = 1025

    #noise 여부 파악
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    if np.median(flatness) > 0.5:   #median() : flatness 가 array 라면 그 중 가운데 값을 출력
        return {
            "type" : "Noise",
            "fundamental" : None
        }   # 아 여기선 이렇게 return 해버리고, noise 가 아닐경우 계속 계산되게 함! <- 여기서 끝

    # if np.nanstd(f0) > 50:   => 노이즈 추출 다른 방법 
    #     waveform = "Noise"

    # if np.mean(voiced) < 0.5 :
    #     waveform = "noise"

    # => 노이즈 아니면 계속 진행!
    # ── 2. waveform estimation ─────────────────────────────────────────
    f0, _, voiced = librosa.pyin(y, fmin=43.066, fmax=sr/2) 
            #sr의 1/2 값이 max 여야 문제 안 생김 (fmin=21.533을 20으로 설정했었더니 문제가 있었음)
            #f min = 보다 sr/frame_length 가 커야함
    fundamental = np.nanmedian(f0)  #Nan(값없음)을 제외하고 f0의 대표값을 뽑음. => 남은 값 들중 중앙값을 선택 (평균은 outlier(이상치) 에 약하니까)
        #f0 = [440, 441, 1000, 439, np.nan] 의 값이라면 nanmedian => 440

    if np.isnan(fundamental):
        return {        #return 을 튜플로 받아서 좀더 깔끔하게 print할 수 있게 됨
            "type" : "Unknown",
            "fundamental" : None
        }
    #if np.isnan(fundamental): #이게 nan 인지? : nan (피치를 못찾겠는 상황)
       # return("Unknown (no pitch detected)") #return 은 def -> 이 함수 안에서만 사용가능함 그냥은 그냥 print 써야가능.
        # pitch가 아님 -> non-perodic (애초에 f0 기본 주파수가 정의 안 됨)
        # Nan (데이터/계산 결과) -> 알고리즘이 값을 못 냈다
        # 실제로 노이즈에도 에너지 피크가 존재해서 랜덤 피크를 기음 or 배음처럼 잡아버림

    
    #hz_to_note : 12-tet, equal temperament 기반 계산을 함
    #1 oct = 12 semi tones -> semitone ratio = 2^(1/12)
    # n = n=12⋅log2(440f) => n을 가장 가까운 반음으로 매핑 (반음번호로 변환)
    harmonics = list(range(1, 9))
    energies = [get_harmonic_energy(D, freqs, fundamental, n) for n in harmonics]
    print(energies)
        #energie : 이론적 amplitude 가 아니고, STFT magnitude 기반의 평균 에너지

    #sine 판별
    fund_energy = energies[0]
    harmonic_energy = np.sum(energies[1:])

    if harmonic_energy < fund_energy * 0.1 :
        return {
            "type" : "Sine",
            "fundamental" : fundamental
        }
    
    odd_energy = np.mean([energies[n] for n in [0, 2, 4, 6]]) #list(1, 9) 에서의 energies[0] : 0번 인덱스가 1f, 1번 인덱스가 2f이니까 0부터 시작해야함
    even_energy = np.mean([energies[n] for n in [1, 3, 5, 7]])
    total = odd_energy + even_energy
    odd_ratio = odd_energy / (total + 1e-8) 
        #1e-8 = 0.00000001 (1*10^(-8) = 0.00000001) - exponent notation
        #이렇게나 작은 값으로 계산해서 혹시 total이 0일 지도 모르니 계산 안정성을 주기 위한 아주 작은 수 

    print(f"\n홀수 배음 비율 : {odd_ratio:.2f}") #전체중에서 홀수배음이 차지하는 비율 
    print(f"짝수 배음 비율 : {1-odd_ratio:.2f}")

    decay = energies[2] / (energies[4] + 1e-8)  #wnd vs 4th 비교

    #return fundamental  #오케이 return 은 함수안에서 만나는 순간 그 밑의 값들은 다 실행이 안나고 끝나버림! 
                        #그래서 return 할 값 을 모아서 한번에 처리할 예정

    print(decay)

    if odd_ratio > 0.75:
        if decay > 2:
            waveform = "Triangle (급격한 감소)"
        else:
            waveform = "Square (홀수배음 지배)"
        
    elif 0.4 < odd_ratio <= 0.75:
        waveform = "Sawtooth (전체 배음 존재)"
    
    else:
        waveform = "Complex / Other"

    return {
        "type" : waveform,  #함수 내에서 return을 위해서 해버리면 그 밑은 더이상 실행이 안되기 때문에 
        "fundamental" : fundamental
    }                                # => 맨 밑에다가 뺄 변수들 다 모아서 출력
    

result = classify_Waveform(y, sr)  #이렇게 지역변수를 빼올 수 있음
    

print(f"파형 추정: {result['type']}")

if result["fundamental"] is None:
    print("Fundamental freq : 없음")
else:
    f0 = result["fundamental"]
    print(f"Fundemental freq : {f0:.1f}Hz ({librosa.hz_to_note(f0)})")

"""
print(energies) 하면 아래와 같이 square 파형의 배음 구조에 따른 에너지가 나오는데
: [np.float32(9.513988), np.float32(0.006657973), np.float32(3.108985), np.float32(0.004048447), np.float32(1.8963943), np.float32(0.0026647737), np.float32(1.359458), np.float32(0.0021029632)]
    => 이랬을때 
    decay = energies[2] / (energies[4] + 1e-8) 여기서 energies[0]과 [2]를 비교했을때 
    3이 나와서 여전히 triangle 로 분류됨
    => 근데 [2], [4]를 비교할 경우, 3.10 / 1.89 라서 1.64 정도 나옴
        => 그래서 square 로 잘 분류 할 수 있음

"""



# print(f"\n파형 추정 : {waveform}")
# print(f"\nFundamental freq : {fundamental:.1f} Hz ({librosa.hz_to_note(fundamental)})")
    # 함수 안에서만 존재하는 "fundamental"변수는 -> local variable : 지역변수

# if waveform == "Noise":
#     print("파형 추정: Noise")
#     print("Fundamental freq : 없음")
# elif fundamental is None:
#     print(f"파형 추정 : {waveform}")
#     print("Fundamental freq : 없음")
# else:
#     print(f"파형 추정 : {waveform}")
#     print(f"Fundamental freq : {fundamental:.1f}Hz ({librosa.hz_to_note(fundamental)})")
 # 이거 좀 비효율적



