# day 03 waveform estimation

import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

y, sr = librosa.load("audio_sample/sine.wav")   
    #frame lengh = 2048 (default)
    #22050 / 2048 = 약 10.77Hz (fmin <= sr / frame_length)
    #y = normalized 된 amp
    #sr = sample rate. default = 22050

def get_harmonic_energy(D, freqs, fundamental, n):
    target = fundamental * n    
    idx = np.argmin(np.abs(freqs - target)) 
                        #index : 전체 fft 한 그래프에서 target 값을 왜 빼지? -> 그럼 배음 없는 주파수만 남는거 아니야?
                        #np.argmin() : index of minimum -> 배열에서의 가장 작은 값의 위치(index)를 반환. 값 자체가 아니라 인덱스
                        #freqs 의 배열중에서 target freq 와 제일 간격이 작은 index 찾음 (아래의 argmin()확인)
    return np.mean(D[max(0,idx-2):idx+3, :])
    #D : 2차원 배열. D[...] 안의 내용은 배열에서 어떤 부분을 선택할지 지정하는 slicing.
    #ex. [2:5, :] -> 2~4번째 주파수 bin + 모든 시간 frame 선택하는 것(:, 모든 열)
    #max(0,idx-2) : 음수 index 나오면 안되니까 최소를 0으로 제한
    #target 주변 +/- 2 bin 의 범위로 -> 정확한 bin 값을 중심으로 안정적인 값을 얻기 위함 

def classify_Waveform(y, sr):
    # ── 1. Average spectrum (Harmonic) ────────────────────────
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)  #개수는 2048 //2 + 1 = 1025

    # ── 2. waveform estimation ─────────────────────────────────────────
    f0, _, voiced = librosa.pyin(y, fmin=43.066, fmax=sr/2) 
            #sr의 1/2 값이 max 여야 문제 안 생김 (fmin=21.533을 20으로 설정했었더니 문제가 있었음)
            #f min = 보다 sr/frame_length 가 커야함
    fundamental = np.nanmedian(f0)  #이 시간 전체에서 제일 대표적인 pitch 를 뽑는것

    if np.isnan(fundamental):
        print("Unknown (no pitch detected)") #return 은 def -> 이 함수 안에서만 사용가능함 그냥은 그냥 print 써야가능.

    print(f"\nFundamental freq : {fundamental:.1f} Hz ({librosa.hz_to_note(fundamental)})")
    #hz_to_note : 12-tet, equal temperament 기반 계산을 함
    #1 oct = 12 semi tones -> semitone ratio = 2^(1/12)
    # n = n=12⋅log2(440f) => n을 가장 가까운 반음으로 매핑 (반음번호로 변환)
    harmonics = list(range(1, 9))
    energies = [get_harmonic_energy(D, freqs, fundamental, n) for n in harmonics]

    fund_energy = energies[0]
    harmonic_energy = np.sum(energies[1:])

    if harmonic_energy / (fund_energy + 1e-8) < 0.1:
        return "Sine (단일파)"
    
    odd_energy = np.mean([get_harmonic_energy(D, freqs, fundamental, n) for n in [1, 3, 5, 7]])
    even_energy = np.mean([get_harmonic_energy(D, freqs, fundamental, n) for n in [2, 4, 6, 8]])
    total = odd_energy + even_energy
    odd_ratio = odd_energy / (total + 1e-8) 
        #1e-8 = 0.00000001 (1*10^(-8) = 0.00000001) - exponent notation
        #이렇게나 작은 값으로 계산해서 혹시 total이 0일 지도 모르니 계산 안정성을 주기 위한 아주 작은 수 

    print(f"\n홀수 배음 비율 : {odd_ratio:.2f}") #전체중에서 홀수배음이 차지하는 비율 
    print(f"짝수 배음 비율 : {1-odd_ratio:.2f}")

    decay = energies[1] / (energies[3] + 1e-8)  #wnd vs 4th 비교

    if odd_ratio > 0.75:
        if decay > 2:
            return "Triangle (급격한 감소)"
        else:
            return "Square (홀수배음 지배)"
        
    elif 0.4 < odd_ratio <= 0.75:
        return "Sawtooth (전체 배음 존재)"
    
    else:
        return "Complex / Other"

 

print(f"\n파형 추정 : {classify_Waveform(y, sr)}")

"""
+ pYIN : YIN algorithm의 확장판 (pitch detection용도)

    +YIN
       : 시간마다 "이 소리의 기본 주파수(F0)이 뭔지" 추정하는 알고리즘
       y  = [0.1, 0.3, -0.2, ...] 이런식의 진폭에 대한 데이터 -> 이걸로는 A4 인지 몇 Hz 인지 알 수 없음
       (주기(period)를 찾아서 -> 주파수로 바꾸는 과정이 필요)

       주기 T 
       ex. 440Hz -> T = 1(s)/440 => 약 0.00227sec
    원본:   ~~~~~~~~
    shift:    ~~~~~~~~ -> 잘 겹치면 주기를 찾을 수 있음
    =>but, YIN 의 약점은 노이즈에 약함, 잘못된 Pitch 를 잡기도 함

    +pYIN : YIN 에 probabilistic (확률 개념)을 추가함 => 기본음을 찾기 위함
        => 이게 pitch 일 확률을 계산함
        (f0, voiced_flag, voiced_prob)
        - f0 : 시간별 pitch 값 (f0 = fundamental freq 를 의미함. f1 = 두번째 배음)
        - voiced_flag : 이 프레임에 "소리 있음?" -> True or False 
        - voiced_prob : pitch 가 맞을 확률 (0~1)
         + nan : pitch 없음을 뜻함

    pyin 의 결과 f0 는 [440, 442, nan, nan, 439, 441, nan] -> 이런 구조. 
        +nan = pitch 못 찾은 구간(무음, 노이즈, 비주기 신호)
        - nanmedian = (1)Nan 제거-> (2)정렬 -> (3)가운데 값 선택 -> 중앙값 
            => 평균으로 쓰게 되면 튀는 값 (혼자 큰값) 에 약하기 때문에
            => 중앙값 (median)을 사용함

f0, _, voiced = librosa.pyin(y, fmin=21.533, fmax=sr/2) 
 => 에서의 fmin=20으로 했더니 나온 warning    
    UserWarning: With fmin=21.533, sr=22050 and frame_length=2048, 
    less than two periods of fmin fit into the frame, which can cause 
        => 하나의 프레임 안의 최소 pitch(min) 의 두개의 주기가 들어가지 않음 
           - pyin 내부에서 안정적인 pitch 검출을 위해서는 적어도 2주기 이상이 필요함
    inaccurate pitch detection. Consider increasing to fmin=21.533 or frame_length=2051.

"""

"""
import numpy as np


(1)간단한 1차원 배열일때

a = [5, 2, 7, 1, 3]
inx = np.argmin(a)
print(idx) #출력 : 3 (제일 작은 1이 있는 3번 인덱스가 출력됨)
pirnt(a[idx]) #출력 : a[3] = 1출력 (3번째 인덱스의 실제값 1 출력)

(2)abs 와 함께 사용하는 배열일때

freqs = [10, 20, 30, 40]
target = 22
idx = np.argmin(np.abs(np.array(freqs) - target))
print(inx) #출력 : 1 (1번 인덱스와 제일 가까움으로)

"""

