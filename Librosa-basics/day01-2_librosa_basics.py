# day 01 librosa basics 

#library

import librosa  #오디오 분석 엔진(소리 -> 숫자. 의미있는 feature 을 추출함) => numpy 배열(숫자)로 보여줌
import librosa.display  #오디오 용 "시각화 보조툴" -> librosa 결과를 matplotlib에 맞게 그려줌 (x, y축에 맞게 변환) ex. librosa.display.specshow(D, x_=, y_axis="log")
                        #시간축이나, y축을 보기좋게 scaling 해줌 ex. log : Hz 를 log 스케일로 보여준다던지
import matplotlib.pyplot as plt     # 그래프 그리는 본체
import matplotlib.font_manager as fm  # 폰트 관리 (어떤 글꼴을 쓸지 제어) -> 밑에서 우선은 안쓰고 있음 ex. fm.findSystemFonts() : 시스템 폰트 목록 확인
import numpy as np

plt.rcParams["font.family"] = 'AppleGothic' #그래프에서 한글 깨짐 방지를 위해 씀 . 그리고 '' 랑 "" 의 차이 : 파이썬에서는 완전히 동일하다
plt.rcParams['axes.unicode_minus'] = False  # "-"" 깨지는 문제 해결  -> 이거 없으면 걍 네모로 나옴

"""
    rcParams : 설정 딕셔너리(config)
    matplotlib 의 전체 기본 설정값을 모아둔 딕셔너리 인데, 거기에서의 font.family 라는 키값의 Value 를 AppleGothic 으로 바꾼것 
{
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    ...
} => 이런식의 설정 딕셔너리에서 설정값을 다 바꾸기 위함     

+ 한번에 여러개 다 바꾸려면 :
    plt.rcParams.update({
        "font.family": "AppleGothic",
        "font.size": 12
})

"""

y, sr = librosa.load("Librosa-basics/audio_sample/noise.wav")   #file -> waveform 으로 변환
    #librosa.lod ("_") : 하나의 함수가 두개의 값을 "tuple" 형태로 반환함
    #return y, sr 을 내부에서 반환함 
    # y : 오디오 데이터(numpy array) normalized amplitude! : [0.01, -0.02, 0.03, ...]
    # sr : 샘플링 레이트(int) : 22050 (1초당 샘플 개수)
    #기본적으로 librosa 는 모든 파일을 같은 기준으로 맞추려고 자동 리샘플링 함 44100이던 48000이던 -> 22050
    #이걸 원본으로 유지하려면 ("_+_.wav", sr=None) 표기 추가 & 자동으로 mono 로 변환됨 (, mono=False) 표기 추가 하면 원본 유지

fig, axes = plt.subplots(4, 1, figsize=(8, 8))   #4행 1열짜리 그래프 공간(subplot)+여러개의 그래프 슬롯도 한번에 만들고, 전체 도화지(fig)랑 각각의 영역(axes)를 가져와라 (14,12) = inch 단위
fig.suptitle("Day 1 - Sound basic analysis", fontsize=14)   #전체 제목

# ── 1. 파형 ───────────────────────────────────────────────────
librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4A90D9")
    #시간 영역의 신호(y:normalized amplitude)를 사람이 보기 쉽게 그려주는 함수
    #waveshow definition 들어가면 * 표시가 있는데, 그 위는 그냥 위치로 y 이렇게만 써도 되지만, 그 밑은 다 sr=, ax= 이렇게 key를 제시하고 써야함
    #waveshow(y, 44100)  ❌ -> 이건 에러! (sr은 keyword-only)
    #model.fit(X=X, y=y) 처럼 이렇게 그냥 일부러 이름 맞춰서 씀 
axes[0].set_title("Waveform")   #개별 행의 제목
axes[0].set_xlabel("Time(second)")
axes[0].set_ylabel("Amplitude")

# ── 2. 스펙트로그램 ───────────────────────────────────────────
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  #가장 큰 값을 0dB 기준으로 삼음
    #waveform -> STFT -> magnitude -> dB변환 (linear -> log)
    #소리 -> 시간-주파수 에너지(dB)로 바꿔서 사람이 보기좋게 만듦
    # ❗️ librosa.stft(y) : 복소수 행렬(complex matrix)로 출력해줌 - 시간(frame) x 주파수(bin) 의 형태로. 각 값 = 진폭+위상 정보 => 아직 "크기"가 아니라 복소수 상태임
    # ❗️ np.abs(librosa.stft(y)) : |a+ bi| = sqrt(a^2 + b^2) - 각 주파수의 세기(amplitude)
    # ❗️ librosa.amplitude_to_db(np.abs(librosa.stft(y)) : dB로 변환!
    # 내부에서 하는 계산 : 20*log10(amplitude/ref) 근데 여기서의 ref 는 가장 큰 값이 기준임(0dB로 가장 큰값을 맞춰버림)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=axes[1], cmap="magma")
    #D(시간 * 주파수 에너지)를 색으로 표현해서 그림으로 그림
    #sr=sr : STFT 는 프레임 단위이기 때문에 sr 이 있어야 second 단위로 변환 가능
    #x_axis = "time" : 프레임번호를 second 로 변환해서 보여줌
    #y_axis = "log" : 주파수 축을 로그로 변환해서 보여줌 (낮은 주파수 -> 자세히, 높은 주파수 -> 압축) : 인간의 청각 구조와 맞추기 위함 
    #cmap = "magma" : 이건 약간 색상 프리셋같은 느낌 
        # ex. 
            # cmap="viridis"
            # cmap="plasma"
            # cmap="inferno"
            # cmap="gray"
axes[1].set_title("spectrogram(STFT)")

"""(1) librosa.stft(y)
    print(np.abs(librosa.stft(y)))  -> return : complex matrix (복소수 행렬) 출력 : 크기 + 위상
    => 해당 주파수 bin 대역이 시간에 따라서 어떻게 변하는지 보여주는 배열 
    시간 ----------------> (각 주파수 성분의 세기(amplitude)+위상)가 시간에 따라 어떻게 변하는지 보여줌
    freq 1 [-1.2076843e+00+0.0000000e+00j  3.8955867e+00+0.0000000e+00j ...
    freq 2 [-2.7485041e-02-2.0390887e+00j -5.4297032e+00+1.4911677e-01j ...
    ...
    freq n [ 7.8668876e-05-1.2007133e-06j  9.5071908e-07-4.4977140e-05j ...
    세로 ⬇️ 주파수 낮은 순서에서 높은 순서 순 (freq bin)
    """

# print(librosa.stft(y))  #얘는 크기 + 위상 (complex) 보여줌 -> +0.0000000e+00j 존재
# print("\n", np.abs(librosa.stft(y))) #얘는 크기만 (magnitude)만 보여줌 . abs 는 단순히 부호만 없애는게 아니고, 복소수 -> 에너지 크기만 남기는 과정임
# print(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max))

"""(2) np.abs(librosa.stft(y))
        -> 크기만 보여줌 (magnitude)
    : freq 1 [1.2076843e+00 3.8955867e+00 8.0116825e+00 ... 1.3038161e-01 ...
    : freq 2 [1.0714541e-04 4.0306164e-05 4.0123255e-06 ... 8.0193950e-06 ...
    ...-> 시간에 흐름에 따른 각 주파수의 진폭 변화

"""

"""(3) librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max) 
        -> np.abs(librosa.stft(y)) 는 linear 한 값이라서 사람이 느끼는 방식과는 맞지 않음
            => 인간 청각에 맞게 log 로 바꿈! (1->2 (두배), 10 ->20 (두배). 절대적인 량은 1, 10으로 차이가 많지만 비슷한 증가로 느낌)
                ❗️ 로그 스케일로 변환 필요!
            => dB = 20 * log10(S/ref) -> 가장 큰 값을 기준(0dB)로 잡을 예정
            => 그럼 최대값은 0dB가 되고, 나머지는 음수 dB가 됨 
                (in spectrogram : 밝은 부분 - 0dB근처, 어두운 부분 - -80dB근처)

    : freq 1 [-44.77568  -34.60329  -28.340275 ... -64.11042  -46.72722  -43.886765]
    : freq 2 [-44.732655 -31.885052 -28.374023 ... -64.45378  -47.757317 -45.20644 ]
    ...
    : freq n [-80.       -80.       -80.       ... -80.       -80.       -76.22669 ]]
        => 제일 작은 값이 -80dB 으로 limiting 되는 이유는 원래대로인 -무한대로 색상을 매핑하면 색상스케일이 무너지기 때문 -> 정규화/스케일링 불가능
        [0, -20, -40, -∞, -∞, -∞] -> [0, -20, -40, -80, -80, -80]
    
    ❗️ librosa.power_to_db() : power -> 10* log10()
    ‼️ librosa.amplitude_to_db() : amplitude -> 20*log10()
    
"""


# ── 3. MFCC ───────────────────────────────────────────────────
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfcc, x_axis="time", y_axis="mel", ax=axes[2], cmap="coolwarm", fmax=16000)
axes[2].set_title("MFCC - 음색지문")
    #MFCC : 사운드의 음색지문 같은것 = Mel-Frequency Cepstral Coefficients
        #Mel Freq : 사람 귀의 주파수 인식에 맞춘 로그 스케일
        #Cepstral : 스펙트럼을 다시 주파수 영역으로 변환한것
        #Coefficients : 계수들 -> 특징 숫자들 . 
            # => 사람이 느끼는 음색을 압축해서 숫자로 표현한 것.
            # y = 아까 불러온 오디오 파일에서의 normalized amplitude array (numpy 1차원 배열)
            # spectrogram 은 두 악기가 거의 비슷하게 보일 수 있지만 (같은음, 진폭이라면) -> 데이터 양이 많기때문에 악기 구분 효율이 떨어짐 : 시간*주파수 확인
            # MFCC 는 Mel scale 을 적용해서 harmonic env + timbre 특징을 압축해서 보여줌. 더 명확 : 음색/ 악기 특징을 압축 -> AI 입력 최적화
            #13~20개의 계수로 줄임 
    #나중에 신스 식별시에 이걸 AI 에 먹이게 됨

"""
1) 원본 STFT : 시간 * 주파수 전체에너지 -> 각 주파수 bin 수백~수천개 가능
    => 1초짜리 피아노 음 : STFT = 1025freq * 43 frames = 44075 데이터 (n_fft=2048이면 FFT 결과 길이 = 2048/2 + 1 = 1025 (복소수FFT 대칭 성질 때문에))
        + 전체 시간은 43개로 나누고, 전체 주파수는 1025개로 나눔 (1개의 audio bin : 한개의 주파수 구간)
2) MFCC : Mel scale + DCT 로 압축 (13~20개의 계수로 줄임) -> 정보 압축 
    => 13계수 : 43 frames = 559 데이터 ( 약 80배 압축 )
    + MFCC : Mel filter bank -> log -> DCT -> n_mfcc 개 계수만 사용
        => n_mfcc = 13 이면 1025개의 주파수 정보를 13개의 숫자로 요약 (인간 귀가 느끼는 음색 기준으로 중요한 정보만 남김)

mel scale : m = 2595 * log10(1+f/700) 
    f : 실제 Hz
    m = Mel scale 값 (저주파는 촘촘히, 고주파는 넓게)

DCT : Discrete Cosine Transform. Log mel spectrum -> DCT 적용 
    => 1025 freq → Mel filter → 40 Mel bands → log → DCT → 13 계수
"""


# ── 4. RMS 에너지 (엔벨로프 미리보기) ────────────────────────
rms = librosa.feature.rms(y=y)[0]
times = librosa.times_like(rms, sr=sr)
axes[3].plot(times, rms, color="#E85D8A", linewidth=1.5)
axes[3].fill_between(times, rms, alpha=0.3, color="#E85D8A")  #그래프 아래 색 채우기 
axes[3].set_title("RMS energy - envelope")
axes[3].set_xlabel("time(seconds)")
axes[3].set_ylabel("energy")

# print(rms) 시에 [0]이 안붙어 있다면 [[]] -> 이렇게 2차원 배열이 출력 됨 
# 우리는 안쪽의 1행만 필요하니 [0] 붙여야 함

plt.tight_layout()  #그래프 레이아웃 자동으로 정리해주는 함수
plt.savefig("Day1_result.png", dpi=150)
plt.show()  
    #얘는 blocking 함수라서 이 창을 끌때까지 이 밑 코드들은 계산이 안됨
    #밑 print 코드를 위거 위로 올리거나, plt.show(block=False) -> 로 실행하면 창 뜨면서 바로 아래 코드 실행됨

# ── 기본 정보 출력 ─────────────────────────────────────────────
info = {
    "sample rate" : f"{sr} Hz",
    "duration" : f"{len(y)/sr:.2f} seconds",
    "samples" : f"{len(y):,} samples",
    "max amplitude" : f"{np.max(np.abs(y)):.4f}",
    "RMS average" : f"{np.mean(rms):.4f}",
    "MFCC shape" : f"{mfcc.shape} <-13개 계수, 시간 프레임 수"
}

for k, v in info.items():
    print(f"{k:<18}: {v}")




""" (아래와 같이 되어있으면 보기 불편함 ( : 까지의 길이 맞추기))
print(f"samplerate: {sr}Hz")
print(f"duration: {len(y)/sr:.2f}seconds")
print(f"sample: {len(y):,} samples")
print(f"max amplitude: {np.max(np.abs(y)):.4f}")
print(f"RMS average: {np.mean(rms):.4f}")
print(f"MFCC shape: {mfcc.shape} <- 13개 계수, 시간 프레임 수")
"""

"""
✅ Things to study
 - MFCC
 - DCT
 - Cepstral
 - mel scale , mel spectrogram

"""