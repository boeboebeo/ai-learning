import librosa
import numpy as np

y, sr = librosa.load("Librosa-basics/audio_sample/noise.wav")

D = librosa.stft(y)
print(len(D)) 
    # n_fft = 2048이 기본값이므로, D의 길이는 2048/2 + 1
    # 1025
print(D[1][:3])
    # D 는 '주파수*시간수'를 담은 2차원 복소수 배열을
    #  [[]] 이렇게 출력
    # D.shape = (주파수_bin 수, 시간_frame 수) = D[f, t] = magnitude + phase 정보
    # [2.468402 -1.5913999j  2.3116953+0.85510653j 1.6040945+1.1924436j ]
        # 첫번째 인덱스의 3번째 시간까지의 정보

print(D[1, :]) # 첫번째 주파수 bin에서의 전체 시간 프레임을 출력
print(D[1][:3]) # 첫번째 주파수 bin에서의 처음 3개 시간 프레임만 출력

    #따라서 D[1, :3] = D[1][:3] 둘은 같은 같은 표현

    #D[f, t]
        # f = 주파수 index
        # t = 시간 index
    
""" D의 형태
    [
    [f0 freq bin], -> 각각은 주파수 행 -> 0Hz (범위가 아닌 딱 그구간을 말함)
    [f1 freq bin], -> 약 10.77Hz
    [f2 freq bin], -> 약 21.54Hz
    ]
        그 주파수 행 안의 a, b, c, 이렇게는 시간의 흐름별
    """

# 각 주파수 간격(sr=22050, n_fft=2048일때) 
    # = 22050 / 2048 = 약 10.77Hz
# 전체 D의 개수는 2048/2 + 1 = 1025 (Nyquist)



