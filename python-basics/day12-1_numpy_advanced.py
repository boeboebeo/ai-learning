# day 12 - numpy advanced 
#실제 오디오 신호를 numpy 로 표현하기

import numpy as np

sample_rate = 44100
duration = 1.0 
t = np.linspace(0, duration, int(sample_rate*duration))  
    #np.linspace : 0부터 duration까지 세번째 값으로 균일하게 나눔 
    #linspace : 0부터 1.0까지 44100개로 나눔 => 44100개의 시간포인트가 생김
    # → [0, 0.0000226, 0.0000453, ... 1.0] : 균일하게 나눠진 숫자 배열 만들어줌


#A4 = 440Hz
#np.sin() : 수학적으로 사인파를 계산함
#np.sin(2*np.pi*440*t) = 2pi * freq * t => 물리학/수학에서 주파수를 각속도로 변환하는 공식
#t : 각 시간 포인트에서 지금 진폭이 어디쯤인지 알기위해서!!!! 
#np.pi = 3.1415926535...(numpy가 미리 저장해둠)= pi 대신 3.14라고 쓰면 명확한 주파수 안나옴


audio_signal = np.sin(2*np.pi*440.0*t)
    #공기진동 모델링!
    #한 바퀴는 2π
    #y=sin(2πft) / f = freq / t = 시간()
    # t 라는 배열 전체가 한번에 연산됨 ! (벡터 연산)

"""
내부적으로 벡터연산이 일어나는 상태 
[
 sin(2π440 * 0),
 sin(2π440 * 0.0000226),
 sin(2π440 * 0.0000453),
 ...
 => 하나하나의 샘플마다 각각의 진폭값을 구해서 배열한다
]
"""


"""
`t` 의 44100개 포인트마다 각각 계산:

샘플 0    : t=0.0000000초 → 진폭 = 0.0
샘플 1    : t=0.0000226초 → 진폭 = 0.062...
샘플 25   : t=0.0005670초 → 진폭 = 1.0  ← 첫 번째 최대값
샘플 50   : t=0.0011340초 → 진폭 = 0.0
샘플 75   : t=0.0017010초 → 진폭 = -1.0 ← 첫 번째 최소값
샘플 100  : t=0.0022680초 → 진폭 = 0.0  ← 한 주기 완성!

=> 최대값도 440번 나옴. 

"""

print("===Audio Signal===")
print(f"Total samples : {len(audio_signal)}") 
    #샘플의 개수 = 오디오길이*샘플레이트
    #1초짜리 audio = 44100*1 = 44100 samples
print(f"Range : {audio_signal.min():.2f} ~ {audio_signal.max():.2f}")
    #오디오 신호는 보통 -1 ~ 1 사이 
    #값이 이것보다 크면 클리핑 위험 있음

# C major chord (도 미 솔)
c4 = np.sin(2*np.pi*261.63*t)
e4 = np.sin(2*np.pi*329.63*t)
g4 = np.sin(2*np.pi*392.00*t)
chord = (c4 + e4 + g4) / 3  #진폭 줄이기 위해 3으로 나눔

print("\n===C Major Chord===")
print(f"Shape : {chord.shape}") 
    #shape : 배열의 구조(크기, 차원). 이 데이터가 몇 차원이고 각 차원에 몇개가 있는지 알려주는 함수
    #chord => 오디오 신호이기 때문에 1차원 배열 (시간에 따라 값만 있음)
    #(44100, ) : 1차원 배열이라는 뜻 . 
    #(44100, 2) : 2차원 배열 . (2 : L, R)
print(f"Max amplitude : {chord.max():.2f}")
    #신호의 최대값을 출력해서 소리가 얼마나 큰지 확인
