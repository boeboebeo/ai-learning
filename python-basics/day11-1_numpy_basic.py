# day 11 numpy basics

import numpy as np
#numpy 라는 라이브러리를 np 라는 이름으로 별명 붙임
    #numpy.mean() = np.mean()
    #np. 하고 뜨는 목록들이 다 numpy 로 사용할 수 있는 함수 목록


bpms = np.array([171, 20, 114, 84, 110, 103, 95, 128, 76]) 
    # 이렇게 배열 만들어야지, boolean indexing 같은 numpy 기능 쓸 수 있음
    # 단순 list 를 numpy array로 변환하는 작업
print("BPM array:", bpms)
print(f"Average BPM: {np.mean(bpms):.1f}") 
    #mean : 평균 
    #np.mean(bpms) : numpy 의 미리 지정된 함수를 갔다 쓴것!
print(f"Max BPM : {np.max(bpms)}")
print(f"Min BPM : {np.min(bpms)}")
print(f"Std deviation : {np.std(bpms):.1f}")
    #np.std(bpms) : 표준 편차. BPM 이 평균에서 얼마나 퍼져있는지를 측정!
    #BPM 들이 비슷하면, std 낮음 / BPM 들의 폭이 크면, std 높음


#filtering
print(f"\nFast songs (BPM >= 110): {bpms[bpms >= 110]}")
    # Boolean Indexing 
    # bpms = np.array([171, 67, 114]) -> 아까 np로 이렇게 배열을 했기때문에 
    # bpms[bpms >= 110] -> true 인것만 뽑아주는 numpy에서만 되는 기능

print(f"Slow songs (BPM < 90): {bpms[bpms < 90]}")

#2D 배열 - [BPM , energy, valence] 
    #valence : 음악의 감정적 긍정도(높으면 ex. 0.9 -> 밝은 느낌)
song_features = np.array([
    [171, 0.8, 0.9],
    [67,  0.3, 0.2],
    [114, 0.7, 0.8],
    [84,  0.4, 0.3],
])
    # np.array([]) : 로 행렬배열을 해야지 -> 열단위로 데이터를 뽑을 수 있음

print("\nShape:", song_features.shape)
print("All BPMs:", song_features[:, 0]) # [:, 0] - 전체행의 0번째 열
print("All energies:", song_features[:, 1])
