import numpy as np

# 곡들의 특성 데이터 [bpm, energy, valence, loudness]
song_features = np.array([
    [171, 0.8, 0.9, -4.2],
    [67,  0.3, 0.2, -12.1],
    [114, 0.7, 0.8, -5.3],
    [84,  0.4, 0.3, -9.8],
    [128, 0.9, 0.7, -3.1],
    [95,  0.5, 0.5, -7.4],
])

bpms = np.array(song_features[:, 0])
energes = np.array(song_features[:, 1])
#얘를 꼭 np.array() 로 처리해줘야 boolean indexing 가능 
    #여기서 np.array([song_features[:, 1]]) -> 이렇게 또 대괄호[]로 감싸버리면 리스트가 되어 나와버리기 때문에 [] 빼기!

print(f"Average BPM: {np.mean(bpms):.1f}")
print(f"Max BPM: {np.max(bpms):.0f}")
print(f"Min BPM: {np.min(bpms):.0f}")
print(f"Std deviation: {np.std(bpms):.1f}")

print(f"\nAverage energe: {np.mean(song_features[:,1])}")
print(f"Average Loudness : {np.mean(song_features[:,3]):.1f}")

print(f"\nFast songs (BPM >= 100): {bpms[bpms >= 100].astype(int)}") # 다 깔끔하게 int 형으로 만들어버리려고 .astype(int) 로 만듦
print(f"High Energy (Energy >= 0.6) : {energes[energes >= 0.6]}")



"""
출력 형태

Average BPM: 109.8
Max BPM: 171
Min BPM: 67
Std deviation: 33.7

Average energe: 0.6
Fast songs: [171. 114. 128.] 
    -> numpy 에서는 정수 + 소수 섞이면 전부 float 로 통일 해버림 ( 같은 type 이여야 해서 )
    => 깔끔하게 출력하고 싶으면 {bpms[bpms >= 100].astype(int)}넣으면 됨 
Energy >= 0.6 : [0.8 0.7 0.9]

"""