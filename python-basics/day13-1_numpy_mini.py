# day 13 - numpy mini project

import numpy as np

# 10곡의 feature 데이터
# [BPM, energy, valence, danceability]
features = np.array([
    [171, 0.8, 0.9, 0.7],  # Blinding Lights
    [67,  0.3, 0.2, 0.2],  # Someone Like You
    [114, 0.7, 0.8, 0.8],  # Dynamite
    [84,  0.4, 0.3, 0.3],  # Stay With Me
    [103, 0.8, 0.9, 0.8],  # Levitating
    [97,  0.5, 0.7, 0.6],  # Golden Hour
    [110, 0.6, 0.7, 0.7],  # Peaches
    [95,  0.7, 0.8, 0.7],  # Watermelon Sugar
    [135, 0.4, 0.6, 0.6],  # Bad Guy
    [96,  0.8, 0.9, 0.8],  # Shape of You
])

feature_names = ["BPM", "Energy", "Valence", "Danceability"]

print("===Feature Statistics===")

# for key, value in my_dict.items(): -> 이게 key , value 값 같이 가지고 오는 것 

for i, name in enumerate(feature_names): #인덱스(번호. 0부터 불러옴), 값 같이 가지고 오는것 
    col = features[:, i]    # 그럼 여기서는 features 안에 들어있는 0번의 열 전체를 불러오는 것 
                            # : 모든 행 , i : i번째 열
    print(f"{name}:")
    print(f"    Mean: {np.mean(col):.2f}")  
    print(f"    Std : {np.std(col):.2f}")
    print(f"    Min : {np.min(col):.2f}")
    print(f"    Max : {np.max(col):.2f}")   #하나씩 (0, "BPM") 이렇게 세트로 불러와서 for 문 돌리는것 

    if name == "BPM":
        col = col.astype(int) #지금 BPM 만 정수형이라서 뒤에 . 붙으므로 그것만 뒤에 점 안붙여서 출력하려면 
    print(col)

#상관관계 - BPM 과 energy의 관계

bpm = features[:, 0]
energy = features[:, 1]
correlation = np.corrcoef(bpm, energy)[0, 1] #[행, 열] -> 0번째 행,  첫번째 열

print(f"\nBPM vs Energy correlation : {correlation:.2f}")

print("\n", correlation)

"""
corrcoef : 
[ bpm 데이터 ] => 하나의 벡터 (길이 10)
[ energy 데이터 ] => 하나의 벡터 (길이 10)

=> 이렇게 두 벡터를 불러와서 2 * 2 의 행렬을 만듦 

[[corr(bpm, bpm),     corr(bpm, energy)],
 [corr(energy, bpm),  corr(energy, energy)]] -> 이렇게! 

 =

 [[1.0,   r], bpm vs bpm , energy vs energy 는 자기 자신이므로 1 이 나옴
 [r,   1.0]] => r 이 우리가 궁금한 값 

 Pearson r : Pearson correlation coefficeint (math formula)

실제상황에서는 곡단위로 최대 RMS = 1 로 정규화를 함
    => 곡 전체 레벨에 상관없이 곡 안에서 상대적 에너지를 측정

    https://spotifywebapipython.readthedocs.io/en/latest/spotifywebapipython/models/audiofeatures.html?utm_source=chatgpt.com

"""



