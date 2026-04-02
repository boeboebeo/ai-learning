# day 17 - pandas mini project

import pandas as pd
import numpy as np

data = {
    "title": ["Blinding Lights", "Someone Like You", "Dynamite",
              "Stay With Me", "Levitating", "Golden Hour",
              "Peaches", "Watermelon Sugar", "Bad Guy", "Shape of You"],
    "artist": ["The Weeknd", "Adele", "BTS", "Sam Smith",
               "Dua Lipa", "JVKE", "Justin Bieber", "Harry Styles",
               "Billie Eilish", "Ed Sheeran"],
    "bpm": [171, 67, 114, 84, 103, 97, 110, 95, 135, 96],
    "genre": ["Pop", "Ballad", "Pop", "Ballad", "Pop",
              "Pop", "R&B", "Pop", "Pop", "Pop"],
    "energy": [0.8, 0.3, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7, 0.4, 0.8],
    "valence": [0.9, 0.2, 0.8, 0.3, 0.9, 0.7, 0.7, 0.8, 0.6, 0.9],
}

df = pd.DataFrame(data)

#mood souce = enery + valence 조합이라면
#energy = 곡의 강도, valence = 긍정/부정 감정
df["mood_score"] = (df["energy"] + df["valence"]) / 2
    #df에 새로운 열(column)을 추가하는것 -> 오른쪽 값을 행한걸 "mood_score"라는 새로운 열로 추가
    #df["mood_score"]라는 값이 이미 있으면 덮어씀, 없다면 덮어씀


print("===Top 3 Happiest Songs===")
print(df.nlargest(3, "mood_score")[["title", "artist", "mood_score"]])

print("\n===Top 3 Saddest Songs===")
print(df.nsmallest(3, "mood_score")[["title", "artist", "mood_score"]])

#간단 추천 시스템

def recommend_similar(df, title, n=3):  
        #아 이거 class 아니고 함수. class 는 여러 함수(메서드)를 묶어서 관리한다 -> 상태를 기억 (걍 함수는 일회성)
        #여기에서의 df 는 함수에 "전달된 데이터" -> 그래서 밑에서 그걸 복사해서 씀
    target = df[df["title"] == title].iloc[0]   
        #iloc[0]:첫번째 행, iloc[3:5]:3~4번째 행 (끝 숫자 포함 안됨)
        #df["title"] == title : title 이 "Levitating"인 행들만 True 가 됨 -> 그 행들중 첫번째 하나만 가지고 오는것 (iloc[0]): 결과가 비어있다면 에러터짐
        #현실에서는 같은 곡이 여러버전으로 Levitating(Remix) .. (live) 등등으로 존재하기 때문에 여러개중 하나만 쓰겠다는 안전장치 
    """
        filtered = df[df["title"] == title]

        if len(filtered) == 0:
            return "곡을 찾을 수 없음"

        target = filtered.iloc[0]
    """
    df = df.copy() #원본의 df 를 그대로 복사해서 새로운 df 를 만든다 -> 이건 이 함수 안에서만 쓰는것 (원본의 df의 내용을 막 바꿔놓으면 안되니까) -> 밖에서는 영향없음
    df["score"] = np.sqrt(
        (df["bpm"] - target["bpm"]) ** 2 +  # 제곱 
        (df["energy"] - target["energy"]) ** 2 * 100  # 제곱한다음에 100배 함 because, bpm 이 더 값이 엄청 크니까 영향력이 훨씬 커져버림 -> 에너지에 가중치줘서 비중 맞춤
    )
    result = df[df["title"] != title].nsmallest(n, "score") #!= : 같지 않으면, 타이틀로 입력한 기준곡은 제외해야함 . 아 위에서 n 정해놨음 (3개로)
    return result[["title", "artist", "bpm"]]

print("\n===Songs similar to Levitating===")
print(recommend_similar(df, "Levitating"))


"""
about. . . =>  np.sqrt(x^2, y^2)

df["bpm"]은 숫자 하나가 아니고 [171, 67, 114, 84, ...] 전체 열.
np.sqrt() : 모든 값에 대해 루트를 씌움
=> df["score"] 은 bpm 값 에 제곱한것 (A) 과 에너지 값에 제곱하고 백 곱한것(B) 를 더해서 루트를 씌움 -> 각 리스트를 한번에 계산할수 있다
=> 각 좌표의 두 점 사이의 거리로 보여 하니 : 피타고라스 사용해서 구한 것

❗️ cosine similarity

"""