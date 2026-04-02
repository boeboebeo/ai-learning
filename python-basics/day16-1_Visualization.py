# Day 16 - Visualization
# 데이터를 그래프로 표현하기
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "title": ["Blinding Lights", "Someone Like You", "Dynamite",
              "Stay With Me", "Levitating", "Golden Hour",
              "Peaches", "Watermelon Sugar", "Bad Guy", "Shape of You"],
    "bpm": [171, 67, 114, 84, 103, 97, 110, 95, 135, 96],
    "genre": ["Pop", "Ballad", "Pop", "Ballad", "Pop",
              "Pop", "R&B", "Pop", "Pop", "Pop"],
    "energy": [0.8, 0.3, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7, 0.4, 0.8],
}
df = pd.DataFrame(data)

fig, axes = plt.subplots(2, 2, figsize=(10, 8)) #2행 2열까지 그래프 공간 생성 axes[0,0] ~ [1,1]
fig.suptitle("Music Data Analysis ._.", fontsize = 16)  #전체 제목
    #fig => figure. canvas
    #axes = 그 안에 있는 각각의 그래프 공간
    #figsize = (12, 8) -> 숫자 클수록 그래프 커짐 (가로, 세로)


#1. BPM 분포
axes[0, 0].hist(df["bpm"], bins=5, color="skyblue", edgecolor="black")
    #histogram : 데이터 분포를 막대그래프로 보여줌. df["bpm"]값들이 어떤 구간에 얼마나 있는지 보여줌
    #bins = 구간 개수 . 5개의 구간으로 나눠서 빈도 계산 -> 최소, 최대값 사이를 균일하게 나눔
axes[0, 0].set_title("BPM Distribution")    #이 칸의 그래프 제목
axes[0, 0].set_xlabel("BPM")
axes[0, 0].set_ylabel("Count")

#2. 장르별 곡 수
genre_counts = df["genre"].value_counts()   #df["genre"]: 장르를 선택해서, .value_counts() : 각 장르가 데이터 프레임안에서 몇번 나오는지 계산
axes[0, 1].bar(genre_counts.index, genre_counts.values, color="salmon", edgecolor="black")
    #.bar(x축될 값, y축될 값, color = , edgecolor =)
    #.bar( . . ) -> 가로 막대 그래프 만들어짐
axes[0, 1].set_title("Songy by Genre")
axes[0, 1].set_xlabel("Genre")
axes[0, 1].set_ylabel("Count")
print(genre_counts) #결과는 Series 형태. index = 장르이름 / values = 해당 장르 곡수를 나타냄

    #연속형 데이터 → hist
    #범주형 데이터 → bar

#3. BPM vs Energy 산점도
axes[1, 0].scatter(df["bpm"], df["energy"], color="green", alpha=0.7, s=100)
    #.scatter (x, y, .. ) : 산점도(scatter plot) 그리는 함수
    #alpha = 점 투명도 (0 : 완전 투명, 1 : 불투명)
    #s = 100 점 크기
axes[1, 0].set_title("Average BPM by genre")
axes[1, 0].set_xlabel("Genre")
axes[1, 0].set_ylabel("Average BPM")

    #hist : 한 변수 분포
    #scatter : 두 변수 관계/분포 

#4. 장르별 평균 BPM
genre_bpm = df.groupby("genre")["bpm"].mean()
print(genre_bpm.round(2))
axes[1, 1].bar(genre_bpm.index, genre_bpm.values, color="purple", alpha=0.7)
axes[1, 1].set_title("Average BPM by Genre")
axes[1, 1].set_xlabel("Genre")
axes[1, 1].set_ylabel("Average BPM")

plt.tight_layout()  #자동으로 그래프 요소가 겹치지 않게 여백 위치 조정해줌
plt.savefig("music_analysis.png")   #위 fig 를 저장하는것
plt.show()
print("Chart image saved!")