#day 15 pandas advanced

import pandas as pd

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
    "year": [2019, 2011, 2021, 2014, 2020, 2022, 2021, 2019, 2019, 2017],
}

df = pd.DataFrame(data)     #DataFrame 이라는 class(설계도)로 객체생성
                            #df.head(), df.groupby(), df.nlargest() 다 dataFrame 객체가 가지고 있는 기능들
    #내가 넣은 딕셔너리 형태의 데이터를 row & col 이 존재하는 표(table) 형태로 변환 
    #행렬(maxtrix)와는 살짝 다름 : 숫자만 있는 느낌 / DataFrame : 각 열에 이름 있고, 엑셀처럼 다룸. 숫자+문자열+날짜 등의 다양한 데이터 타입 가능


def classify_tempo(bpm):
    if bpm < 80:
        return "Slow"
    elif bpm < 120:
        return "Mid-tempo"
    else:
        return "Fast"
    
df["tempo_category"] = df["bpm"].apply(classify_tempo)
    #bpm 값 하나하나에 함수(classify_tempo)를 적용해서 새로운 열 tempo_category를 만듦
    #0 Fast , 1 Slow, 2 Mid-tempo, 3 Mid-tempo 등의 새로운 하나를 만듦! 출력해보면 됨 
    #print(df["tempo_category"])
    #.apply : 각 값에다가 하나씩 다 클래스 적용 => classify_tempo(171)..
    #그리고 df["tempo_category"] : 로 새로운 열로 저장 !
    #.apply 는 for 문을 pandas 스럽게 쓴것 


    #카테고리별 요약 통계만든는 것 : groupby + agg
print("===Stats by Genre===")
print(df.groupby("genre").agg({ #groupby : 장르별로 묶어서, agg() : aggregation(집계) . 각 그룹에 대해 어떤 계산을 할것인지 정의하는 부분 
    "bpm" : ["mean", "min", "max"], #문자열이여도 미리 정의된 함수들 .
    "energy" : "mean"
}).round(2))

    #"bpm" 은 평균, 최소, 최대
    #"energy" 는 평균 => 을 각각 구함 => column 이 2단 구조가 됨 (bpm 밑에 mean/min/max)
    #round(2) 는 소숫점 이하 두자리 까지 반올림 

print("\n===Top 3 Highest BPM===")
print(df.nlargest(3, "bpm")[["title", "artist", "bpm"]])
    #BPM 이 가장 높은 곡 3곡을 골라서, 제목/아티스트/BPM 만 보여줘!

print("\n===Tempo Distribution===")
print(df["tempo_category"].value_counts(normalize=True))  # 각 값이 몇번나왔는지 카운트 : .value_counts()
    #normalize=True 가 () 안 에 들어가면 퍼센테이지로 나옴 . ex. 0.7, 0.2, 0.1

df.to_csv("music_dataset.csv", index=False) 
print("\n✅ Saved to music_dataset.csv")
    #df.to_csv() : dataFrame 이 가지는 기능 -> DataFrame을 엑셀용 파일로 내보내기
    #"music_dataset.csv" : 파일 이름
    #.csv : 엑셀에서 열수있는 파일
    #index=False : 인덱스 때문에 열이 쓸데없이 하나 더 생겨서 그거 빼고 저장

