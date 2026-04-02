#day 14 _ pandas basic
    # pandas : 데이터를 "표(테이블)" 형태로 다루기 위해 만들어진 라이브러리
    # 행(row) , 열(col) 구조를 다룸

#install : pip3 install pandas
#version check : print(pd.__version__)

import pandas as pd

#python dictionary 형태 (key : 열(colums) , value : 각 열에 들어갈 list)
data = {
    "title": ["Blinding Lights", "Someone Like You", "Dynamite",
              "Stay With Me", "Levitating", "Golden Hour",
              "Peaches", "Watermelon Sugar"],
    "artist": ["The Weeknd", "Adele", "BTS", "Sam Smith",
               "Dua Lipa", "JVKE", "Justin Bieber", "Harry Styles"],
    "bpm": [171, 67, 114, 84, 103, 97, 110, 95],
    "genre": ["Pop", "Ballad", "Pop", "Ballad",
              "Pop", "Pop", "R&B", "Pop"],
    "energy": [0.8, 0.3, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7],
    "year": [2019, 2011, 2021, 2014, 2020, 2022, 2021, 2019],
}

#df = pd.DataFrame(data, index=["1", "2", "3", "4", "5", "6", "7", "8"])  #이렇게 인덱스 1부터 바꾸는 건 수천수백개가 된다면 너무 비효율적
df = pd.DataFrame(data)
    #DataFrame : 데이터를 행/열 구조로 변환
df.index = range(1, len(df)+1) 
    # range (start num, stop number) 인데 마지막은 포함이 되지 않으므로 마지막 숫자까지 포함하려면 len(df)+1 해줘야 함

print(df)

print("\n===Basic Info===")
print(df.shape) #DataFrame 의 구조 (행/ 열의 개수)를 알려주는 속성 -> (행 개수, 열 개수) 이렇게 출력됨
print(df.dtypes)  #각 col (열)의 데이터 타입 확인 int64(정수형), float64(소수점 있는 숫자)

print("\n===First 3 rows===")
print(df.head(3))   #.head(3) : 앞 3행 가져오기 / .tail(n) : 뒤 n행 가져오기 
                    # df[2.5] : 2~4번째 행 가져오기(마지막 행 포함 안됨)/ df.loc[2:4] 라면 인덱스 이름을 기준으로 선택가능 : 인덱스 이름 2~4행 
print("\n", df.tail(3))

print("\n", df.loc[4:5])

print("\n===BPM Statistics===")
print(df["bpm"].describe()) 

#describe : 기본적으로 count(데이터 개수), mean, std(표준편차), min, 25%(1사분위 수. 데이터를 4등분 했을때 25%지점), 50%, 75%, max

print("\n===Average BPM by Genre===")
print(df.groupby("genre")["bpm"].mean().round(1))
    #.groupby("해당 열을 기준으로 같은 장르끼리 그룹화")["그룹화 후 bpm 열만 선택"].
    # .mean() ; 각 그룹끼리의 bpm 평균 계산 / .round(1) : 소수점 1의 자리까지 반올림 

print("\n===Pop songs===")
pop = df[df["genre"] == "Pop"] 
    #[df["genre"] == "Pop"] : genre 열에서 "Pop"인 행마다 True/False 표시 -> 이렇게 Boolean mask 를 만들어서 조건에 맞는 행만 뽑기  
    #df[df["genre"] == "Pop"] : 위에서 True 인 행만 선택 (Pop 장르의 모든 행만 남게됨 )
print(pop[["title", "artist", "bpm"]]) #그 중에서 title, artist, bpm 열만 보여줌 