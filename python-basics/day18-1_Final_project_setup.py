# day 18 - Final project setup

import json
import pandas as pd
import numpy as np

#def __init__(self) : 는 객체를 만들때 자동으로 실행되는 함수임
    #system = Music....() : 이러 호출하자마자 __init__은 실행된다고 보면됨
    #system1 = MusicIntelligenceSystem() : system1.df = "A 데이터"
    #system2 = MusicIntelligenceSystem() : system2.df = "B 데이터"
        #=> 이렇게 객체마다 다른 데이터를 가지게 하려고 분리해서 쓰는것 (해당 객체의 전용 저장공간 접근 키 : self)


class MusicIntelligenceSystem:
    def __init__(self): #self : 현재의 이 system 이라는 객체를 가리키는 변수.
        self.df = None  #self.df = 이 system 이 라지고 있는 df . 아직 자리는 없지만 자리만 만들어둠!
                        #아직 아무값도 안 들어왔다. 라는 상태를 명확하게 표현하기 위해서 None 으로 초기값 셋팅
                        #None : 아직 시작도 안했다는 걸 의미하기 위해서 ([]뭐 이런 시작했는데 비어있는 요런 표기말고)
        print("🕹️ Music Intelligence System initialized!")

    def load_data(self, data):
        self.df = pd.DataFrame(data)  #self.df 라는 곳에 저장. data에서 변환해서 table 생성한것을 
        self.df["tempo_category"] = self.df["bpm"].apply(self._classify_tempo) 
            #_classify_tempo 메서드 이용해서 bpm값을 하나씩 꺼내서 새로운 column 생성
            #.apply() : 각 행마다 함수 실행
        self.df["mood_score"] = (   #energy와 valence 값 더해서 나눈값으로 mood_score 라는 새로운 column생성
            self.df["energy"] + self.df["valence"]
        ) / 2
        print(f"✅ Loaded {len(self.df)} songs")    
            #df 의 길이만큼 불러옴! -> 근데 여기서의 길이는 전체 행의 개수 = 인덱스 개수
            #title, artist 같은건 컬럼 이름(열 이름) 임. 인덱스에 포함안됨(데이터 줄만 포함됨)

    def _classify_tempo(self, bpm):
        if bpm < 80:
            return "Slow"
        elif bpm < 120:
            return "Mid-tempo"
        else:
            return "Fast"
        
    def get_stats(self):
        bpms = np.array(self.df["bpm"]) #np : 숫자계산 엔진 / pd : 데이터 관리도구(table)
                                        #여기선 : "bpm"값 다 가지고 와서 numpy 배열로 변환 => 계산을 numpy 로 하려고
                                        #self.df["bpm"].mean() -> pandas 로도 내부적으로 numpy 쓸수도 있음
 
        return {
            "total" : len(self.df),
            "avg_bpm" : round(np.mean(bpms), 1),
            "std_bpm" : round(np.std(bpms), 1), #standard deviation. SD
        }
    
    def save(self, filename):
        self.df.to_csv(filename, index=False)   #이건 index 값은 같이 저장안하겠다는것
        print(f"✅Saved to {filename}")



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

system = MusicIntelligenceSystem() #system 이란 변수에서 클래스 사용할것임 
print("Setup complete! Ready for Day 19 ! ")