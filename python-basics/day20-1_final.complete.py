#day 20 - final project complete! 

import numpy as np
import pandas as pd
import json

class MusicIntelligenceSystem:
    def __init__(self):
        self.df = None
    
    def load_data(self, data):
        self.df = pd.DataFrame(data)
        self.df["tempo_category"] = self.df["bpm"].apply(self._classify_tempo)
        self.df["mood_score"] = (
            self.df["energy"] + self.df["valence"]
        ) / 2
        print(f"✅ Loaded {len(self.df)} songs")

    def _classify_tempo(self, bpm):
        if bpm < 80: return "Slow"
        elif bpm < 120: return "Mid-tempo"
        else: return "Fast"

    def get_stats(self):
        bpms = np.array(self.df["bpm"])
        return {
            "total_songs" : len(self.df),
            "avg_bpm" : round(np.mean(bpms), 1),
            "std_bpm" : round(np.std(bpms), 1),
            "max_bpm" : int(np.max(bpms)),  #값을 정수형으로 바꿔줌 . 96.0 -> 96
            "min_bpm" : int(np.min(bpms)),
        }
    
    def recommend(self, title, n=3):
        target = self.df[self.df["title"] ==title].iloc[0]
        df = self.df.copy()
        df["distance"] = np.sqrt(
            ((df["bpm"] - target["bpm"]) / 100) ** 2 +
            (df["energy"] - target["energy"]) ** 2 +
            (df["valence"] - target["valence"]) ** 2
        ).round(4)  #그냥 column 자체를 네자리수로 반올림 해버리기
        return df[df["title"] != title].nsmallest(n, "distance")[
            ["title", "artist", "bpm", "distance"]
        ]
    
    def get_mood_playlist(self, mood="happy", n=3):
        if mood == "happy":
            return self.df.nlargest(n, "mood_score")[["title", "artist"]]
        elif mood == "sad":
            return self.df.nsmallest(n, "mood_score")[["title", "artist"]]
        elif mood == "energetic":
            return self.df.nlargest(n, "energy")[["title", "artist"]]
    
    def analyze_by_genre(self):
        return self.df.groupby("genre").agg({   #장르별로 묶어서 -> 각 컬럼에 대해서 어떤 계산을 할지 정하는게 agg({})!
            "bpm" : ["mean", "count"],  #문자열이여도 pandas가 내부에서 해석함
            "energy" : "mean",
            "mood_score" : "mean",
        }).round(2)
    
    def show_full_report(self):
        print("\n" + "="*50)
        print(" 🕹️ MUSIC INTELLIGENCE SYSTEM REPORT")
        print("="*50)

        print("\n📊 Basic Stats:")
        for k, v in self.get_stats().items():
            print(f"    {k}:{v}")
        
        print("\n🎸 Analysis By genre:")
        print(self.analyze_by_genre())

        print("\n ⏲️ Tempo Distribution:")
        for tempo, count in self.df["tempo_category"].value_counts().items():
            bar = "🟩" * count
            print(f"    {tempo:12} {bar} ({count})")    #tempo 를 줄 맞춰서 네모칸 동일한 타이밍에서 시작하게 하려고 :12 표기함

            #value_counts() : 값이 몇번씩 등장했는지 세는 함수 -> 딕셔너리 처럼 생긴 pandas series 를 출력함
            #slow : 4, mid-tempo : 3, fate : 3 이렇게 출력 ! (딕셔너리는 아니고) Pandas.Series 형태 => 그래서 item() 가능

        print("\n 😀 Mood Analysis:")
        print(" Top 3 Happiest:")
        happy = self.df.nlargest(3, "mood_score")[["title", "mood_score"]]
        for _, row in happy.iterrows(): # _ : 은 이 값은 안쓸거니까 무시한다는 뜻 iterrows() 가 (index, row) 이렇게 두 값 빼오는데 index 필요없음
            print(f"    - {row['title']} ({row['mood_score']:.2f})")

        print("="*50)

    def save_report(self, filename):
        self.df.to_csv(filename, index=False)
        print(f"\n✅ Report Saved to {filename}")

#실행

system = MusicIntelligenceSystem()

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

system.load_data(data)
system.show_full_report()

print("\n🔍 Recommendations for 'Levitating':")
print(system.recommend("Levitating"))

print("\n🎧 Energetic Playlist:")
print(system.get_mood_playlist("energetic"))

system.save_report("music_intelligence_report.csv")

print("\n" + "="*50)
print(" 💐 20-day Python Journey Complete!")
print("="*50)