#day 19 - Final Project: Features 

import numpy as np
import pandas as pd

class MusicIntelligneceSystem:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.df["tempo_category"] = self.df["bpm"].apply(self._classify_tempo)
        self.df["mood_score"] = (self.df["energy"] + self.df["valence"]) / 2

    def _classify_tempo(self, bpm):
        if bpm < 80: return "Slow"
        elif bpm <100: return "Mid-tempo"
        else: return "Fast"

    def recommend(self, title, n=3):
        """
        Simple recommendation based on audio features.
        """
        target = self.df[self.df["title"] == title].iloc[0]
        df = self.df.copy()

        #거리 계산 (BPM, energy, valence 기반)
        df["distance"] = np.sqrt(
            ((df["bpm"] - target["bpm"]) / 100) ** 2 +
            (df["energy"] - target["energy"]) ** 2 + 
            (df["valence"] - target["valence"]) ** 2
        )   
            #여기서 이렇게 제곱을 해서 구한다음에 루트를 씌우는건 나오는 값이 마이너스 일수 있기 때문도 있지만
            #x, y, z 공간에서의 거리 구하는 공식이 : 루트(x차이^2 +y차이^2 + z차이^2) 이기 때문

        result = df[df["title"] != title].nsmallest(n, "distance")
        return result[["title", "artist", "bpm", "mood_score", "distance"]] #여기서 디스턴스 소수점 2자리까지만 보려면
    
    def get_mood_playlist(self, mood="happy", n=3): #mood 를 안넣으면 자동으로 "happy"를 사용하라고 기본값으로 적용해놓은 것 
        """Create playlist based on mood."""
        df = self.df.copy()
        if mood == "happy":
            return df.nlargest(n, "mood_score")[["title", "artist"]]
        elif mood == "sad":
            return df.nsmallest(n, "mood_score")[["title", "artist"]]
        else:
            return df.sample(n)[["title", "artist"]]
        

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

system = MusicIntelligneceSystem(data)

print("===Recommendations for Levitating===")
print(system.recommend("Levitating"))

print("\n===Happy Playlist===")
print(system.get_mood_playlist("happy"))

print("\n===Sad Playlist===")
print(system.get_mood_playlist("sad"))