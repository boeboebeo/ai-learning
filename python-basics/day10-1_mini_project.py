#day 10 - mini project 1 : music analyzer

import json

#곡 하나의 설계도 . 곡 하나의 데이터를 담음
class Song:
    def __init__(self, title, artist, bpm, genre, key=None, energy=None):
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.genre = genre 
        self.key = key
        self.energy = energy

    def get_tempo(self):
        if self.bpm < 80:
            return "slow"
        elif self.bpm < 120:
            return "mid-tempo"
        else:
            return "fast"
        
    def to_dict(self):
        return {
            "title" : self.title,
            "artist" : self.artist,
            "bpm" : self.bpm,
            "genre" : self.genre,
            "key" : self.key,
            "energy" : self.energy,
        }
    
#분석기 설계도 . 곡들을 모아서 분석, 여기에 song 들이 담겨있음
class MusicAnalyzer:
    def __init__(self):
        self.songs = [] #[song1, song2, song3 .. ] 가 여러개 담겨있음

    def add_song(self, song):
        self.songs.append(song)
    
    def get_stats(self):
        bpms = [s.bpm for s in self.songs] #Song 객체의 bpm 을 꺼내오는 것 : s.bpm
        return {
            "total_songs" : len(self.songs),
            "average_bpm" : round(sum(bpms) / len(bpms), 1), #소수점 한자리로 반올림
            "max_bpm" : max(bpms),
            "min_bpm" : min(bpms),
        }
    

    
    def save(self, filename): #여기서 to_dict()가 쓰임
        data = [s.to_dict() for s in self.songs]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False) #Song 객체를 딕셔너리로 변환해서 저장
        print(f"✅ Saved to {filename}")

    def filter_by_genre(self, genre):
        return [s for s in self.songs if s.genre == genre] 
            #Song 객체의 genre 를 꺼내오는 것 : s.genre
            #if s.genre = genre : 내가 찾는 그 장르가 맞을때만 리턴 (== genre : 에 각 장르가 들어가는 것)

    def show_report(self):
        print("\n" + "="*40) # ==== 이런 표기 만드는 것
        print("       🎵 MUSIC ANALYZER REPORT")
        print("="*40)
        for key, value in self.get_stats().items(): #get_stats()가 리턴하는 딕셔너리를 .item()으로 key, value 로 뽑아서 출력
            print(f"    {key}:{value}")
        print("\n Genre breakdown:")
        genres = set(s.genre for s in self.songs) #set : 중복을 제거해주는 함수 => pop, ballad 두 장르만 남았고,
        for genre in genres: 
            count = len(self.filter_by_genre(genre))  #해당 장르가 들어오면 그 장르의 개수를 셈
            print(f"    - {genre}: {count} songs")
        print("="*40)


#분석기 생성
analyzer = MusicAnalyzer()  
#클래스는 설계도라서, 직접 쓸 수 없음! 이렇게 객체를 만들어서 찍어내야 함
# ❌ 직접 MusicAnalyzer.add_song(song) -> 이렇게는 안됨
#()를 붙이는 이유는 ! 실행하라는 뜻 -> 괄호가 없으면 실행이 안됨
    #괄호 칠때 __init__이 자동으로 실행됨

#곡들 생성
songs_data = [
    Song("Blinding Lights", "The Weeknd", 171, "Pop", "F minor", 0.8),
    Song("Someone Like You", "Adele", 67, "Ballad", "A major", 0.3),
    Song("Dynamite", "BTS", 114, "Pop", "B major", 0.7),
    Song("Stay With Me", "Sam Smith", 84, "Ballad", "C major", 0.4),
    Song("Levitating", "Dua Lipa", 103, "Pop", "B major", 0.8),
]

#분석기 안에 곡 넣기
for song in songs_data:
    analyzer.add_song(song)

analyzer.show_report()
analyzer.save("analyzed_playlist.json")  #파일로 저장

#파이썬에서는 (나는 이미 폴더 안에 있으므로) 파일만 만들 수 있음
#goeun@songo-eun-ui-MacBookAir python-basics % 나는 이미 python-basics 폴더 안에 있음

