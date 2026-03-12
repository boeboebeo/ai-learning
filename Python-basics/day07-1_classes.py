# Day 07 - Classes

# class : 뭔가를 만들어 낼 수 있는 틀
# __init__ : 처음 만들때 무얼 넣을지 정의. title, artist, bpm, genre 는 무조건 있어야 함
    #self.title = title : song 1 이 부르면 song1자신의 타이틀에 넣어줘!

class Song:
    def __init__(self, title, artist, bpm, genre, key=None):
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self. genre = genre
        self.key = key

        #song1.get_tempo() : song1의 bpm 으로 템포 계산
        #mothod : 클래스 안의 함수 ex. def get_tempo(self):

    def get_tempo(self):
        if self.bpm < 80:
            return "Slow"
        elif self.bpm < 120:
            return "Mid-tempo"
        else: return "Fast"

    def describe(self):
        print(f"{self.title} by {self.artist}")
        print(f"    BPM : {self.bpm} ({self.genre})")
        if self.key:
            print(f"    Key : {self.key}")

class Playlist:
    def __init__(self, name):
        self.name = name
        self.songs = []

    def add_song(self, song):
        self.songs.append(song)

    def get_average_bpm(self):
        if not self.songs:
            return 0
        return sum(s.bpm for s in self.songs) / len(self.songs)
    
    def show(self):
        print(f"\nPlaylist: {self.name}")
        print(f"Totle : {len(self.songs)} songs")
        print(f"Average BPM : {self.get_average_bpm():.1f}")
        for i, song in enumerate(self.songs, 1):
            print(f"{i}. {song.title} - {song.artist}")



song1 = Song("Blinding Lights", "The Weeknd", 171, "Pop", "F minor")
song2 = Song("Someone Like You", "Adele", 67, "Ballad", "A major")
song3 = Song("Dynamite", "BTS", 114, "Pop", "B major")

my_playlist = Playlist("My Favorites")
my_playlist.add_song(song1)
my_playlist.add_song(song2)
my_playlist.add_song(song3)
my_playlist.show()


"""
**전체 구조:**

class Song        → 설계도
__init__          → 재료 받는 곳
self.title 등     → 각자의 데이터 저장
get_tempo, describe → 각자의 기능

"""

