# Day 07 - Classes

# class : 뭔가를 만들어 낼 수 있는 틀
# __init__ : 처음 만들때 무얼 넣을지 정의. title, artist, bpm, genre 는 무조건 있어야 함
    #self.title = title : song 1 이 부르면 song1자신의 타이틀에 넣어줘!

class Song:  #Song 만들때 필요한 재료들 받음
    def __init__(self, title, artist, bpm, genre, key=None):
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self. genre = genre
        self.key = key

        #song1.get_tempo() : song1의 bpm 으로 템포 계산
        #mothod : 클래스 안의 함수 ex. def get_tempo(self):

    def get_tempo(self):
        if self.bpm < 80: #self.bpm : 자기자신의 bpm 으로 각각 계산함
            return "Slow"
        elif self.bpm < 120:
            return "Mid-tempo"
        else: return "Fast"

    def describe(self): 
        print(f"\n{self.title} by {self.artist}")
        print(f"    BPM : {self.bpm} ({self.genre}) ({self.get_tempo()})")

        if self.key: #self.key 가 있을때만 출력. None 이면 출력 안함
            print(f"    Key : {self.key}")  #밑에서 따로 Print로 함수 호출 시 -> None 출력 (Return 이 없기때문에)

class Playlist:
    def __init__(self, name):   #Playlist 만들때 name만 받음, songs 는 빈 리스트로 시작
        self.name = name
        self.songs = []

    def add_song(self, song): #song 은 그냥 매개변수. - 무언갈 받게됨
        self.songs.append(song)

    def get_average_bpm(self):
        if not self.songs: #songs 가 비어있으면 -> 0리턴
            return 0
        return sum(s.bpm for s in self.songs) / len(self.songs) #list comprehension
    
    def show(self):
        print(f"\nPlaylist: {self.name}")
        print(f"Totle : {len(self.songs)} songs")
        print(f"Average BPM : {self.get_average_bpm():.1f}")  #method 안에서 다른 method 호출
        for i, song in enumerate(self.songs, 1): 
            print(f"{i}. {song.title} - {song.artist}")



song1 = Song("Blinding Lights", "The Weeknd", 171, "Pop", "F minor")
song2 = Song("Someone Like You", "Adele", 67, "Ballad", "A major")
song3 = Song("Dynamite", "BTS", 114, "Pop", "B major")

my_playlist = Playlist("My Favorites")
my_playlist.add_song(song1) #song 1이 song 그릇에 들어감
my_playlist.add_song(song2) #song1, 2, 3 가 실제 객체
my_playlist.add_song(song3)
my_playlist.show()
song2.describe()
#print(f"{song1.describe()}) -> 이거 하면 None 출력됨 (Return 이 없는 method 이기 때문에)


"""
**전체 구조:**

class Song        → 설계도
__init__          → 재료 받는 곳
self.title 등     → 각자의 데이터 저장
get_tempo, describe → 각자의 기능

"""

