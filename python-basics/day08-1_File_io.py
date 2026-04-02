#day08_file_I/O saving and loading data


#json : 모든 언어가 읽을 수 있는 공통 형식. 데이터를 저장하는 방식
    #데이터가 저장되어 있지않으면 e.g. 인스타 팔로워 목록이 껐다 켜도 그대로 있는것 , 서버 -> 앱 데이터 전달 및 저장이 안됨
    #just, data 형식 => 텍스트 파일! (메모장으로도 열림)
import json

playlist = [
    {"title": "Blinding Lights", "artist": "The Weeknd",
     "bpm": 171, "genre": "Pop", "key": "F minor"},
    {"title": "Someone Like You", "artist": "Adele",
     "bpm": 67, "genre": "Ballad", "key": "A major"},
    {"title": "Dynamite", "artist": "BTS",
     "bpm": 114, "genre": "Pop", "key": "B major"},
]


# Save -> 파이썬 리스트를 playlist.json 파일로 저장
with open("playlist.json", "w", encoding="utf-8") as f:
    json.dump(playlist, f, indent=2, ensure_ascii=False)
print("✅ Saved to playlist.json")

#with : 파일을 열고닫는것을 자동으로 해줌. -> with 를 쓰면 블록 끝났을때 자동으로 닫힘
    #"playlist.json" : 파일이름 / 경로 (열거나 만들 파일)
    #w=쓰기, r=읽기모드 !
    #encoding="utf-8" : 한글 깨짐 방지
    #as f : 이 파일을 f 라고 부를게!

#json.dump : 파이썬 리스트를 jSON 파이로 변환해서 저장
    #(저장할 데이터(playlist), 어디에(f), 들여쓰기 2칸(indent=2), 한글깨짐 방지(=ensure_ascii=False))



# Load -> playlist.json 파일을 파이썬으로 불러오기
with open("playlist.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
print(f"✅ Loaded {len(loaded)} songs") #loaded 안에 몇개가 있는지 : len

for song in loaded:
    print(f"- {song['title']} ({song['bpm']} BPM)")

#"playlist.json" : 읽을 파일 이름
    #r = 읽기모드
    #loded = json.load(f): 파일(f)을 파이썬으로 변환해서 loaded 에 담기

    #json.load : 파일 -> 파이썬 (불러오기)
    #json.dump : 파이썬 -> 파일 (저장)

    #for song in loaded : 이제 loaded 가  파이썬 리스트니까 그냥 for 문 돌리면 됨

# Save as text -> 불러온 데이터를 **텍스트 파일로 저장**
with open("playlist_summary.txt", "w") as f:
    for song in loaded:
        f.write(f"{song['title']} - {song['artist']}\n")
print("✅ Summary saved!")

#여기는 json 이 아니라 txt 그냥 텍스트 파일
    #for song in loaded : 아까 불러온 loaded 리스트를 하나씩 꺼내서
    #f. write(..) : 파일에 한 줄씩 써줘!
    #json.dump 와의 차이점 -> jSON 형식으로 저장 (데이터 형식. 다시 불러와서 쓸 수 있음)
        #f.write -> 그냥 텍스트로 저장 (메모장으로 읽는 용도)
        

#실제로 사용되는 큰 서비스는 JSON 파일 말고 -> DB(DataBase)를 씀
    #근데 데이터베이스도 데이터를 주고받을 때 JSON 형식을 씀 (저장방식이기도 하고, 주고받는 형식이기도 함)