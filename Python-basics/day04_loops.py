#Analyzing a playlist using for loops and while loops


#Dictionary - 'key : value' 형태로 이루어짐 in list

'''
tuple vs dictionary 

tuple - 순서로 접근
    song = ("Blinding Lights", "The Weeknd")
    song[0] #'Blinding Lights'

dictionary - key로 접근 (: -> key, value 연결 기호)
    song = {"title": "Blinding Lights", "artist": "The Weeknd"}
    song["title"] #'Blinding Lights'
'''


playlist = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "bpm": 176.4, "genre": "Pop"},
    {"title": "Someone Like You", "artist": "Adele", "bpm": 60.3, "genre": "Ballad"},
    {"title": "Dynamite", "artist": "BTS", "bpm": 114.4, "genre": "Pop"},
    {"title": "Stay With Me", "artist": "Sam Smith", "bpm": 84.2, "genre": "Ballad"},
]

#For loop - print all songs 
print("=== My Playlist ===")
for i, song in enumerate(playlist, 1): #enumerate (playlist) -> 이면 0. 부터 번호 생김
    print(f"{i}. {song['title']} - {song['artist']} ({song['bpm']} BPM)")

    #for i : 번호 자동으로 카운트 (1, 2, 3..) -> 번호를 만드는건 아니고, 순서대로 돌기만 함
    #song in enumerate : 각 딕셔너리가 1번부터 번호 만들어줌 -> 그걸 받는게 i
        # enumerate가 이런 쌍을 만들어줌 - (1, {"title": "Blinding Lights"...})


#Calculate average BPM
total_bpm = 0
for song in playlist:   #Playlist 내에서 dictionary 쭉 
    total_bpm += song["bpm"]
average_bpm = total_bpm / len(playlist) #len() : playlist 내부 개수(길이) 세주는 함수
print(f"\nAverage BPM: {average_bpm:.5f}")  #자동 반올림 해서 보여줌 
print(f"Average BPM: {round(average_bpm, 2)}")  #이렇게 세자리 수까지만 반올림해서 볼 수 도 있음

#\n : 한줄 띄어쓰기
#:.3f : 소수점 세자리 까지만 표기
#:.0f : 소수점 없이 표기

#Find fastest and slowest song
fastest = max(playlist, key=lambda x: x["bpm"])
slowest = min(playlist, key=lambda x: x["bpm"])
print(f"\nFastest: {fastest['title']} ({fastest['bpm']} BPM)")
print(f"Slowest: {slowest['title']} ({slowest['bpm']} BPM)")


#lambda = 이름 없는 일회용 함수
#key=lambda x: x["bpm"] -> key : bpm만 꺼내서 비교하라고 알려줌 (x는 Varaible)


#While loop - count pop songs


while_count = 0
index = 0

while index < len(playlist):
    if playlist[index]["genre"] == "Pop": #pop -> 소문자면 카운트 안됨
        while_count += 1
    index += 1
print(f"\nNumber of Pop songs : {while_count}")



#For loop
for_count = 0
for song in playlist:
    if song["genre"] == "Pop":
        for_count += 1
print(f"\nNumber of Pop songs : {for_count}")



'''
# 번호 필요없을 때
for song in playlist:

# 번호 필요할 때
for i, song in enumerate(playlist, 1):
'''

'''
while 조건:
    실행

    #조건이 True 인 동안 계속 반복
    while index < len(playlist)

    => index = 5 -> 5 < 5 X : 종료 !
'''
