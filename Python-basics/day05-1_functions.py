#reusable functions

def classify_tempo(bpm):
    if bpm < 80:
        return "Slow"
    elif bpm < 120:
        return "Mid-tempo"
    else:
        return "Fast"
    
def classify_energy(energy):
    if energy < 0.3:
        return "Low energy"
    elif energy < 0.6:
        return "Mid energy"
    else :
        return "High envery"
    
def get_song_summary(title, artist, bpm, energy):
    return f"{title} by {artist} -> {classify_tempo(bpm)} / {classify_energy(energy)}"

#list comprehension 
    #bpms = []
    #for song in playlist:
        #bpms.addend(song["bpm"]) 과 같음 append : 하나씩 리스트에 추가하는 함수

def get_playlist_stats(playlist):
    bpms = [song["bpm"] for song in playlist]
    return {
        "total_songs" : len(playlist),
        "average_bpm" : round(sum(bpms) / len(bpms), 1),
        "max_bpm" : max(bpms),
        "min_bpm" : min(bpms),
    }

print("=== song summaries ===")
print(get_song_summary("Blinding Lights", "The Weeknd", 171, 0.8))
print(get_song_summary("Someone Like You", "Adele", 67, 0.3))

playlist = [
    {"title": "Blinding Lights", "bpm": 171},
    {"title": "Someone Like You", "bpm": 67},
    {"title": "Dynamite", "bpm": 114},
]

print("\n=== Playlist Stats ===")
stats = get_playlist_stats(playlist)
for key, value in stats.items():
    print(f"{key}:{value}")

#for i in playlist
    #딕셔너리를 for문에 넣으면 파이썬은 key만 순서대로 나옴 : value 까지 필요하면 .item() 붙여야 함

#upper() : 점 앞을 대문자로!
song = "mudmud"
print(f"\n{song.upper()}")

#bpms.append : bpms 이란 리스트에 하나하나 추가함 
    #append 는 추가만 하는 함수라 print 랑 같이 쓰면 None 나옴 -> 따로 써줘야 함
bpms = []
for song in playlist:
    bpms.append(song["bpm"])

print(bpms)

