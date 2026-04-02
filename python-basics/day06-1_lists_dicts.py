#Lists & Dictionaries

my_genres = ["Pop", "R&B", "Ballad", "Dance", "Hip-hop"]
my_genres.append("Jazz")
my_genres.remove("Dance")
print("My favorite genres :", my_genres)

bpm_list = [171, 67, 114, 84, 103]
print("Ascending : ", sorted(bpm_list))
print("Descending : ", sorted(bpm_list, reverse=True))

#sorted(bpm_list, reverse=True)
    #sorted()의 기본값은 오름차순(작은 -> 큰) 이기 때문에 반대로 하려면 reverse=True
    #reverse=False 로 하게 되면 원래 셋팅과 같게 오름차순 !    

song = {
    "title": "Blinding Lights",
    "artist": "The Weeknd",
    "bpm": 171,
    "key": "F minor",
    "time_signature": "4/4"
}

print(f"\nKey : {song['key']}") #
print(f"Time signature : {song['time_signature']}")

my_playlist = {
    "workout" : ["Blinding Lights", "Dynamite", "Levitating"],
    "study": ["Someone Like You", "Stay With Me"],
    "chill": ["Peaches", "Golden Hour"],
}

print("\n=== My Playlist ===")
for name, songs in my_playlist.items():
    print(f"{name} : {len(songs)} songs")

    #my_playlist.items():key, value 모두 빼오는 기능
    #stats.items() : key + value 
    #stats.keys(): key
    #stats.values() : value 만 ! 빼옴