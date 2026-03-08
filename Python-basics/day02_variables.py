#Variables and Data types

song_title = "Blinding Lights"  #str(문자열)
artist = "The Weekend"  #str
genre = "Pop"  #str
bpm = 171  #int(정수형)
duration = 3.22 #float(소수형)
energy = 0.8    #float
is_favorite = True  #bool (불리언)

#method 1
print("===song Info===")
print(f"Title: {song_title}")
print(f"Artist : {artist}")
print(f"Genre : {genre}")
print(f"BPM : {bpm}")
print(f"Duration : {duration} min")
print(f"Energy : {energy}")
print(f"Favorite : {is_favorite}")

#method 2 
print(f"""
===song Info===
Title : {song_title}
Artist : {artist}
Genre : {genre}
BPM : {bpm}
Duration : {duration}min
Energy : {energy}
Favorite : {is_favorite}
""")
