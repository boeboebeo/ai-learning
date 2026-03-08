#Conditionals
#using if/elif/else

#define function
#(bpm) -> parameter
def classify_tempo(bpm):
    if bpm < 60:
        return "Very slow"
    elif bpm < 80:
        return "Slow"
    elif bpm < 110:
        return "Mid-tempo"
    elif bpm < 140:
        return "Fast"
    else:
        return "Very fast"
    
#Test with real songs
songs = [
    ("Someone Like you", "Adele", 67),
    ("Blinding Lights", "The Weeknd", 171),
    ("banana", "mudmud", 138),
    ("sports", "mudmud", 108)

]

print("===Tempo Classification===")
for title, artist, bpm in songs:
    tempo = classify_tempo(bpm)
    print(f"{title} by {artist} ({bpm} BPM) -> {tempo}")
