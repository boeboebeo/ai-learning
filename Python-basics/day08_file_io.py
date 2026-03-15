#day08_file_I/O saving and loading data

import json

playlist = [
    {"title": "Blinding Lights", "artist": "The Weeknd",
     "bpm": 171, "genre": "Pop", "key": "F minor"},
    {"title": "Someone Like You", "artist": "Adele",
     "bpm": 67, "genre": "Ballad", "key": "A major"},
    {"title": "Dynamite", "artist": "BTS",
     "bpm": 114, "genre": "Pop", "key": "B major"},
]

# Save
with open("playlist.json", "w", encoding="utf-8") as f:
    json.dump(playlist, f, indent=2, ensure_ascii=False)
print("✅ Saved to playlist.json")

# Load
with open("playlist.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)
print(f"✅ Loaded {len(loaded)} songs")

for song in loaded:
    print(f"- {song['title']} ({song['bpm']} BPM)")

# Save as text
with open("playlist_summary.txt", "w") as f:
    for song in loaded:
        f.write(f"{song['title']} - {song['artist']}\n")
print("✅ Summary saved!")
