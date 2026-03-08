#Example

def classify_popular(streams):
    if streams < 500000:
        return "Underground"
    elif streams < 1000000:
        return "Rising"
    elif streams < 10000000:
        return "Popular"
    else:
        return "Mega Hit"
    
"""
list vs Tuple

list : [] can add or remove / mutable(나중에 코드로 변경가능)
    songs = ["Hype Boy", "Blinding Lights"]
    songs[0] = "New song" : 이렇게 바꿀 수 있음
Tuple : (), fixed, read only / immutable(변경불가능)
    song = ("Hype Boy", "NewJeans")
    song[0] = "New Song"  # 실행하면 에러남!
"""

songs = [
    ("Hype Boy", "NewJeans", 15000000),
    ("UNFORGIVEN", "LE SSERAFIM", 8000000),
    ("mudmud song", "mudmud", 300000),
    ("garage track", "unknown", 1200)
]

print("===Popularity Classification===")
for song, singer, streams in songs:
    streaming = classify_popular(streams)
    print(f"{song} by {singer} ({streams} streams) -> {streaming}")

