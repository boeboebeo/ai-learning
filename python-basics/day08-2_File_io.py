# day08-2_File_inout_example

"""
**구현할 것:**

1. `movies.json` 으로 저장
2. `movies.json` 불러와서 출력

- Inception (8.8) by Nolan
- Parasite (8.5) by Bong
- Interstellar (8.6) by Nolan

"""


import json

movies = [
    {"title": "Inception", "director": "Nolan", "rating": 8.8, "genre": "Sci-Fi"},
    {"title": "Parasite", "director": "Bong", "rating": 8.5, "genre": "Thriller"},
    {"title": "Interstellar", "director": "Nolan", "rating": 8.6, "genre": "Sci-Fi"},
]

#with : 는 마지막에 f.close()없이도 알아서 닫아주는 키워드 -> 자동으로 닫힘
    #open()은 파이썬 내장 함수

with open("movies.json", "w", encoding="utf-8")as f:
    json.dump(movies, f, indent=2, ensure_ascii=False) #json.dump = 저장
print("✅ Saved to movies.json")

with open("movies.json", "r", encoding="utf-8")as f:
    loaded = json.load(f)
print(f"✅ loaded {len(loaded)} movies")

for movie in loaded:
    print(f"- {movie['title']} ({movie['rating']}) by {movie['director']}")

with open("movies_summary.txt", "w")as f:
    for movie in loaded:
        f.write(f"{movie['title']} - {movie['director']}\n") 
print("✅ summary saved!")

#f.write() : 파일에 써줘
#f.read() : 파일 읽어줘
#f.close() : 파일 닫아줘