movies = [
    {"title": "Inception", "director": "Nolan", "rating": 8.8, "genre": "Sci-Fi"},
    {"title": "Parasite", "director": "Bong", "rating": 8.5, "genre": "Thriller"},
    {"title": "Interstellar", "director": "Nolan", "rating": 8.6, "genre": "Sci-Fi"},
    {"title": "La La Land", "director": "Chazelle", "rating": 8.0, "genre": "Romance"},
    {"title": "Tenet", "director": "Nolan", "rating": 7.3, "genre": "Sci-Fi"},
]


print("=== My movie list ===")

for i, movie in enumerate(movies, 1):
    print(f"{i}. {movie['title']} - {movie['director']} ({movie['rating']})") #여기 '' 작은 따옴표

total_rating = 0
for rate in movies:
    total_rating += rate['rating']

average = total_rating // len(movies) # / : 소수점 나눗셈, // 정수 나눗셈
print(f"\nAverage rate : {average:.2f}")

Min_rate = min(movies, key=lambda x: x["rating"])
Max_rate = max(movies, key=lambda x: x["rating"])
print(f"\nMinimum rate : {Min_rate['title']} - {Min_rate['rating']}")
print(f"Maximum rate : {Max_rate['title']} - {Max_rate['rating']}")


count = 0
index = 0
while index < len(movies):
    if movies[index]['genre'] == "Sci-Fi": #movies[index] -> movies 의 index 번째(0번째 -> Inception 딕셔너리) 딕셔너리를 가리킴
                                            #while : 은 index 를 직접 움직여 줘야 함 
        count += 1
    index += 1

print(f"\nNumber of Sci-Fi : {count}")
