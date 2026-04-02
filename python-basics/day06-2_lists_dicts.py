#day-6-2_lists_dicts_examples 

my_genres = ["Action", "Comedy", "Horror", "Romance", "Sci-Fi"]
my_genres.append("Thriller")
my_genres.remove("Horror")
print(my_genres)

ratings = [8.8, 7.3, 9.1, 6.5, 8.0]
print("Ascending : ", sorted(ratings))
print("Descending: ", sorted(ratings, reverse=True))

movie = {
    "title": "Inception",
    "director": "Nolan",
    "year": 2010,
    "country": "USA",
    "language": "English"
}

print(f"director : {movie['director']}")
print(f"country : {movie['country']}")

my_collection = {
    "watched": ["Inception", "Parasite", "Tenet"],
    "wishlist": ["Dune", "Oppenheimer"],
    "favorites": ["Interstellar", "La La Land", "Her"],
}

print("\n=== my_collection ===")
for name, movies in my_collection.items(): #not item -> items
    print(f"{name} : {len(movies)} movies") #len ( )