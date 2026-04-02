#day 10-2 mini project 

import json #json 을 불러와야 밑에 Json 파일로 저장하는걸 대응할 수 있음

class Book:
    def __init__(self, title, author, page, genre, rating=None):
        self.title = title
        self.author = author
        self.page = page
        self.genre = genre 
        self.rating = rating

    def get_length(self):
        if self.page >= 300:
            return "Long"
        else :
            return "Short"
        
    def to_dict(self): #딕셔너리 구성하는 법
        return { #key : value 순서! 헷갈리지 말기
            "title" : self.title,
            "author" : self.author,
            "page" : self.page,
            "genre" : self.genre,
            "rating" : self.rating,
        }

class BookAnalyzer:
    def __init__(self):
        self.books = []

    def add_book(self, book): #(self, book) 형태여야 book 이라는 객체가 self.books 안에 들어갈 수 있음
        self.books.append(book)

    def get_states(self):
        pages = [b.page for b in self.books]
        return{     #딕셔너리 형태로는 : 표기가 들어가야 함 
            "total_books" : len(self.books),
            "average_books" : round(sum(pages)/len(self.books), 1),
            "max_page" : max(pages),
            "min_page" : min(pages),
        }
    
    def save(self, filename):
        data = [b.to_dict() for b in self.books] #to_dict() 괄호! -> 함수를 실행해줘야 실행한 뒤 딕셔너리를 받을 수 있음 : 괄호가 없으면 그냥 함수 자체가 담기게 됨
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ saved to {filename}")


    def filter_by_genre(self, genre):
        return [b for b in self.books if b.genre == genre]


    def show_report(self):
        print("\n" + "="*40) # 가운데를 + 로 해야 앞에가 한칸 안 띄어짐
        print("        📚 BOOK ANALYZER REPORT ")
        print("="*40)
        for key, value in self.get_states().items(): #not item ->  itmes 
            print(f"    {key} : {value}") #하나하나 프린트 할 필요 없이 이렇게 for 문 돌려서 출력하는 방법.

        print("\n Genre breakdown: ")
        genres = set(b.genre for b in self.books)
        for genre in genres:
            count = len(self.filter_by_genre(genre))
            print(f"    - {genre} : {count} books")
        print("="*40)

analyzer = BookAnalyzer()

books_data = [
    Book("Atomic Habits", "James Clear", 320, "Self-help", 4.8),
    Book("The Alchemist", "Paulo Coelho", 208, "Fiction", 4.5),
    Book("Clean Code", "Robert Martin", 431, "Programming", 4.7),
    Book("Sapiens", "Yuval Harari", 443, "History", 4.6),
    Book("The Great Gatsby", "Fitzgerald", 180, "Fiction", 4.2),
]

for book in books_data:
    analyzer.add_book(book)

analyzer.show_report()
analyzer.save("analyzed_booklist.json") #filename

