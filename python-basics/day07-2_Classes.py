# day 07 - classes.example

class Book:
    def __init__(self, title, author, pages, genre, available=True):
        self.title = title
        self.author = author
        self.pages = pages
        self.genre = genre
        self.available = available

    def describe(self):
        print(f"\n{self.title} by {self.author}")
        print(f"    genre : {self.genre}")
        print(f"    pages : {self.pages} ({self.is_long()})")

        if self.available: 
            print(f"    available : Yes") #True or False 출력
        else:
            print(f"    available : No")
        
    def is_long(self):
        if self.pages >= 300 :
            return "Long read"
        else :
            return "Quick read"
        
class Library:
    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):  #그럼 저기 self.books 안에 내가 추가하는 book 이 들어가는 것. 이름은 뭐여도 상관 없음
        self.books.append(book)

        #self : 이 객체 자신 ex. My_booklist.add_book(book1) 이면 
        #   My_booklist 가 self 가 됨
        #   other_list.add_book(book1) 이면 other_list 가 self 가 됨
        #   method 끼리는 book같은거 공유 안함 걍 다른 존재
        #   self.books -> 이런건 모든 메소드 에서 공유!


    def show(self):
        print("My favorite books")
        print(f"\nTotal : {len(self.books)} books\n")
        print(f"Average Pages : {self.average():.1f}")

        for i, book in enumerate(self.books, 1):
            print(f"{i}. {book.title} - {book.author}")


    def average(self): #method 이름은 소문자로 시작 !
        if not self.books:
            return 0
        return sum(p.pages for p in self.books) / len(self.books)


book1 = Book("Atomic Habits", "James Clear", 320, "Self-help")
book2 = Book("The Alchemist", "Paulo Coelho", 208, "Fiction")
book3 = Book("Clean Code", "Robert Martin", 431, "Programming")

My_booklist = Library("my favorites")
My_booklist.add_book(book1)
My_booklist.add_book(book2)
My_booklist.add_book(book3)
My_booklist.show()
book1.describe()
