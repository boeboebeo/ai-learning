# day 07 - 3 classes _ example

class MenuItem:
    def __init__(self, name, price, calories, category, is_hot=False):
        self.name = name
        self.price = price
        self.calories = calories
        self.category = category
        self.is_hot = is_hot

    def describe(self): #menu 하나 디테일 설명
        print(f"\nname : {self.name} - {self.price}")
        print(f"    calories : {self.calories}")
        print(f"    category : {self.category}")
        print(f"    is expensive? : {self.is_expensive()}")
        if self.is_hot:
            print("    is hot? : Yes") #기본값 = False. True 면 (뜨거움)
        else: 
            print("    is hot? : No") #is_hot 안 넣으면 자동으로 False (안 뜨거움)

    def is_expensive(self): #비싼지 싼지 설명
        if self.price >= 5000:
            return "Premuium"
        else:
            return "Regular"
        
class Cafe:
    def __init__(self, name):   #Cafe 는 이름만 받으면 됨
        self.name = name
        self.menus = [] 

    def add_item(self, menu): #self. - 만 모든 메소드에서 공유!
        self.menus.append(menu)

    def show(self):
        print("===menu list===")
        print(f"\nAverage price : {self.average():.1f}")
        for i, menu in enumerate(self.menus, 1):
            print(f"{i}. {menu.name} - {menu.price}") #여기가! 각 menu 가 하나씩 있는거니까 self.name 이면 안되고! menu.name 이여야 함


    def average(self):
        if not self.menus:
            return 0
        else:
            return sum(b.price for b in self.menus) / len(self.menus)
        


item1 = MenuItem("Americano", 4500, 10, "Coffee", True)
item2 = MenuItem("Latte", 5500, 180, "Coffee", True)
item3 = MenuItem("Cheesecake", 6500, 450, "Dessert")

my_cafe = Cafe("Claude Cafe")
my_cafe.add_item(item1)
my_cafe.add_item(item2)
my_cafe.add_item(item3)
my_cafe.show()
item2.describe()

# 추가하고 show(), describe() 호출 하기!