def classify_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
    
def get_student_summary(name, subject, score):
    return(f"{name} in {subject} -> {classify_grade(score)}")

students = [
    {"name": "Alice", "subject": "Math", "score": 92},
    {"name": "Bob", "subject": "English", "score": 78},
    {"name": "Eve", "subject": "Science", "score": 65},
    {"name": "Jay", "subject": "Math", "score": 88},
]

#def get_class_stats(students) <= 여기 파라미터 안에 그 students 리스트가 들어감 
    #함수를 부를 때 넣는 값이 여기에 담김 
    
def get_class_stats(students):
    scores = [student["score"] for student in students]
    return {
        "total_students" : len(scores),
        "average_score" : round(sum(scores) / len(scores), 1),
        "max_score" : max(scores),
        "min_score" : min(scores),
    }


print("===all students===")
for student in students:
    print(get_student_summary(student["name"], student["subject"], student["score"]))

print("\n===students_stats===")
stats = get_class_stats(students) 
for key, value in stats.items():
    print(f"{key}:{value}")