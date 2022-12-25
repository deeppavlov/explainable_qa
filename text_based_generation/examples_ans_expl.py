import requests


# Generate detailed answer explanation from the question and paragraph

questions_list = [
                   "Кто был первым человеком в космосе?",
                   "Где живут кенгуру?",
                   "Какое самое глубокое озеро в мире?"
                 ]
for question in questions_list:
    res = requests.post("http://0.0.0.0:8006/ans_expl", json={"questions": [question]}).json()
    print(res)
