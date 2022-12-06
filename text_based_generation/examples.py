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

# Find the answer, the answer paragraph, and generate answer explanation

question_par_list = [["Кто был первым человеком в космосе?",
                      "12 апреля 1961 года Юрий Гагарин стал первым человеком в мировой истории, совершившим полёт в "
                      "космическое пространство."],
                     ["Где живут кенгуру?",
                      "Водятся кенгуру в Австралии, в Тасмании, на Новой Гвинее и на архипелаге Бисмарка. "
                      "Завезены в Новую Зеландию. Большинство видов — наземные, обитают на равнинах, поросших густой "
                      "высокой травой и кустарником."],
                     ["Какое самое глубокое озеро в мире?",
                      "Байкал — озеро тектонического происхождения в южной части Восточной Сибири, самое глубокое "
                      "озеро на планете, крупнейший природный резервуар пресной воды и самое большое по площади "
                      "пресноводное озеро на континенте."]
                     ]

for question, paragraph in question_par_list:
    res = requests.post("http://0.0.0.0:8006/generate", json={"questions": [question],
                                                              "paragraphs": [[paragraph]]}).json()
    print(res)
