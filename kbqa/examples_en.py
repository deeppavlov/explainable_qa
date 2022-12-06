import requests


questions = ["When did Jean-Paul Sartre move to Le Havre?",
             "What is the capital of Belarus?",
             "What team did Lionel Messi play for in 2004?",
             "Who is the painter of Mona Lisa?",
             "What position was held by Harry S. Truman on 1/3/1935?",
             "Who directed Forrest Gump?"
             ]

for question in questions:
    res = requests.post("http://0.0.0.0:8008/respond", json={"questions": [question]}).json()
    print(res)
