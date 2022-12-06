import requests


questions_list = ["When did Jean-Paul Sartre move to Le Havre?",
                  "Who directed Forrest Gump?",
                  "What is the capital of Belarus?",
                  "What team did Lionel Messi play for in 2004?",
                  "Who is the painter of Mona Lisa?",
                  "What position was held by Harry S. Truman on 1/3/1935?",
                  "Who played Sheldon Cooper in The Big Bang Theory?"
                  ]

for question in questions_list:
    res = requests.post("http://0.0.0.0:8007/ans_expl", json={"questions": [question]}).json()
    print(res)
