import requests


triplets_list = [[["Belarus", "capital", "Minsk"]],
                 [["Forrest Gump", "director", "Robert Zemeckis"]],
                 [["Jean-Paul Sartre", "residence", "Le Havre"], ["Le Havre", "start time", "1931"]],
                 [["Lionel Messi", "member of sports team", "FC Barcelona"], ["FC Barcelona", "start time", "2004"]],
                 [["Mona Lisa", "creator", "Leonardo da Vinci"]],
                 [["Harry S. Truman", "position held", "United States senator"], ["United States senator", "start time", "1935"]]
                 ]
for triplet in triplets_list:
    res = requests.post("http://0.0.0.0:8007/generate", json={"triplets": [triplet]}).json()
    print(res)
