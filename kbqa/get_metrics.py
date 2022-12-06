import requests


res = requests.post("http://0.0.0.0:8008/get_metrics", json={"num_samples": 3000}).json()
print(res)
