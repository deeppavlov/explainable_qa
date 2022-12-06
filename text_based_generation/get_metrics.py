import requests


res = requests.post("http://0.0.0.0:8006/get_metrics", json={"num_samples": 20}).json()
print(res)
