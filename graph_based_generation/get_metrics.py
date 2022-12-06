import requests


res = requests.post("http://0.0.0.0:8007/get_metrics", json={"num_samples": 100}).json()
print(res)
