import requests


res = requests.post("http://0.0.0.0:8006/get_metrics_expl", json={"num_samples": "all"}).json()
print(res)
