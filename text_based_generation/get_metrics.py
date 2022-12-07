import requests


res = requests.post("http://0.0.0.0:8006/get_metrics",
                    json={"num_samples": 10000, "metric_types": ["ans_expl_given_paragraph"]}).json()
print(res)
