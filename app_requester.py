import requests, json

resp = requests.post("http://localhost:5000/predict", json={'text': 'Make a platformer', 'len':96})

print(resp.json(), resp.status_code, resp.reason)