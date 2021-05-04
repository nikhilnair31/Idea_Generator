import requests, json

resp = requests.post("http://localhost:5000/predict", json={'len':300})

print(resp.json(), resp.status_code, resp.reason)