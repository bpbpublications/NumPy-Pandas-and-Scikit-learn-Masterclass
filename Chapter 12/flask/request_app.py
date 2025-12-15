import requests
import json
url = "http://localhost:5000/predict/"

with open('sample_cust_input.json', 'r') as f:
    payload = json.load(f)

response = requests.post(url, 
                         json=payload)
print(response.json())
