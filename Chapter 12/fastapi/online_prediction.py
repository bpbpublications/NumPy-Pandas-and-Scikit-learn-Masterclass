import requests

url = "http://localhost:8000/predict"

data = {
    "CustomerID": "CUST00001",
    "Age": 35,
    "Gender": "Male",
    "Tenure": 22.46,
    "MonthlyCharges": 86.31,
    "ServiceUsage": 1.36,
    "ContractType": "Month-to-Month",
    "PaymentMethod": "Credit Card",
    "CustomerSupportCalls": 0.0
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Prediction Response:", response.json())
