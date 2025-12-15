import requests
url = "http://localhost:8000/model"
file_path = "models/stack_class_pipe_new.joblib"

with open(file_path, "rb") as f:
    files = {"file": ("stack_class_pipe_new.joblib", f, 
                      "application/octet-stream")}
    response = requests.put(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
