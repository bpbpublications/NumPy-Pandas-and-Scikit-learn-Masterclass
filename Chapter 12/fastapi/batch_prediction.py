import requests
import json
from loguru import logger
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logger.add("logs/fastapi_app.log", rotation="1 MB", retention="10 days")

# Endpoint
url = "http://localhost:8000/batch_predict"

# Load JSON payload
with open("json/batch.json", "r") as f:
    data = json.load(f)

# Send request
logger.info("Sending batch prediction request...")
response = requests.post(url, json=data)

# Process response
if response.status_code == 200:
    result = response.json()
    
    # Timestamped filename for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "json"
    filename = os.path.join(save_path, f"batch_response_{timestamp}.json")
    
    with open(filename, "w") as out:
        json.dump(result, out, indent=4)
    
    logger.success(f"Batch prediction successful. Response saved to {filename}")
    logger.debug(f"Prediction response: {json.dumps(result, indent=2)}")
else:
    logger.error(f"Batch prediction failed. Status: {response.status_code}, Error: {response.text}")
