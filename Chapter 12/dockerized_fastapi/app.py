import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
import os
import shutil
from loguru import logger

# SETUP LOGGING
os.makedirs("logs", exist_ok=True)
logger.add("logs/fastapi_app.log", rotation="1 MB", retention="10 days")

# MODEL LOADING FUNCTION
MODEL_PATH = "models/stack_class_pipe.joblib"

def load_model(path=MODEL_PATH):
    """Load the ML model from disk."""
    try:
        model = joblib.load(path)
        logger.success(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        raise

model = load_model()  # Load model on app startup

# INITIALIZE FASTAPI APP
app = FastAPI(
    title="FastAPI Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn risk. Supports batch, model versioning, and helpful info endpoints."
)

# ENUMS FOR ENUMERATED FIELDS
class GenderEnum(str, Enum):
    Male = 'Male'
    Female = 'Female'

class ContractTypeEnum(str, Enum):
    MonthToMonth = 'Month-to-Month'
    OneYear = 'One-Year'
    TwoYear = 'Two-Year'

class PaymentMethodEnum(str, Enum):
    CreditCard = 'Credit Card'
    BankTransfer = 'Bank Transfer'
    ElectronicCheck = 'Electronic Check'
    MailedCheck = 'Mailed Check'

# PYDANTIC SCHEMAS FOR INPUT VALIDATION
class CustomerInput(BaseModel):
    CustomerID: str = Field(..., example="CUST00001")
    Age: float = Field(..., example=35)
    Gender: GenderEnum = Field(..., example='Male')
    Tenure: float = Field(..., example=22.46)
    MonthlyCharges: float = Field(..., example=86.31)
    ServiceUsage: float = Field(..., example=1.36)
    ContractType: ContractTypeEnum = Field(..., example='Month-to-Month')
    PaymentMethod: PaymentMethodEnum = Field(..., example='Credit Card')
    CustomerSupportCalls: float = Field(..., example=0.0)

class BatchInput(BaseModel):
    data: List[CustomerInput]

# TARGET LABEL MAP
CHURN_RISK_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# PREDICTION FUNCTION
def make_prediction(input_df):
    """Runs the prediction using the loaded model and returns result(s)."""
    input_X = input_df.drop(columns=['CustomerID'])
    preds = model.predict(input_X)
    pred_probs = model.predict_proba(input_X)
    results = []
    for i, row in input_df.iterrows():
        pred = preds[i]
        pred_proba = pred_probs[i]
        result = {
            "CustomerID": row['CustomerID'],
            "prediction": CHURN_RISK_MAP.get(pred, "Unknown"),
            "prediction_probs": {
                "Low Risk": float(pred_proba[0]),
                "Medium Risk": float(pred_proba[1]),
                "High Risk": float(pred_proba[2])
            }
        }
        results.append(result)
    return results

# ONE SINGLE PREDICTION ENDPOINT
@app.post("/predict", tags=["Prediction"])
def predict_online(body: CustomerInput):
    """Predict churn risk for a single customer."""
    try:
        input_df = pd.DataFrame([body.dict()])
        logger.debug(f"Online prediction input: {input_df}")
        result = make_prediction(input_df)[0]
        logger.success(f"Prediction result for {body.CustomerID}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

# BATCH PREDICTION ENDPOINT
@app.post("/batch_predict", tags=["Prediction"])
def predict_batch(body: BatchInput):
    """Predict churn risk for a batch of customers."""
    try:
        input_df = pd.DataFrame([item.dict() for item in body.data])
        logger.debug(f"Batch prediction input: {input_df}")
        results = make_prediction(input_df)
        logger.success(f"Batch prediction completed. {len(results)} results returned.")
        return results
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

# MODEL VERSIONING ENDPOINT
@app.put("/model", tags=["Model Versioning"])
async def update_model(file: UploadFile):
    """
    Update the deployed model. Accepts a new .joblib file and swaps the in-memory model.
    """
    global model
    logger.info(f"Received request to update model with file: {file.filename}")
    try:
        # Save the uploaded model temporarily
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        tmp_path = os.path.join(model_dir, "tmp_model.joblib")
        with open(tmp_path, "wb") as temp_buffer:
            shutil.copyfileobj(file.file, temp_buffer)
        logger.debug(f"Model file saved temporarily to {tmp_path}")
        # Test-load the new model
        ph_model = joblib.load(tmp_path)
        # Atomically replace and update
        model = ph_model
        shutil.move(tmp_path, MODEL_PATH)
        logger.success(f"Model updated successfully to {MODEL_PATH}")
        return {"message": "Model updated successfully", "model_path": MODEL_PATH}
    except (OSError, IOError) as e:
        logger.error(f"File handling error: {e}")
        raise HTTPException(status_code=500, detail=f"File handling error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error updating model: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during model update.")

# HEALTH CHECK ENDPOINT
@app.get("/health", tags=["Health"])
def health_check():
    """Basic liveness and inference check."""
    logger.info("Health check endpoint called")
    try:
        # Dummy inference with made-up but valid data
        dummy = pd.DataFrame([{
            "Age": 30,
            "Gender": "Male",
            "Tenure": 5,
            "MonthlyCharges": 50,
            "ServiceUsage": 1,
            "ContractType": "Month-to-Month",
            "PaymentMethod": "Credit Card",
            "CustomerSupportCalls": 1,
            "CustomerID": "dummy"
        }])
        _ = make_prediction(dummy)
        logger.success("Health check passed")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}

# MODEL INFO ENDPOINT
@app.get("/model_info", tags=["Model Versioning"])
def model_info():
    """
    Return info about the currently loaded model (path, last update time).
    """
    try:
        mod_time = os.path.getmtime(MODEL_PATH)
        logger.info("Model info endpoint called")
        return {
            "model_path": MODEL_PATH,
            "last_updated": pd.to_datetime(mod_time, unit='s').isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# FEATURE ENDPOINT
@app.get("/features", tags=["Docs"])
def features():
    """
    Returns the expected input features and types for the prediction endpoint.
    """
    logger.info("Features endpoint called")
    return CustomerInput.schema()["properties"]


# METRICS ENDPOINT
@app.get("/metrics", tags=["Health"])
def metrics():
    """
    Returns basic model file stats for monitoring.
    """
    try:
        stat = os.stat(MODEL_PATH)
        logger.info("Metrics endpoint called")
        return {
            "model_file_size_bytes": stat.st_size,
            "last_updated": pd.to_datetime(stat.st_mtime, unit='s').isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve metrics")

# ROOT WELCOME ENDPOINT
@app.get("/", tags=["Docs"])
def root():
    """Welcome message and link to docs."""
    return {
        "message": "Welcome to the FastAPI Churn Prediction API!",
        "docs": "/docs",
        "redoc": "/redoc"}