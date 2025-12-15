import joblib
import numpy as np
import pandas as pd
from flask import Flask
from flask_restx import Api, Resource, fields
from loguru import logger
import os

# ==== Logging setup ====
os.makedirs("logs", exist_ok=True)
logger.add("logs/api.log", rotation="1 MB")

# ==== Load model pipeline ====
MODEL_PATH = "models/stack_class_pipe.joblib"

try: 
    model = joblib.load(MODEL_PATH)
    logger.success(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    raise

# ==== Flask app setup ====
app = Flask(__name__)
api = Api(app, version="1.0", title="Customer Churn Prediction API",
          description="Predict churn risk category for customer", 
          doc="/docs")
ns = api.namespace("predict", 
                   description="Churn prediction operations")

# ==== Define the expected input for Swagger docs ====
input_model = api.model("Input", {
    'CustomerID': fields.String(example='CUST00001'),
    'Age': fields.Float(example=34.78),
    'Gender': fields.String(enum=['Male', 'Female']),
    'Tenure': fields.Float(example=22.46),
    'MonthlyCharges': fields.Float(example=86.31),
    'ServiceUsage': fields.Float(example=1.36),
    'ContractType': fields.String(enum=['Month-to-Month', 
                                        'One-Year', 'Two-Year']),
    'PaymentMethod': fields.String(enum=['Credit Card', 'Bank Transfer', 
                                         'Electronic Check', 'Mailed Check']),
    'CustomerSupportCalls': fields.Float(example=0.0)})

# ==== Map the churn categories to the risk labels ====
CHURN_RISK_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# ==== Define the prediction endpoint ====
@ns.route("/")
class Predict(Resource):
    """
    Endpoint to predict churn risk category for a customer.
    """
    @ns.expect(input_model)
    def post(self):
        data = api.payload
        logger.info(f"Received prediction request: {data}")
        # Prepare the input DataFrame

        try:
            input_df = pd.DataFrame([{
                'CustomerID': data['CustomerID'],
                'Age': data['Age'],
                'Gender': data['Gender'],
                'Tenure': data['Tenure'],
                'MonthlyCharges': data['MonthlyCharges'],
                'ServiceUsage': data['ServiceUsage'],
                'ContractType': data['ContractType'],
                'PaymentMethod': data['PaymentMethod'],
                'CustomerSupportCalls': data['CustomerSupportCalls'],
            }])

            input_X = input_df.drop(columns=['CustomerID'])
            logger.debug(f"Input DataFrame for prediction: {input_X}")
            # Predict the churn risk category
            pred = model.predict(input_X)[0]
            pred_proba = model.predict_proba(input_X)[0]
            logger.info(f"Prediction result: {pred}, Probabilities: {pred_proba}")
            #Create a response dictionary
            response = {
                    "CustomerID": data['CustomerID'],
                    "prediction": CHURN_RISK_MAP.get(pred, "Unknown"),
                    "prediction_probs": {
                        "Low Risk": float(pred_proba[0]),
                        "Medium Risk": float(pred_proba[1]),
                        "High Risk": float(pred_proba[2])}
                        }
            logger.success(f"Prediction result for {data['CustomerID']}: {response}")
            return response, 200
        except Exception as e:
            logger.error(f"Error processing prediction request: {e}")
            return {"error": str(e)}, 400
        
# ==== Health check endpoint ====       
@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        logger.info("Health check endpoint called")
        if model is None:
            logger.error("Model is not loaded!")
            return {"status": "unhealthy", 
                    "reason": "Model not loaded"}, 500
        try:
            dummy = pd.DataFrame([{
                "Age": 30,
                "Gender": "Male",
                "Tenure": 5,
                "MonthlyCharges": 50,
                "ServiceUsage": 1,
                "ContractType": "Month-to-Month",
                "PaymentMethod": "Credit Card",
                "CustomerSupportCalls": 1,
            }])
            _ = model.predict(dummy)
        except Exception as e:
            logger.error(f"Model inference failed during health check: {e}")
            return {"status": "unhealthy", "reason": "Inference failed"}, 500
        logger.info("Health check passed")
        return {"status": "healthy"}, 200
    
if __name__ == "__main__":
    logger.info("Starting Churn Classifier API Server.")
    app.run(debug=False)