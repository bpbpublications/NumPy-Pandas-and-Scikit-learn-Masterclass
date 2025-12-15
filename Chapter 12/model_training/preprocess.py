import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pipeline.binner import TenureBinner
from loguru import logger

# Set up logging to file and console
os.makedirs("logs", exist_ok=True)
logger.add("logs/preprocess.log", rotation="1 MB")

# =============== COLUMN GROUPS ===============
minmax_median_cols = ['Age']
standard_mean_cols = ['MonthlyCharges']
standard_median_cols = ['CustomerSupportCalls']
robust_median_cols = ['ServiceUsage']
categorical_cols = ['Gender', 'ContractType', 'PaymentMethod']

# =============== PIPELINES ===============
minmax_median_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

standard_mean_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

standard_median_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

robust_median_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

tenure_binner_pipeline = Pipeline([
    ('tenure_binner', TenureBinner(
        column='Tenure',
        bins=[0, 6, 15, 22, float('inf')],
        labels=['New', 'Developing', 'Established', 'Loyal']
    )),
    ('encoder', OneHotEncoder())
])

# =============== STRUCTURED COLUMN TRANSFORMER ===============
structured_preprocessor = ColumnTransformer(transformers=[
    ('minmax_median', minmax_median_pipeline, minmax_median_cols),
    ('standard_mean', standard_mean_pipeline, standard_mean_cols),
    ('standard_median', standard_median_pipeline, standard_median_cols),
    ('robust_median', robust_median_pipeline, robust_median_cols),
    ('categorical', categorical_pipeline, categorical_cols)
])

# =============== FULL PREPROCESSOR (FeatureUnion) ===============
full_preprocessor = FeatureUnion([
    ('structured', structured_preprocessor),
    ('tenure_bins', tenure_binner_pipeline)
])

# =============== SAVE FUNCTION ===============
def save_preprocessor(preprocessor, path='models/preprocessor.joblib'):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(preprocessor, path)
        logger.success(f"Preprocessor saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessor: {e}")

# =============== OPTIONAL: FIT/SAVE FROM DATA ===============
def fit_and_save_preprocessor(data_path='data/synth_customer_churn.csv', 
                              save_path='models/preprocessor.joblib'):
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info("Data loaded successfully")

        # Drop target and ID columns for X
        X = df.drop(columns=['ChurnCategory', 'CustomerID'])
        logger.info("Fitting full preprocessor pipeline...")
        full_preprocessor.fit(X)
        logger.success("Preprocessor fitted successfully")

        save_preprocessor(full_preprocessor, save_path)
    except Exception as e:
        logger.exception(f"Error during fit and save: {e}")

if __name__ == "__main__":
    logger.info("Starting preprocessor fit and save script")
    fit_and_save_preprocessor()
    logger.success("Script finished")
