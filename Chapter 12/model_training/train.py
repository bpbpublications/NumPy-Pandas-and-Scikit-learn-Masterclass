import os
import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from loguru import logger

# --- PATHS ---
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

RAW_DATA_PATH = os.path.join(DATA_DIR, "synth_customer_churn.csv")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "stack_class_pipe.joblib")
TRAIN_LOG_PATH = os.path.join(LOGS_DIR, "train.log")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logger.add(TRAIN_LOG_PATH, rotation="1 MB")

# --- PARAMETERS ---
# These parameters can be adjusted based on your dataset and requirements
OVERSAMPLE_TARGET1_RATIO = 1.4  
OVERSAMPLE_TARGET2_RATIO = 2.5  
RANDOM_SEED = 42  

# If you have a custom transformer, import it so joblib can load the preprocessor
from pipeline.binner import TenureBinner

def main():
    try:
        # 1. Load raw data
        logger.info(f"Loading raw data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        y = df['ChurnCategory'].map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2})
        X = df.drop(columns=['ChurnCategory', 'CustomerID'])
        logger.success("Raw data loaded.")

        # 2. Load preprocessor
        logger.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.success("Preprocessor loaded.")

        # 3. Train/test split
        logger.info("Splitting data into train/test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        logger.success("Data split complete.")

        # 4. Define stacking classifier
        base_learners = [
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True)),
            ('gb', GradientBoostingClassifier())
        ]
        meta_model = LogisticRegression()
        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_model
        )
        logger.info("Stacking classifier defined.")

        # 5. Define resampler
        original_counts = Counter(y_train)
        target_counts = {
            1: int(original_counts[1] * OVERSAMPLE_TARGET1_RATIO),
            2: int(original_counts[2] * OVERSAMPLE_TARGET2_RATIO)
        }
        smotenn = SMOTEENN(sampling_strategy=target_counts, random_state=RANDOM_SEED)
        logger.info("SMOTEENN resampler set up.")

        # 6. Build pipeline: Preprocessor + Resample + Model
        stacking_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("resample", smotenn),
            ("stacking_clf", stacking_clf)
        ])
        logger.info("Full pipeline created. Training now...")

        stacking_pipeline.fit(X_train, y_train)
        logger.success("Pipeline training complete.")

        # 7. Save pipeline (to models/)
        joblib.dump(stacking_pipeline, MODEL_SAVE_PATH)
        logger.success(f"Model pipeline saved to {MODEL_SAVE_PATH}")

        # 8. Evaluate
        y_pred = stacking_pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        logger.info("Classification report (test set):\n" + report)

    except Exception as e:
        logger.exception(f"Error in training script: {e}")

if __name__ == "__main__":
    logger.info("Starting training script.")
    main()
    logger.info("Training script finished.")
