import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold


ESTIMATORS = {
    'random_forest_classifier': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'random_forest_regressor': RandomForestRegressor,
    'linear_regression': LinearRegression
}

def load_config(config_path):
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_preprocessor(config):
    """
    Create a preprocessing pipeline based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    # Load configuration
    cat_features = config['data']['categorical_features']
    num_features = config['data']['numerical_features']
    preprocessing_cfg = config.get('preprocessing', {})
    use_pca = preprocessing_cfg.get('use_pca', False)
    pca_components = preprocessing_cfg.get('pca_components', 2)
    use_feature_selection = preprocessing_cfg.get('use_feature_selection', False)
    selection_method = preprocessing_cfg.get('method', 'rfecv')

    num_steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]

    if use_pca:
        num_steps.append(('pca', PCA(n_components=pca_components)))

    if use_feature_selection and selection_method == 'rfecv':
        estimator_name = preprocessing_cfg.get('estimator', 'random_forest')
        task_type = config.get("task", "classification")

        # Determine the appropriate estimator key
        if task_type == "regression":
            estimator_key = f"{estimator_name}_regressor"
            cv_strategy = KFold(n_splits=preprocessing_cfg.get('cv', 5))
            scoring_metric = preprocessing_cfg.get('scoring', 'r2')
        else:
            estimator_key = f"{estimator_name}_classifier"
            cv_strategy = StratifiedKFold(n_splits=preprocessing_cfg.get('cv', 5))
            scoring_metric = preprocessing_cfg.get('scoring', 'f1_macro')

        estimator_class = ESTIMATORS.get(estimator_key)
        if not estimator_class:
            raise ValueError(f"Unsupported estimator for RFECV: {estimator_key}")
        estimator = estimator_class(random_state=config['data']['random_state'])

        num_steps.append(('feature_selection', RFECV(
            estimator=estimator,
            cv=cv_strategy,
            scoring=scoring_metric
        )))

    num_pipeline = Pipeline(steps=num_steps)
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    return preprocessor
