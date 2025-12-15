import yaml
import pandas as pd
import joblib
import importlib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from preprocessing import load_config, create_preprocessor
from scipy.stats import randint

def load_data(path):
    """
    Load data from a CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(path)

def create_model(config):
    """
    Create a regression model based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        model: Initialized regression model.
    """

    class_path = config['model']['model_class']
    module_path, class_name = class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model_params = config['model'].get('model_params', {})
    return model_class(**model_params)

def parse_param_dist(param_config):
    """
    Parse the parameter distribution configuration for RandomizedSearchCV.
    Args:
        param_config (dict): Parameter configuration.
    Returns:
        dict: Parameter distribution for RandomizedSearchCV.
    """
    dist = {}
    for param, conf in param_config.items():
        if conf['type'] == 'int':
            dist[f'regressor__{param}'] = randint(conf['low'], conf['high'])
        elif conf['type'] == 'choice':
            dist[f'regressor__{param}'] = conf['values']
    return dist

def main(config_path):
    """
    Main function to load configuration, data, preprocess, train model, and evaluate. 
    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    df = load_data(config['data']['path'])
    X = df.drop(columns=config['data']['target'])
    y = df[config['data']['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    preprocessor = create_preprocessor(config)
    model = create_model(config)
    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', model)
    ])

    if config['model']['tune_hyperparameters']:
        param_dist = parse_param_dist(config['model']['param_distributions'])
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=config['search']['n_iter'],
            cv=config['search']['cv'],
            scoring=config['search']['scoring'],
            random_state=config['data']['random_state'],
            n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("Best Parameters:", search.best_params_)
    else:
        best_model = pipe.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("RMSE", root_mean_squared_error(y_test, y_pred))
    print("Feature Importances:", best_model.named_steps['regressor'].feature_importances_)

    output_path = config.get('output', {}).get('model_path', 'outputs/regression_model.joblib')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main("config_regression.yaml")
