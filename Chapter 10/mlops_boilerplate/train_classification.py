
import yaml
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocessing import load_config, create_preprocessor
from scipy.stats import randint
import numpy as np
import importlib
import os

def load_data(path):
    """Load data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(path)

def create_model(config):
    """Dynamically load the model class and create an instance.

    Args:
        config (dict): Configuration dictionary containing model class and parameters.
        Example:
        config = {
            'model': {
                'model_class': 'sklearn.ensemble.RandomForestClassifier',
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            }
        }
        The 'model' key should contain the 'tune_hyperparameters' key, which is a boolean indicating
        whether to tune hyperparameters or not.
        The 'param_distributions' key should contain the hyperparameters to tune.
        Example:
        config = {
            'model': {
                'tune_hyperparameters': True,
                'param_distributions': {
                    'n_estimators': {
                        'type': 'int',
                        'low': 10,
                        'high': 200
                    },
                    'max_depth': {
                        'type': 'int',
                        'low': 1,
                        'high': 20
                    },
                    'min_samples_split': {
                        'type': 'choice',
                        'values': [2, 5, 10]
                    }
                }
            }
        }
        The 'param_distributions' key should contain the hyperparameters to tune.
    

    Returns:
        model: An instance of the model class specified in the configuration.
        The model class should be specified in the format 'module.ClassName'.
        The parameters for the model should be specified in the 'model_params' key.
        The 'model_params' should be a dictionary with parameter names as keys and their values.
    """
    class_path = config['model']['model_class']
    module_path, class_name = class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model_params = config['model'].get('model_params', {})
    return model_class(**model_params)

def parse_param_dist(param_config):
    """Parse the parameter distribution configuration for hyperparameter tuning.

    Args:
        param_config (dict): Dictionary containing parameter distributions.
        The parameter distributions should be specified in the 'param_distributions' key.
        The 'param_distributions' should be a dictionary with parameter names as keys and their values.
        Example:
        param_config = {
            'n_estimators': {
                'type': 'int',
                'low': 10,
                'high': 200
            },
            'max_depth': {
                'type': 'int',
                'low': 1,
                'high': 20
            },
            'min_samples_split': {
                'type': 'choice',
                'values': [2, 5, 10]
            }
        }

    Returns:
        dict: Dictionary containing the parameter distributions for hyperparameter tuning.
        The parameter distributions should be specified in the 'param_distributions' key.
        The 'param_distributions' should be a dictionary with parameter names as keys and their values.
        Example:
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_depth': randint(1, 20),
            'min_samples_split': [2, 5, 10]
        }
    """
    dist = {}
    for param, conf in param_config.items():
        if conf['type'] == 'int':
            dist[f'classifier__{param}'] = randint(conf['low'], conf['high'])
        elif conf['type'] == 'choice':
            dist[f'classifier__{param}'] = conf['values']
    return dist

def main(config_path):
    """Main function to train a classification model.
    Args:
        config_path (str): Path to the configuration file.
        The configuration file should be in YAML format.
    """
    config = load_config(config_path)
    df = load_data(config['data']['path'])
    X = df.drop(columns=config['data']['target'])
    y = df[config['data']['target']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=config['data']['test_size'], 
                                                        random_state=config['data']['random_state'])
    preprocessor = create_preprocessor(config)
    model = create_model(config)
    pipe = Pipeline([('preprocessing', preprocessor), ('classifier', model)])

    if config['model']['tune_hyperparameters']:
        param_dist = parse_param_dist(config['model']['param_distributions'])
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=config['search']['n_iter'],
                                    cv=config['search']['cv'], scoring=config['search']['scoring'],
                                    random_state=config['data']['random_state'], n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("Best Parameters:", search.best_params_)
    else:
        best_model = pipe.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    output_path = config.get('output', {}).get('model_path', 'best_model.joblib')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    main("config_classification.yaml")
