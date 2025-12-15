from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, mean_squared_log_error)
import numpy as np

def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['R²'] = r2_score(y_true, y_pred)
    
    metrics['MSLE'] = (
        mean_squared_log_error(y_true, y_pred)
        if np.all(y_pred >= 0) else None
    )
    metrics['MAPE(%)'] = (
        np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if np.all(y_true != 0) else None
    )
    
    return metrics
