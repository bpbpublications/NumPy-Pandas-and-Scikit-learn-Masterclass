from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, mean_squared_log_error)
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics for model evaluation.
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    Returns
    -------
    metrics : dict
        Dictionary containing computed metrics.
    
    Notes
    -----
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: Coefficient of Determination
    - MSLE: Mean Squared Logarithmic Error
    - MAPE: Mean Absolute Percentage Error
    - MAPE(%) is expressed as a percentage.
    - MSLE is only computed if all predicted values are non-negative.
    - MAPE is only computed if all true values are non-zero.
    - The function handles cases where the predictions or true values are
      not suitable for certain metrics (e.g., negative predictions for MSLE,
      zero true values for MAPE).
    """
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
