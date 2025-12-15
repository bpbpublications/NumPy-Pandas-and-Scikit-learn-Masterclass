import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
def use_decomp_in_df(
        data: pd.DataFrame,column: str, period: int = 12,mode: str = "additive", 
        fill_method: str = "fill_zero", **decompose_args):
    """"Decomposes a time series in a DataFrame and adds the components to the DataFrame"""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("The DataFrame index must be a DatetimeIndex")
    decomp = seasonal_decompose(data[column], model=mode, period=period, **decompose_args)
    data['Trend'] = decomp.trend
    data['Seasonality'] = decomp.seasonal
    data['Residual'] = decomp.resid

    if fill_method == "drop":
        data.dropna(inplace=True)
    elif fill_method == "zero":
        data.fillna(0, inplace=True)
    return data, decomp
