from sklearn.model_selection import TimeSeriesSplit

def rolling_forecasting_origin(series, n_splits:int=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    for train_index, test_index in tscv.split(series):
        train, test = series.iloc[train_index], series.iloc[test_index]
        splits.append((train, test))

    return splits

def hold_out_split(series, test_size:float=0.2):
    split_idx = int(len(series) * (1 - test_size))
    train, test = series[:split_idx], series[split_idx:]
    return train, test
