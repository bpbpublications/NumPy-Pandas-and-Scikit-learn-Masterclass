import itertools
import statsmodels.api as sm
import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd

def search_arima_orders(y_train, 
                        s=12,
                        p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3),
                        P_range=range(0, 2), D_range=range(0, 2), Q_range=range(0, 2),
                        period_str="M", 
                        verbose=True,
                        top_n=10):


    if not isinstance(y_train.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        raise ValueError("Index must be a PeriodIndex or DatetimeIndex.")
    if y_train.index.freq is None:
        try:
            y_train.index = y_train.index.to_period(period_str)
        except:
            y_train.index.freq = period_str

    use_seasonal = len(y_train) >= 2 * s
    results = []

    orders = list(itertools.product(p_range, d_range, q_range))
    seasonal_orders = [(0, 0, 0)] if not use_seasonal else list(itertools.product(P_range, D_range, Q_range))

    total_combos = len(orders) * len(seasonal_orders)
    if verbose:
        print(f"🔍 Evaluating {total_combos} (p,d,q)(P,D,Q) combinations...")

    for order in tqdm(orders, desc="Grid Searching", disable=not verbose):
        for seasonal in seasonal_orders:
            seasonal_order = (seasonal[0], seasonal[1], seasonal[2], s) if use_seasonal else (0, 0, 0, 0)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.tsa.statespace.SARIMAX(
                        y_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    result = model.fit(disp=False)
                    results.append({
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'aic': result.aic
                    })
            except Exception:
                continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='aic').reset_index(drop=True)

    if verbose:
        print("\n✅ Top Models by AIC:")
        print(results_df.head(top_n))

    return results_df
