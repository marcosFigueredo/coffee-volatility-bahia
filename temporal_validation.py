"""Annual refits before January 1; forecasts issued after the close at t.

Horizons count observed quotes, not calendar days. Supervised training
labels must be fully observed strictly before the annual refit cutoff.
"""
import numpy as np
import pandas as pd


def annual_split(df, test_year, h):
    df = df.sort_values('date').copy()
    df['date'] = pd.to_datetime(df['date'])
    df['target_end_date'] = df['date'].shift(-h)
    cutoff = pd.Timestamp(test_year, 1, 1)
    train = df[(df.date < cutoff) & (df.target_end_date < cutoff)].dropna(
        subset=[f'target_rv_h{h}', 'return_1d'])
    test = df[df.date.dt.year == test_year].dropna(subset=[f'target_rv_h{h}'])
    return train, test


def prediction_rows(test, pred, train, series, h, year, model, experiment):
    out = test[['date', 'target_end_date']].copy()
    out['series'], out['horizon'], out['test_year'] = series, h, year
    out['model'], out['experiment'] = model, experiment
    out['train_cutoff'] = pd.Timestamp(year, 1, 1)
    out['train_last_date'] = train.date.max()
    out['train_last_target_end'] = train.target_end_date.max()
    out['n_train'] = len(train)
    out['actual_rv'], out['predicted_rv'] = test[f'target_rv_h{h}'].to_numpy(), pred
    out['valid_prediction'] = np.isfinite(out.predicted_rv) & (out.predicted_rv >= 0)
    ratio = np.maximum(out.actual_rv ** 2, 1e-4) / np.maximum(out.predicted_rv ** 2, 1e-4)
    out['qlike'] = ratio - np.log(ratio) - 1
    out['absolute_error'] = abs(out.actual_rv - out.predicted_rv)
    out['squared_error'] = (out.actual_rv - out.predicted_rv) ** 2
    return out
