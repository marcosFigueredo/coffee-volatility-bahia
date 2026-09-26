"""E1-E4 ML experiments (section 14 of the plan) for coffee volatility
forecasting. Same walk-forward scheme (expanding window, refit per test
year -- section 17) and metrics (QLIKE, MAE, RMSE -- section 19) as
04_baselines_garch_har.py, so results are directly comparable against the
E0 econometric baselines.

Supervised training labels must end strictly before January 1 of the test
year. Individual predictions and training boundaries are saved for audit
and comparison on identical forecast dates.

Feature sets (ablation per section 13/22):
  E1_own              -- only the target series' own history
  E2_cross_commodity  -- E1 + other coffee variety + soy/cotton/cocoa +
                          national (CEPEA) and international (ICE) coffee
                          benchmarks
  E3_financial        -- E1 + USD/BRL, SELIC, VIX, Brent, Dollar Index,
                          S&P500, Ibovespa, US Treasury 10Y, Fed Funds
  E4_full             -- E1 + E2 + E3

Models: Random Forest, LightGBM, XGBoost (section 15's ML family; LSTM/TCN
are left for a later pass). All three see the exact same
training-median-imputed feature matrix, so no model gets an unfair edge
from native NaN handling (LightGBM/XGBoost support it, sklearn's
RandomForestRegressor does not) -- fairness matters here because CEPEA and
several other columns only exist from 2016 onward, so pre-2016 folds have
real, non-trivial missingness in E2/E4.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from temporal_validation import annual_split, prediction_rows
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
MASTER_PATH = DATA_DIR / "master_dataset_cafe.csv"
OUT_PATH = DATA_DIR / "ml_results_cafe.csv"

HORIZONS = (1, 5, 20)
MIN_TRAIN_YEARS = 3
MIN_TRAIN_OBS = 250
QLIKE_EPS = 1e-4  # see 04_baselines_garch_har.py -- same stale/zero-return series

warnings.filterwarnings("ignore")

OWN_HISTORY_COLS = [
    "return_1d", "return_lag1", "return_lag2", "return_lag3", "return_lag4", "return_lag5",
    "hist_vol_5", "hist_vol_10", "hist_vol_20",
]
CROSS_COMMODITY_COLS = [
    "other_variety_return", "other_variety_vol_5",
    "soy_return", "soy_vol_5", "cotton_return", "cotton_vol_5", "cocoa_return", "cocoa_vol_5",
    "ice_coffee_c_return", "cepea_return",
]
FINANCIAL_COLS = [
    "usd_brl_return", "usd_brl_vol_5", "selic", "vix", "vix_change",
    "brent_return", "dollar_index_return", "sp500_return", "ibovespa_return",
    "us_treasury_10y", "fed_funds",
]

EXPERIMENTS = {
    "E1_own": OWN_HISTORY_COLS,
    "E2_cross_commodity": OWN_HISTORY_COLS + CROSS_COMMODITY_COLS,
    "E3_financial": OWN_HISTORY_COLS + FINANCIAL_COLS,
    "E4_full": OWN_HISTORY_COLS + CROSS_COMMODITY_COLS + FINANCIAL_COLS,
}


# n_jobs pinned to 1 everywhere: joblib's multiprocessing backend
# (RandomForestRegressor's n_jobs=-1) hangs indefinitely in this sandboxed
# environment regardless of actual CPU availability (confirmed with a
# trivial 2000-row fit: 0.03s of CPU accumulated after 41 minutes of wall
# time -- not slowness, a genuine deadlock, most likely the sandbox
# restricting subprocess/shared-memory creation that loky needs). Single
# process/thread execution works reliably, just slower -- RandomForest is
# the main cost (~7s/fit even freed up), so n_estimators/max_depth are
# trimmed to keep the full walk-forward grid tractable; LightGBM/XGBoost
# use their own internal (non-joblib) threading and stayed fast already.
MODEL_FACTORIES = {
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1,
    ),
    "LightGBM": lambda: lgb.LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, min_child_samples=10,
        random_state=42, verbosity=-1, n_jobs=1,
    ),
    "XGBoost": lambda: xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=1,
    ),
}


def qlike(true_var: np.ndarray, pred_var: np.ndarray) -> float:
    ratio = np.maximum(true_var, QLIKE_EPS) / np.maximum(pred_var, QLIKE_EPS)
    return float(np.mean(ratio - np.log(ratio) - 1))


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for k in range(1, 6):
        df[f"return_lag{k}"] = df["return_1d"].shift(k)
    return df


def fit_predict(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    model = MODEL_FACTORIES[model_name]()
    model.fit(X_train_imp, y_train)
    pred = model.predict(X_test_imp)
    return np.clip(pred, 1e-6, None)


def evaluate_series(name: str, df: pd.DataFrame, predictions=None) -> list[dict]:
    df = add_lag_features(df).reset_index(drop=True)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    years = sorted(df["year"].unique())
    start_year = years[0]
    test_years = [y for y in years if y >= start_year + MIN_TRAIN_YEARS and y <= years[-1]]

    records = []
    for h in HORIZONS:
        y_col = f"target_rv_h{h}"
        for exp_name, feat_cols in EXPERIMENTS.items():
            print(f"  h={h} {exp_name}: {len(test_years)} folds...", flush=True)
            for test_year in test_years:
                train, test = annual_split(df, test_year, h)
                if len(train) < MIN_TRAIN_OBS or test.empty:
                    continue

                X_train, y_train = train[feat_cols], train[y_col].to_numpy()
                X_test = test[feat_cols]
                true_rv = test[y_col].to_numpy()
                true_var = true_rv ** 2
                nonzero = true_rv > 0

                for model_name in MODEL_FACTORIES:
                    print(f"    {test_year} {model_name} (n_train={len(train)})...", end=" ", flush=True)
                    t0 = time.time()
                    pred = fit_predict(model_name, X_train, y_train, X_test)
                    print(f"{time.time() - t0:.1f}s", flush=True)
                    if predictions is not None:
                        predictions.append(prediction_rows(test, pred, train, name, h, test_year, model_name, exp_name))
                    pv, tv, tr = pred ** 2, true_var, true_rv
                    records.append({
                        "series": name, "horizon": h, "experiment": exp_name,
                        "model": model_name, "test_year": test_year, "n_train": len(train),
                        "n_test": len(test),
                        "qlike": qlike(tv, pv),
                        "qlike_nonzero_target": qlike(tv[nonzero], pv[nonzero]) if nonzero.sum() else np.nan,
                        "mae": float(np.mean(np.abs(tr - pred))),
                        "rmse": float(np.sqrt(np.mean((tr - pred) ** 2))),
                    })
    return records


def main() -> None:
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    all_records = []
    predictions = []
    for (variety, location), g in master.groupby(["target_variety", "target_location"]):
        name = f"{variety}_{location}"
        print(f"Evaluating {name} ({len(g)} obs)...", flush=True)
        all_records.extend(evaluate_series(name, g, predictions))
        pd.concat(predictions, ignore_index=True).to_csv(DATA_DIR / "ml_predictions_cafe.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame.from_records(all_records)
    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(results)} rows to {OUT_PATH}")

    summary = results.groupby(["experiment", "model", "horizon"]).agg(
        n_folds=("test_year", "count"),
        qlike=("qlike", "mean"),
        qlike_nz=("qlike_nonzero_target", "mean"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
    ).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
