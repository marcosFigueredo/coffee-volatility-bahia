"""Produce an actual forward-looking volatility forecast (not backtest) for
each café target series: train RandomForest on ALL historical rows that
have a known target_rv_h{h} label, then predict from the single most recent
row (whose features are known but whose target_rv_h{h} is NaN precisely
because it needs future returns nobody has observed yet).

IMPORTANT CAVEAT printed with the results: SEAGRI-BA's local feed lags real
time substantially (last observation 2026-01-21 as of this run, ~7.5 months
behind). "Forecast" here means "volatility over the h trading days
following the last available local quote," which is NOT the same as a
live forecast from today -- most of that window is itself already in the
past relative to when this script runs. This is a data-freshness limitation
of the source, not a modelling choice.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
MASTER_PATH = DATA_DIR / "master_dataset_cafe.csv"
OUT_PATH = DATA_DIR / "forecast_current_cafe.csv"

HORIZONS = (1, 5, 20)
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
# E2 (own + cross-commodity): matched E1 closely in backtesting and was the
# best- or near-best-QLIKE config for RandomForest at every horizon (see
# ml_results_cafe.csv) -- E3/E4's financial features hurt at h=20, so they
# are deliberately left out of this production forecast.
FEATURE_COLS = OWN_HISTORY_COLS + CROSS_COMMODITY_COLS


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for k in range(1, 6):
        df[f"return_lag{k}"] = df["return_1d"].shift(k)
    return df


def forecast_series(name: str, df: pd.DataFrame) -> list[dict]:
    df = add_lag_features(df).reset_index(drop=True)
    last_row = df.iloc[[-1]]
    last_date = last_row["date"].iloc[0]

    records = []
    for h in HORIZONS:
        y_col = f"target_rv_h{h}"
        train = df.dropna(subset=[y_col, "return_1d"])
        X_train, y_train = train[FEATURE_COLS], train[y_col].to_numpy()
        X_last = last_row[FEATURE_COLS]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_last_imp = imputer.transform(X_last)

        model = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=1,
        )
        model.fit(X_train_imp, y_train)
        pred = float(model.predict(X_last_imp)[0])

        tree_preds = np.array([t.predict(X_last_imp)[0] for t in model.estimators_])
        records.append({
            "series": name, "as_of_date": last_date, "horizon_days": h,
            "forecast_rv": round(pred, 4),
            "forecast_rv_p10": round(float(np.percentile(tree_preds, 10)), 4),
            "forecast_rv_p90": round(float(np.percentile(tree_preds, 90)), 4),
            "recent_hist_vol_5": round(float(last_row["hist_vol_5"].iloc[0]), 4),
            "recent_hist_vol_20": round(float(last_row["hist_vol_20"].iloc[0]), 4),
            "n_train": len(train),
        })
    return records


def main() -> None:
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    all_records = []
    for (variety, location), g in master.groupby(["target_variety", "target_location"]):
        name = f"{variety}_{location}"
        print(f"Forecasting {name}...", flush=True)
        all_records.extend(forecast_series(name, g))

    out = pd.DataFrame.from_records(all_records)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nWrote {len(out)} rows to {OUT_PATH}")
    print(
        "\nCAVEAT: SEAGRI-BA's local feed lags real time -- 'as_of_date' below is the "
        "last available local quote, not today. The forecast covers the h trading days "
        "AFTER that date, which for the shorter horizons is itself already in the past "
        "relative to when this script runs. This is a source-freshness limitation, not "
        "a modelling artifact.\n"
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
