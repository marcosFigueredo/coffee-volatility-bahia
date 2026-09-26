"""Model Explainability & Feature Importance (RQ5 & RQ6).

Quantifies the contribution of own-history, cross-commodity (ICE Coffee, CEPEA,
soy, cocoa, cotton) and financial variables (USD/BRL, VIX, Brent, Selic) in
forecasting coffee volatility for Arabica vs Conilon across horizons.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb

import shap  # required: a missing package must fail loudly, not write zeros

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
MASTER_PATH = DATA_DIR / "master_dataset_cafe.csv"

HORIZONS = (1, 5, 20)

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
    "E2_cross_commodity": OWN_HISTORY_COLS + CROSS_COMMODITY_COLS,
    "E4_full": OWN_HISTORY_COLS + CROSS_COMMODITY_COLS + FINANCIAL_COLS,
}


def get_feature_group(col_name: str) -> str:
    if col_name in OWN_HISTORY_COLS:
        return "1_Own_History"
    elif col_name in CROSS_COMMODITY_COLS:
        return "2_Cross_Commodity"
    elif col_name in FINANCIAL_COLS:
        return "3_Financial_Macro"
    return "4_Other"


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for k in range(1, 6):
        df[f"return_lag{k}"] = df["return_1d"].shift(k)
    return df


def run_explainability():
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    records = []

    for (variety, location), g in master.groupby(["target_variety", "target_location"]):
        series_name = f"{variety}_{location}"
        print(f"Explaining {series_name}...", flush=True)
        df = add_lag_features(g).reset_index(drop=True)

        for h in HORIZONS:
            y_col = f"target_rv_h{h}"
            valid_df = df.dropna(subset=[y_col, "return_1d"]).copy()
            if len(valid_df) < 500:
                continue

            for exp_name, feat_cols in EXPERIMENTS.items():
                X = valid_df[feat_cols]
                y = valid_df[y_col].to_numpy()

                imputer = SimpleImputer(strategy="median")
                X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feat_cols)

                # 1. Random Forest Feature Importance
                rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1)
                rf.fit(X_imp, y)
                rf_imp = rf.feature_importances_

                # 2. LightGBM
                lgbm = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, min_child_samples=10, random_state=42, verbosity=-1, n_jobs=1)
                lgbm.fit(X_imp, y)
                lgbm_imp = lgbm.feature_importances_ / max(1, lgbm.feature_importances_.sum())

                # 3. XGBoost
                xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=1)
                xgb_model.fit(X_imp, y)
                xgb_imp = xgb_model.feature_importances_

                # SHAP computation on LightGBM (fast and robust)
                explainer = shap.TreeExplainer(lgbm)
                shap_vals = explainer.shap_values(X_imp)
                shap_mean_abs = np.mean(np.abs(shap_vals), axis=0)
                if not shap_mean_abs.sum() > 0:
                    raise RuntimeError(f"All-zero SHAP values for {series_name} h={h} {exp_name}")
                shap_mean_abs = shap_mean_abs / shap_mean_abs.sum()

                for i, col in enumerate(feat_cols):
                    records.append({
                        "series": series_name,
                        "variety": variety,
                        "location": location,
                        "horizon": int(h),
                        "experiment": exp_name,
                        "feature": col,
                        "feature_group": get_feature_group(col),
                        "rf_importance": float(rf_imp[i]),
                        "lgbm_importance": float(lgbm_imp[i]),
                        "xgb_importance": float(xgb_imp[i]),
                        "shap_importance": float(shap_mean_abs[i]),
                        "mean_importance": float(np.mean([rf_imp[i], lgbm_imp[i], xgb_imp[i]])),
                    })

    df_out = pd.DataFrame(records)
    out_file = DATA_DIR / "shap_importance_summary.csv"
    df_out.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_out)} feature importance rows to {out_file}")

    # Grouped summary by feature category: sum within each series (shares add to 1
    # per fit), then average across series so varieties with more series are not inflated
    grp_summary = (
        df_out.groupby(["series", "variety", "horizon", "experiment", "feature_group"])
        .agg(
            total_mean_importance=("mean_importance", "sum"),
            total_shap_importance=("shap_importance", "sum"),
        )
        .groupby(["variety", "horizon", "experiment", "feature_group"])
        .mean()
        .reset_index()
    )
    grp_file = DATA_DIR / "feature_group_importance_summary.csv"
    grp_summary.to_csv(grp_file, index=False, encoding="utf-8-sig")
    print(f"Saved grouped feature importance to {grp_file}")
    return df_out


if __name__ == "__main__":
    run_explainability()

