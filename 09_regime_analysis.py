"""Volatility Regime and Market Stress Breakdown (RQ4 / H4).

Evaluates whether machine learning and cross-market information (E2, E3, E4)
deliver superior incremental gains during periods of high market turbulence
compared to calm/normal regimes.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dados_commodities_bahia"
QLIKE_EPS = 1e-4


def calc_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    v_act = np.maximum(actual ** 2, QLIKE_EPS)
    v_pred = np.maximum(pred ** 2, QLIKE_EPS)
    ratio = v_act / v_pred
    qlike_val = float(np.mean(ratio - np.log(ratio) - 1.0))
    return {"mae": mae, "rmse": rmse, "qlike": qlike_val}


def run_regime_analysis():
    baseline = pd.read_csv(DATA_DIR / "baseline_predictions_cafe.csv")
    ml = pd.read_csv(DATA_DIR / "ml_predictions_cafe.csv")
    rows = pd.concat([baseline, ml], ignore_index=True)
    rows["configuration"] = rows.experiment + "/" + rows.model

    valid = rows[rows.valid_prediction].copy()
    valid = valid[valid.model != "EGARCH(1,1,1)"].copy()

    records = []

    for (series_name, h), group in valid.groupby(["series", "horizon"]):
        # Find dates with common support across all configs
        piv = group.pivot(index="date", columns="configuration", values="predicted_rv").dropna()
        common_dates = piv.index
        subset = group[group.date.isin(common_dates)].copy()

        # Compute terciles of realized volatility for this series & horizon
        actuals_by_date = subset.groupby("date")["actual_rv"].first()
        q33 = actuals_by_date.quantile(0.3333)
        q66 = actuals_by_date.quantile(0.6667)

        def assign_regime(rv: float) -> str:
            if rv <= q33:
                return "1_Low_Volatility"
            elif rv <= q66:
                return "2_Medium_Volatility"
            else:
                return "3_High_Volatility"

        subset["regime"] = subset["actual_rv"].apply(assign_regime)

        # Also assign specific event windows
        subset["date_dt"] = pd.to_datetime(subset["date"])
        
        # Breakdown 1: By Volatility Tercile
        for (regime, exp, model), g_cfg in subset.groupby(["regime", "experiment", "model"]):
            act = g_cfg["actual_rv"].to_numpy()
            pred = g_cfg["predicted_rv"].to_numpy()
            m = calc_metrics(act, pred)
            records.append({
                "series": series_name,
                "horizon": int(h),
                "breakdown_type": "volatility_tercile",
                "regime_or_period": regime,
                "experiment": exp,
                "model": model,
                "n_obs": len(g_cfg),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "qlike": m["qlike"],
            })

        # Breakdown 2: Key Historical Macro/Agri Periods
        event_periods = {
            "Pre-2020": subset["date_dt"] < "2020-01-01",
            "Covid-19 Shock (2020)": (subset["date_dt"] >= "2020-01-01") & (subset["date_dt"] <= "2020-12-31"),
            "Frost & Supply Crisis (2021-2022)": (subset["date_dt"] >= "2021-01-01") & (subset["date_dt"] <= "2022-12-31"),
            "Post-2023 Normalization": subset["date_dt"] >= "2023-01-01",
        }

        for period_name, mask in event_periods.items():
            sub_period = subset[mask]
            if len(sub_period) < 20:
                continue
            for (exp, model), g_cfg in sub_period.groupby(["experiment", "model"]):
                act = g_cfg["actual_rv"].to_numpy()
                pred = g_cfg["predicted_rv"].to_numpy()
                m = calc_metrics(act, pred)
                records.append({
                    "series": series_name,
                    "horizon": int(h),
                    "breakdown_type": "event_period",
                    "regime_or_period": period_name,
                    "experiment": exp,
                    "model": model,
                    "n_obs": len(g_cfg),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "qlike": m["qlike"],
                })

    df_out = pd.DataFrame(records)
    
    # Calculate incremental gains relative to E1_own within each regime & breakdown
    own = df_out[df_out.experiment == "E1_own"][["series", "horizon", "breakdown_type", "regime_or_period", "model", "mae", "rmse", "qlike"]]
    merged = df_out.merge(
        own, on=["series", "horizon", "breakdown_type", "regime_or_period", "model"],
        suffixes=("", "_own"), how="left"
    )
    merged["mae_gain_pct"] = 100 * (1 - merged["mae"] / merged["mae_own"])
    merged["rmse_gain_pct"] = 100 * (1 - merged["rmse"] / merged["rmse_own"])
    merged["qlike_gain_pct"] = 100 * (1 - merged["qlike"] / merged["qlike_own"])

    out_file = DATA_DIR / "regime_analysis_results.csv"
    merged.to_csv(out_file, index=False)
    print(f"Saved {len(merged)} regime breakdown records to {out_file}")
    return merged


if __name__ == "__main__":
    run_regime_analysis()

