"""Statistical Predictive Superiority Tests (Diebold-Mariano & Harvey-Leybourne-Newbold).

Evaluates whether out-of-sample forecast accuracy differences are statistically
significant across information sets (E1 vs E2, E3, E4) and model families
(ML vs HAR-RV / GARCH).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent / "dados_commodities_bahia"
KEY = ["series", "horizon", "date"]
QLIKE_EPS = 1e-4


def dm_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "SE",
    y_true: np.ndarray | None = None,
    p1: np.ndarray | None = None,
    p2: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Calculate Diebold-Mariano test statistic with Harvey-Leybourne-Newbold (1997) correction.

    Returns:
        (mean_loss_diff, dm_stat_hln, p_value)
        mean_loss_diff > 0 means model 2 has smaller loss than model 1 (model 2 is better).
    """
    if loss == "SE":
        d = e1 ** 2 - e2 ** 2
    elif loss == "AE":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "QLIKE":
        assert y_true is not None and p1 is not None and p2 is not None
        v_true = np.maximum(y_true ** 2, QLIKE_EPS)
        v1 = np.maximum(p1 ** 2, QLIKE_EPS)
        v2 = np.maximum(p2 ** 2, QLIKE_EPS)
        l1 = v_true / v1 - np.log(v_true / v1) - 1.0
        l2 = v_true / v2 - np.log(v_true / v2) - 1.0
        d = l1 - l2
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    n = len(d)
    if n < 5:
        return np.nan, np.nan, np.nan

    d_mean = np.mean(d)
    gamma_0 = np.var(d, ddof=0)

    # Autocovariance up to lag h-1
    auto_cov = 0.0
    for k in range(1, h):
        if n - k > 0:
            gamma_k = np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
            auto_cov += 2.0 * (1.0 - k / h) * gamma_k

    var_d = (gamma_0 + auto_cov) / n
    if var_d <= 1e-12:
        return float(d_mean), 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)

    # HLN small-sample correction factor
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm_stat * hln_factor

    # Two-tailed p-value using Student's t with (n - 1) dof
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln), df=n - 1))

    return float(d_mean), float(dm_hln), float(p_val)


def run_statistical_tests():
    baseline = pd.read_csv(DATA_DIR / "baseline_predictions_cafe.csv")
    ml = pd.read_csv(DATA_DIR / "ml_predictions_cafe.csv")
    rows = pd.concat([baseline, ml], ignore_index=True)
    rows["configuration"] = rows.experiment + "/" + rows.model

    valid = rows[rows.valid_prediction].copy()
    valid["error"] = valid["actual_rv"] - valid["predicted_rv"]

    # Filter to primary models (exclude EGARCH due to convergence limits)
    primary = valid[valid.model != "EGARCH(1,1,1)"].copy()

    records = []

    # Iterate over each series and horizon
    for (series_name, h), group in primary.groupby(["series", "horizon"]):
        # Pivot predictions and errors by date for identical common support
        piv_pred = group.pivot(index="date", columns="configuration", values="predicted_rv").dropna()
        piv_act = group.pivot(index="date", columns="configuration", values="actual_rv").dropna()
        piv_err = group.pivot(index="date", columns="configuration", values="error").dropna()

        common_dates = piv_pred.index
        if len(common_dates) < 30:
            continue

        y_true = piv_act.iloc[:, 0].to_numpy()

        # Test 1: Incremental Gains within each ML model family (E1 vs E2, E3, E4)
        for model in ["RandomForest", "LightGBM", "XGBoost"]:
            m_e1 = f"E1_own/{model}"
            if m_e1 not in piv_pred.columns:
                continue

            for exp in ["E2_cross_commodity", "E3_financial", "E4_full"]:
                m_exp = f"{exp}/{model}"
                if m_exp not in piv_pred.columns:
                    continue

                e_base = piv_err[m_e1].to_numpy()
                e_comp = piv_err[m_exp].to_numpy()
                p_base = piv_pred[m_e1].to_numpy()
                p_comp = piv_pred[m_exp].to_numpy()

                for loss_name in ["SE", "AE", "QLIKE"]:
                    d_mean, stat, pval = dm_test(
                        e_base, e_comp, h=int(h), loss=loss_name,
                        y_true=y_true, p1=p_base, p2=p_comp,
                    )
                    records.append({
                        "series": series_name,
                        "horizon": int(h),
                        "test_type": "incremental_features",
                        "model_1": m_e1,
                        "model_2": m_exp,
                        "loss": loss_name,
                        "n_obs": len(common_dates),
                        "mean_loss_diff (m1-m2)": d_mean,
                        "stat_hln": stat,
                        "p_value": pval,
                        "significant_5pct": pval < 0.05 if not np.isnan(pval) else False,
                        "m2_better": d_mean > 0,
                    })

        # Test 2: ML vs Econometric Baselines (HAR-RV and GARCH)
        for base_model in ["E0/HAR-RV", "E0/GARCH(1,1)"]:
            if base_model not in piv_pred.columns:
                continue

            e_base = piv_err[base_model].to_numpy()
            p_base = piv_pred[base_model].to_numpy()

            for ml_cfg in [c for c in piv_pred.columns if c.startswith(("E1_", "E2_", "E3_", "E4_"))]:
                e_comp = piv_err[ml_cfg].to_numpy()
                p_comp = piv_pred[ml_cfg].to_numpy()

                for loss_name in ["SE", "AE", "QLIKE"]:
                    d_mean, stat, pval = dm_test(
                        e_base, e_comp, h=int(h), loss=loss_name,
                        y_true=y_true, p1=p_base, p2=p_comp,
                    )
                    records.append({
                        "series": series_name,
                        "horizon": int(h),
                        "test_type": "ml_vs_baseline",
                        "model_1": base_model,
                        "model_2": ml_cfg,
                        "loss": loss_name,
                        "n_obs": len(common_dates),
                        "mean_loss_diff (m1-m2)": d_mean,
                        "stat_hln": stat,
                        "p_value": pval,
                        "significant_5pct": pval < 0.05 if not np.isnan(pval) else False,
                        "m2_better": d_mean > 0,
                    })

    df_out = pd.DataFrame(records)
    out_file = DATA_DIR / "statistical_tests_results.csv"
    df_out.to_csv(out_file, index=False)
    print(f"Saved {len(df_out)} statistical test records to {out_file}")
    return df_out


if __name__ == "__main__":
    run_statistical_tests()

