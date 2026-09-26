"""E0 econometric baselines (section 14/15 of the plan): GARCH(1,1) and
HAR-RV, evaluated walk-forward (expanding window, refit annually -- section
17) against the forward realized-volatility targets built by
03_build_master_dataset.py.

For each of the 3 café target series and each horizon h in {1, 5, 20}:
  - HAR-RV: OLS of target_rv_h on [|return_1d|, hist_vol_5, hist_vol_20]
    (the classic Corsi 2009 daily/weekly/monthly decomposition, adapted to
    this dataset's own trailing-vol features), refit on each fold's
    training window only.
  - GARCH(1,1): fit on the training window's return_1d only (no test-period
    data used for parameter estimation -- see fixed-parameter forecasting
    note below), forecast h-day-ahead variance for every day in the test
    year, aggregate to RV_h = sqrt(sum of the h forecast variances).

Fixed-parameter forecasting (avoids two different leakage traps at once):
  arch's forecast() can only walk its variance recursion forward using
  returns it has actually been given. Fitting once per fold on the
  training-only series and then calling forecast() from its own end gives
  just ONE forecast, not one per test day. Refitting parameters at every
  test day would leak nothing but is needlessly slow AND keeps re-estimating
  on data the model is supposed to not have seen yet for that fold. The
  standard middle ground (arch's own "fixed parameter" rolling-forecast
  pattern) is used instead: estimate parameters once per fold on the
  training window, then call `.forecast(..., start=test_start)` against the
  FULL return series with those parameters held fixed -- the variance
  recursion still updates day-by-day using real historical returns (that is
  what h-step-ahead forecasting is supposed to use), but no test-period
  return ever influences the parameter estimates themselves.

Annual HAR training excludes targets ending on/after January 1 of the
test year. GARCH/EGARCH fit returns observed before that cutoff, and
forecast from origin t, matching the target's next h observed returns.

Metrics: QLIKE on variance units; MAE and RMSE on volatility units.
Individual forecasts (including failures) are saved for aligned comparisons.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from temporal_validation import annual_split, prediction_rows
from arch import arch_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
MASTER_PATH = DATA_DIR / "master_dataset_cafe.csv"
OUT_PATH = DATA_DIR / "baseline_results_cafe.csv"

HORIZONS = (1, 5, 20)
MIN_TRAIN_YEARS = 3
MIN_TRAIN_OBS = 250

warnings.filterwarnings("ignore", category=FutureWarning)


QLIKE_EPS = 1e-4  # variance-unit floor; this series has many exact-zero
# forward RV points (stale/unchanged local quotes -- see 01_audit_data.py's
# unchanged_price_pct), and QLIKE is undefined at true_var == 0 (log(0)).
# Flooring both sides is the standard fix in the realized-vol literature.


def qlike(true_var: np.ndarray, pred_var: np.ndarray) -> float:
    ratio = np.maximum(true_var, QLIKE_EPS) / np.maximum(pred_var, QLIKE_EPS)
    return float(np.mean(ratio - np.log(ratio) - 1))


def har_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "daily": df["return_1d"].abs(),
        "weekly": df["hist_vol_5"],
        "monthly": df["hist_vol_20"],
    })
    return X


def fit_har_predict(train: pd.DataFrame, test: pd.DataFrame, h: int) -> np.ndarray | None:
    y_col = f"target_rv_h{h}"
    train_rows = train.dropna(subset=["return_1d", "hist_vol_5", "hist_vol_20", y_col])
    if len(train_rows) < MIN_TRAIN_OBS:
        return None
    X_train = har_features(train_rows).to_numpy()
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    y_train = train_rows[y_col].to_numpy()
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = har_features(test).to_numpy()
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
    pred = X_test @ beta
    return np.clip(pred, a_min=1e-6, a_max=None)


EGARCH_SIMULATIONS = 1000  # EGARCH has no closed-form multi-step forecast;
# arch falls back to Monte Carlo simulation for horizon > 1. Kept modest so a
# full walk-forward grid (3 series x 3 horizons x ~20 annual folds) finishes
# in reasonable time -- this is a variance-reduction/speed tradeoff, not a
# correctness one (see conversation notes: GARCH uses the closed-form path
# instead and needs no such tradeoff).


def aggregate_variance(row):
    values = np.asarray(row, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        return np.nan
    return float(np.sqrt(values.sum()))


def fit_vol_model_predict(full: pd.DataFrame, train_end_idx: int, test_idx: np.ndarray, h: int,
                           vol: str, **vol_kwargs) -> np.ndarray | None:
    """Shared fixed-parameter rolling-forecast routine for GARCH and EGARCH
    (see module docstring for why parameters are estimated once per fold on
    the training window, then held fixed while the variance recursion is
    walked forward day-by-day over the full return series)."""
    train_returns = full["return_1d"].iloc[:train_end_idx].dropna()
    if len(train_returns) < MIN_TRAIN_OBS:
        return None
    am_train = arch_model(train_returns, mean="Zero", vol=vol, dist="normal", **vol_kwargs)
    try:
        res = am_train.fit(disp="off", show_warning=False)
    except Exception:
        return None

    full_returns = full["return_1d"].iloc[:int(test_idx.max()) + 1].fillna(0.0)
    am_full = arch_model(full_returns, mean="Zero", vol=vol, dist="normal", **vol_kwargs)
    fixed = am_full.fix(res.params)

    first_origin = int(test_idx.min())
    forecast_kwargs = dict(horizon=h, start=first_origin, reindex=False)
    if vol == "EGARCH" and h > 1:
        forecast_kwargs.update(method="simulation", simulations=EGARCH_SIMULATIONS,
                               rng=np.random.default_rng(42).standard_normal)
    try:
        fc = fixed.forecast(**forecast_kwargs)
    except Exception:
        return None
    variances = fc.variance  # index = origin, columns h.1..h.h

    preds = np.full(len(test_idx), np.nan)
    for i, t in enumerate(test_idx):
        origin = t
        if origin not in variances.index:
            continue
        row = variances.loc[origin]
        preds[i] = aggregate_variance(row)

    # EGARCH's multi-step forecast has no closed form and falls back to
    # simulating the log-variance recursion forward; whenever the fitted
    # persistence is close to (a near-)unit root -- routine for daily vol of
    # a single illiquid praca -- a handful of simulated paths can wander far
    # in log-space before being exponentiated, blowing up the across-path
    # MEAN that arch reports even though the process is technically still
    # stationary (confirmed by inspecting raw simulation draws while
    # debugging this: same fold, same persistence, most origins are fine,
    # a few explode by 3-4 orders of magnitude). A closed-form GARCH forecast
    # never does this. Rather than let one degenerate origin corrupt an
    # entire fold's average error, drop forecasts that are wildly
    # implausible relative to what the training window itself ever
    # exhibited -- the same spirit as excluding a failed fit outright.
    train_daily_std = train_returns.std()
    if train_daily_std > 0 and np.isfinite(train_daily_std):
        plausibility_ceiling = 50 * train_daily_std * np.sqrt(h)
        preds[preds > plausibility_ceiling] = np.nan
    return preds


def fit_garch_predict(full: pd.DataFrame, train_end_idx: int, test_idx: np.ndarray, h: int) -> np.ndarray | None:
    return fit_vol_model_predict(full, train_end_idx, test_idx, h, vol="GARCH", p=1, q=1)


def fit_egarch_predict(full: pd.DataFrame, train_end_idx: int, test_idx: np.ndarray, h: int) -> np.ndarray | None:
    return fit_vol_model_predict(full, train_end_idx, test_idx, h, vol="EGARCH", p=1, o=1, q=1)


def evaluate_series(name: str, df: pd.DataFrame, predictions=None) -> list[dict]:
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    years = sorted(df["year"].unique())
    start_year = years[0]
    test_years = [y for y in years if y >= start_year + MIN_TRAIN_YEARS and y <= years[-1]]

    records = []
    for h in HORIZONS:
        y_col = f"target_rv_h{h}"
        for test_year in test_years:
            train, test = annual_split(df, test_year, h)
            return_train = df[df["year"] < test_year]
            if test.empty or len(train) < MIN_TRAIN_OBS:
                continue

            har_pred = fit_har_predict(train, test, h)
            garch_pred = fit_garch_predict(df, train_end_idx=return_train.index.max() + 1,
                                            test_idx=test.index.to_numpy(), h=h)
            egarch_pred = fit_egarch_predict(df, train_end_idx=return_train.index.max() + 1,
                                              test_idx=test.index.to_numpy(), h=h)

            true_rv = test[y_col].to_numpy()
            true_var = true_rv ** 2
            # Robustness split (section 19/23): this series has many exact
            # target_rv==0 test points (stale/unchanged local quotes). QLIKE
            # penalizes under-prediction there asymmetrically hard even
            # after flooring, so report QLIKE on the non-zero subset too --
            # it isolates whether a model's QLIKE ranking is being driven by
            # its behaviour on real price moves or by the zero-return days.
            nonzero = true_rv > 0

            for model_name, pred in (
                ("HAR-RV", har_pred), ("GARCH(1,1)", garch_pred), ("EGARCH(1,1,1)", egarch_pred),
            ):
                if predictions is not None:
                    rows = prediction_rows(test, pred if pred is not None else np.full(len(test), np.nan),
                                           train, name, h, test_year, model_name, "E0")
                    if model_name != "HAR-RV":
                        rows["n_train"] = return_train.return_1d.notna().sum()
                        rows["train_last_date"] = pd.to_datetime(return_train.date).max()
                        rows["train_last_target_end"] = pd.NaT
                    predictions.append(rows)
                if pred is None:
                    continue
                mask = ~np.isnan(pred) & ~np.isnan(true_rv)
                if mask.sum() == 0:
                    continue
                pv, tv, tr = pred[mask] ** 2, true_var[mask], true_rv[mask]
                pr = pred[mask]
                nz = nonzero[mask]
                records.append({
                    "series": name, "horizon": h, "model": model_name, "test_year": test_year,
                    "n_test": int(mask.sum()),
                    "qlike": qlike(tv, pv),
                    "qlike_nonzero_target": qlike(tv[nz], pv[nz]) if nz.sum() > 0 else np.nan,
                    "mae": float(np.mean(np.abs(tr - pr))),
                    "rmse": float(np.sqrt(np.mean((tr - pr) ** 2))),
                })
    return records


def main() -> None:
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    all_records = []
    predictions = []
    for (variety, location), g in master.groupby(["target_variety", "target_location"]):
        name = f"{variety}_{location}"
        print(f"Evaluating {name} ({len(g)} obs)...")
        all_records.extend(evaluate_series(name, g, predictions))
        pd.concat(predictions, ignore_index=True).to_csv(DATA_DIR / "baseline_predictions_cafe.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame.from_records(all_records)
    results.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(results)} rows to {OUT_PATH}")

    summary = results.groupby(["series", "horizon", "model"]).agg(
        n_folds=("test_year", "nunique"),
        qlike=("qlike", "mean"),
        qlike_nz=("qlike_nonzero_target", "mean"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
    ).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
