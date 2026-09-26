"""Robustness checks requested in the pre-submission review.

A. QLIKE sensitivity to the variance floor and to zero-variance targets.
B. Incremental gains computed on pooled losses (ratio of means, not mean of ratios).
C. Multiple-testing correction (Holm, Benjamini-Hochberg) for the DM-HLN tests,
   plus DM-HLN tests of QLIKE restricted to non-zero targets.
D. Ex-ante volatility regimes (terciles of trailing hist_vol_20 with thresholds
   estimated only on data before each test year) and HAC tests of whether the
   incremental gain differs between high and low regimes.
E. Similarity between the two Arabica series (VDC vs LEM).

All inputs are the saved out-of-sample predictions; no model is re-estimated.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
DATA = BASE / "dados_commodities_bahia"
KEY = ["series", "horizon", "date"]
EPS_GRID = [1e-4, 1e-2, 1e-1, 1.0]
BLOCKS = ["E2_cross_commodity", "E3_financial", "E4_full"]
ML_MODELS = ["RandomForest", "LightGBM", "XGBoost"]

_spec = importlib.util.spec_from_file_location("dm", BASE / "08_statistical_tests.py")
_dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dm)


def qlike(actual, pred, eps):
    va = np.maximum(np.asarray(actual) ** 2, eps)
    vp = np.maximum(np.asarray(pred) ** 2, eps)
    r = va / vp
    return r - np.log(r) - 1.0


def load_common() -> pd.DataFrame:
    rows = pd.concat([pd.read_csv(DATA / "baseline_predictions_cafe.csv"),
                      pd.read_csv(DATA / "ml_predictions_cafe.csv")], ignore_index=True)
    rows = rows[rows.valid_prediction & (rows.model != "EGARCH(1,1,1)")].copy()
    rows["configuration"] = rows.experiment + "/" + rows.model
    n_cfg = rows.configuration.nunique()
    counts = rows.groupby(KEY).configuration.nunique()
    keys = counts[counts == n_cfg].reset_index()[KEY]
    common = rows.merge(keys, on=KEY, validate="many_to_one")
    common["ae"] = (common.actual_rv - common.predicted_rv).abs()
    common["se"] = (common.actual_rv - common.predicted_rv) ** 2
    return common


def pooled_metrics(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    df = df.assign(q=qlike(df.actual_rv, df.predicted_rv, eps))
    out = df.groupby(["horizon", "experiment", "model"]).agg(
        n=("date", "size"), qlike=("q", "mean"), mae=("ae", "mean"), mse=("se", "mean")).reset_index()
    out["rmse"] = np.sqrt(out.pop("mse"))
    return out


# ---------------------------------------------------------------- A
def qlike_sensitivity(common: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for eps in EPS_GRID:
        frames.append(pooled_metrics(common, eps).assign(scenario=f"all_eps={eps:g}"))
    nz = common[common.actual_rv > 0]
    frames.append(pooled_metrics(nz, 1e-4).assign(scenario="nonzero_target"))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(DATA / "robust_qlike_sensitivity.csv", index=False)

    zero = (common.drop_duplicates(KEY).assign(zero=lambda d: d.actual_rv == 0)
            .groupby(["series", "horizon"]).zero.mean().unstack())
    print("\n[A] Share of zero-variance targets (actual_rv == 0):")
    print((100 * zero).round(1).to_string())

    print("\n[A] Pooled winners by scenario (lowest loss):")
    for (scen, h), g in res.groupby(["scenario", "horizon"], sort=False):
        g = g.assign(cfg=g.experiment + "/" + g.model)
        best = {m: g.loc[g[m].idxmin()] for m in ["qlike", "mae", "rmse"]}
        ml = g[g.model.isin(ML_MODELS)]
        best_ml = ml.loc[ml.qlike.idxmin()]
        garch = g[g.model == "GARCH(1,1)"].iloc[0]
        har = g[g.model == "HAR-RV"].iloc[0]
        print(f"  {scen:16s} h={h:2d} n={int(g.n.iloc[0]):5d} | QLIKE: {best['qlike'].cfg} {best['qlike'].qlike:.3f} "
              f"(GARCH {garch.qlike:.3f}, HAR {har.qlike:.3f}, bestML {best_ml.cfg} {best_ml.qlike:.3f}) | "
              f"MAE: {best['mae'].cfg} {best['mae'].mae:.3f} | RMSE: {best['rmse'].cfg} {best['rmse'].rmse:.3f}")
    return res


# ---------------------------------------------------------------- B
def pooled_gains(sens: pd.DataFrame) -> pd.DataFrame:
    sub = sens[sens.scenario.isin(["all_eps=0.0001", "nonzero_target"]) & sens.model.isin(ML_MODELS)]
    base = sub[sub.experiment == "E1_own"][["scenario", "horizon", "model", "qlike", "mae", "rmse"]]
    g = sub[sub.experiment.isin(BLOCKS)].merge(base, on=["scenario", "horizon", "model"], suffixes=("", "_e1"))
    for m in ["qlike", "mae", "rmse"]:
        g[f"{m}_gain_pct"] = 100 * (1 - g[m] / g[f"{m}_e1"])
    g.to_csv(DATA / "robust_pooled_gains.csv", index=False)
    print("\n[B] Gains over E1 on pooled losses (%), by scenario/model/block (rows) and horizon:")
    for scen in ["all_eps=0.0001", "nonzero_target"]:
        t = g[g.scenario == scen].pivot_table(index=["model", "experiment"], columns="horizon",
                                              values=["qlike_gain_pct", "mae_gain_pct", "rmse_gain_pct"])
        print(f"  -- {scen}")
        print(t.round(2).to_string())
    return g


# ---------------------------------------------------------------- C
def holm(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p); rej = np.zeros(n, bool)
    for i, idx in enumerate(order):
        if p[idx] <= 0.05 / (n - i):
            rej[idx] = True
        else:
            break
    return rej


def bh(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p); rej = np.zeros(n, bool)
    ok = p[order] <= 0.05 * np.arange(1, n + 1) / n
    if ok.any():
        rej[order[: np.max(np.where(ok)) + 1]] = True
    return rej


def multiple_testing(common: pd.DataFrame) -> None:
    dm = pd.read_csv(DATA / "statistical_tests_results.csv")
    p = dm.p_value.fillna(1.0)
    dm["holm_reject"] = holm(p)
    dm["bh_reject"] = bh(p)
    dm.to_csv(DATA / "statistical_tests_results_adjusted.csv", index=False)

    def comp(row):
        if row.test_type == "incremental_features":
            return "E1 vs E2/E3/E4"
        return "ML vs " + row.model_1.split("/")[1]

    dm["comparison"] = dm.apply(comp, axis=1)
    fav = lambda col: dm[col] & dm.m2_better
    tab = dm.assign(sig=dm.significant_5pct, fav=fav("significant_5pct"),
                    holm_sig=dm.holm_reject, holm_fav=fav("holm_reject"),
                    bh_sig=dm.bh_reject, bh_fav=fav("bh_reject")).groupby(["comparison", "loss"]).agg(
        n=("p_value", "size"), sig=("sig", "sum"), fav=("fav", "sum"), bh_sig=("bh_sig", "sum"),
        bh_fav=("bh_fav", "sum"), holm_sig=("holm_sig", "sum"), holm_fav=("holm_fav", "sum")).reset_index()
    tab.to_csv(DATA / "robust_dm_multiple_testing.csv", index=False)
    print("\n[C] DM-HLN counts: raw 5%, BH (FDR 5%), Holm (FWER 5%), over", len(dm), "tests:")
    print(tab.to_string(index=False))

    # DM-HLN on QLIKE restricted to non-zero targets
    recs = []
    nz = common[common.actual_rv > 0]
    for (s, h), g in nz.groupby(["series", "horizon"]):
        pv = g.pivot(index="date", columns="configuration", values="predicted_rv").dropna()
        y = g.groupby("date").actual_rv.first().loc[pv.index].to_numpy()
        pairs = [(f"E1_own/{m}", f"{b}/{m}", "E1 vs E2/E3/E4") for m in ML_MODELS for b in BLOCKS]
        pairs += [(base, c, "ML vs " + base.split("/")[1]) for base in ["E0/HAR-RV", "E0/GARCH(1,1)"]
                  for c in pv.columns if c.startswith(("E1_", "E2_", "E3_", "E4_"))]
        for m1, m2, kind in pairs:
            p1, p2 = pv[m1].to_numpy(), pv[m2].to_numpy()
            d, st, pval = _dm.dm_test(y - p1, y - p2, h=int(h), loss="QLIKE", y_true=y, p1=p1, p2=p2)
            recs.append(dict(series=s, horizon=h, comparison=kind, model_1=m1, model_2=m2,
                             n_obs=len(y), mean_loss_diff=d, stat_hln=st, p_value=pval))
    q = pd.DataFrame(recs)
    q["sig"] = q.p_value < 0.05
    q["fav"] = q.sig & (q.mean_loss_diff > 0)
    q["unfav"] = q.sig & (q.mean_loss_diff < 0)
    q.to_csv(DATA / "robust_dm_qlike_nonzero.csv", index=False)
    print("\n[C] DM-HLN QLIKE on non-zero targets (fav = significant in favour of model_2):")
    print(q.groupby("comparison")[["sig", "fav", "unfav"]].sum().assign(n=q.groupby("comparison").size()).to_string())
    inc = q[q.comparison == "E1 vs E2/E3/E4"].assign(model=lambda d: d.model_2.str.split("/").str[1],
                                                       block=lambda d: d.model_2.str.split("/").str[0])
    print(inc.groupby(["model", "block"])[["fav", "unfav"]].sum().to_string())


# ---------------------------------------------------------------- D
def newey_west_ols(y, X, lags):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    n = len(y)
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = xu[L:].T @ xu[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - X.shape[1]))
    return beta, t, p


def ex_ante_regimes(common: pd.DataFrame) -> None:
    master = pd.read_csv(DATA / "master_dataset_cafe.csv", encoding="utf-8-sig")
    master["series"] = master.target_variety + "_" + master.target_location
    master["date_dt"] = pd.to_datetime(master.date)
    hv = master[["series", "date", "hist_vol_20"]]
    df = common.merge(hv, on=["series", "date"], how="left", validate="many_to_one")

    # thresholds from trailing hist_vol_20 strictly before each test year
    thr = {}
    for (s, y), _ in df.groupby(["series", "test_year"]):
        past = master[(master.series == s) & (master.date_dt < f"{y}-01-01")].hist_vol_20.dropna()
        thr[(s, y)] = (past.quantile(1 / 3), past.quantile(2 / 3))
    q = df.apply(lambda r: thr[(r.series, r.test_year)], axis=1, result_type="expand")
    df["q33"], df["q66"] = q[0], q[1]
    df["regime"] = np.select([df.hist_vol_20 <= df.q33, df.hist_vol_20 <= df.q66],
                             ["1_Low", "2_Medium"], "3_High")
    df.loc[df.hist_vol_20.isna(), "regime"] = np.nan
    df["q_eps"] = qlike(df.actual_rv, df.predicted_rv, 1e-4)
    df["q_nz"] = np.where(df.actual_rv > 0, df.q_eps, np.nan)

    share = df.drop_duplicates(KEY).groupby(["horizon", "regime"]).size().unstack()
    print("\n[D] Ex-ante regime sizes (origins):")
    print(share.to_string())

    perf = df.groupby(["regime", "experiment", "model"]).agg(
        qlike=("q_eps", "mean"), qlike_nz=("q_nz", "mean"), mae=("ae", "mean"), mse=("se", "mean")).reset_index()
    perf["rmse"] = np.sqrt(perf.pop("mse"))
    perf.to_csv(DATA / "robust_exante_regime_performance.csv", index=False)
    print("\n[D] Pooled performance by ex-ante regime (baselines and best ML by QLIKE):")
    for reg, g in perf.groupby("regime"):
        g = g.assign(cfg=g.experiment + "/" + g.model)
        ml = g[g.model.isin(ML_MODELS)]
        print(f"  {reg}: GARCH q={g[g.model=='GARCH(1,1)'].qlike.iloc[0]:.3f} qnz={g[g.model=='GARCH(1,1)'].qlike_nz.iloc[0]:.3f} "
              f"mae={g[g.model=='GARCH(1,1)'].mae.iloc[0]:.3f} | HAR q={g[g.model=='HAR-RV'].qlike.iloc[0]:.3f} "
              f"qnz={g[g.model=='HAR-RV'].qlike_nz.iloc[0]:.3f} mae={g[g.model=='HAR-RV'].mae.iloc[0]:.3f} | "
              f"bestML(q) {ml.loc[ml.qlike.idxmin()].cfg} {ml.qlike.min():.3f} | bestML(qnz) {ml.loc[ml.qlike_nz.idxmin()].cfg} "
              f"{ml.qlike_nz.min():.3f} | bestML(mae) {ml.loc[ml.mae.idxmin()].cfg} {ml.mae.min():.3f}")

    # pooled gains by regime (ratio of pooled losses)
    base = perf[perf.experiment == "E1_own"][["regime", "model", "qlike", "qlike_nz", "mae", "rmse"]]
    g = perf[perf.experiment.isin(BLOCKS)].merge(base, on=["regime", "model"], suffixes=("", "_e1"))
    for m in ["qlike", "qlike_nz", "mae", "rmse"]:
        g[f"{m}_gain_pct"] = 100 * (1 - g[m] / g[f"{m}_e1"])
    g.to_csv(DATA / "robust_exante_regime_gains.csv", index=False)
    print("\n[D] Gains over E1 by ex-ante regime (%, pooled losses):")
    print(g.pivot_table(index=["model", "experiment"], columns="regime",
                        values=["qlike_nz_gain_pct", "mae_gain_pct", "rmse_gain_pct"]).round(2).to_string())

    # HAC test: does the loss differential (E1 - Ek) differ between High and Low regimes?
    recs = []
    d0 = df.dropna(subset=["regime"])
    for (s, h), g in d0.groupby(["series", "horizon"]):
        wide = {}
        for col in ["ae", "se", "q_nz"]:
            wide[col] = g.pivot(index="date", columns="configuration", values=col)
        reg = g.drop_duplicates("date").set_index("date").regime.sort_index()
        for m in ML_MODELS:
            for b in BLOCKS:
                for col, name in [("ae", "AE"), ("se", "SE"), ("q_nz", "QLIKE_nz")]:
                    d = (wide[col][f"E1_own/{m}"] - wide[col][f"{b}/{m}"]).reindex(reg.index)
                    ok = d.notna()
                    y = d[ok].to_numpy()
                    r = reg[ok]
                    X = np.column_stack([np.ones(len(y)), (r == "2_Medium").to_numpy(float),
                                         (r == "3_High").to_numpy(float)])
                    if len(y) < 50 or X[:, 2].sum() < 10:
                        continue
                    beta, t, p = newey_west_ols(y, X, lags=max(5, int(h)))
                    recs.append(dict(series=s, horizon=h, model=m, block=b, loss=name, n=len(y),
                                     gain_low=beta[0], high_minus_low=beta[2], t_high_minus_low=t[2],
                                     p_high_minus_low=p[2]))
    t = pd.DataFrame(recs)
    t["sig_pos"] = (t.p_high_minus_low < 0.05) & (t.high_minus_low > 0)
    t["sig_neg"] = (t.p_high_minus_low < 0.05) & (t.high_minus_low < 0)
    t.to_csv(DATA / "robust_exante_regime_hac_tests.csv", index=False)
    print("\n[D] HAC tests of (gain in High) - (gain in Low): counts of significant +/- at 5%")
    print(t.groupby(["model", "block", "loss"])[["sig_pos", "sig_neg"]].sum()
          .assign(n=t.groupby(["model", "block", "loss"]).size()).unstack("loss").to_string())


# ---------------------------------------------------------------- E
def arabica_similarity() -> None:
    m = pd.read_csv(DATA / "master_dataset_cafe.csv", encoding="utf-8-sig")
    a = m[m.target_variety == "arabica"]
    p = a.pivot_table(index="date", columns="target_location", values="price")
    r = a.pivot_table(index="date", columns="target_location", values="return_1d")
    v = a.pivot_table(index="date", columns="target_location", values="target_rv_h20")
    cols = list(p.columns)
    pc = p.dropna()
    rc = r.dropna()
    vc = v.dropna()
    both_move = ((rc[cols[0]] != 0) & (rc[cols[1]] != 0)).mean()
    any_move = ((rc[cols[0]] != 0) | (rc[cols[1]] != 0)).mean()
    out = dict(common_dates=len(pc), identical_price_pct=100 * (pc[cols[0]] == pc[cols[1]]).mean(),
               price_corr=pc.corr().iloc[0, 1], return_corr=rc.corr().iloc[0, 1],
               both_nonzero_return_pct=100 * both_move, any_nonzero_return_pct=100 * any_move,
               same_return_pct=100 * (np.isclose(rc[cols[0]], rc[cols[1]])).mean(),
               rv20_corr=vc.corr().iloc[0, 1],
               median_rv20_0=vc[cols[0]].median(), median_rv20_1=vc[cols[1]].median())
    pd.DataFrame([out]).to_csv(DATA / "robust_arabica_similarity.csv", index=False)
    print("\n[E] Arabica VDC vs LEM similarity:", cols)
    for k, val in out.items():
        print(f"  {k}: {val:.4f}" if isinstance(val, float) else f"  {k}: {val}")


def main():
    common = load_common()
    print("Common-support rows:", len(common), "configs:", common.configuration.nunique())
    sens = qlike_sensitivity(common)
    pooled_gains(sens)
    multiple_testing(common)
    ex_ante_regimes(common)
    arabica_similarity()


if __name__ == "__main__":
    main()
