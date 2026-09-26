"""Build the integrated master dataset for coffee (Arabica + Conilon)
volatility forecasting in Bahia, per section 25 of
projeto_volatilidade_commodities_bahia_ml.md.

Target series (local, from SEAGRI-BA, one grade picked per praca based on
which has the most observations -- see GRADE_CHOICE below):
  - arabica  / VITORIA DA CONQUISTA / Duro
  - arabica  / LUIS EDUARDO MAGALHAES / Duro
  - conilon  / EUNAPOLIS / Tipo 7

Two kinds of volatility columns are produced and must not be confused:
  - hist_vol_5/10/20: TRAILING realized vol, uses only r_{t-N+1..t}. Safe to
    use as a predictive feature at time t.
  - target_rv_h1/h5/h20: FORWARD realized vol per section 12,
    RV_{t,h} = sqrt(sum_{i=1..h} r_{t+i}^2). Uses FUTURE returns -- this is
    the label the models in section 14/15 are trained to predict, never a
    feature. h is counted in the target series' own trading days (its
    observed dates), matching the RV_{t,h} definition literally.

All exogenous (cross-commodity + financial) series are computed on their own
native calendar, then merge_asof'd (backward, tolerance 7 days) onto each
target series' own observation dates -- so a sparse local quote never gets a
same-day value from an exogenous series it couldn't have known yet, and
never silently forward-fills across long gaps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
OUT_PATH = DATA_DIR / "master_dataset_cafe.csv"

ASOF_TOLERANCE = pd.Timedelta("7D")

TARGETS = [
    {"variety": "arabica", "location": "VITORIA DA CONQUISTA", "grade": "Rio",
     "file": "seagri_ba_cafe_arabica.csv"},
    {"variety": "arabica", "location": "LUIS EDUARDO MAGALHAES", "grade": "Rio",
     "file": "seagri_ba_cafe_arabica.csv"},
    {"variety": "conilon", "location": "EUNAPOLIS", "grade": "Tipo 7",
     "file": "seagri_ba_cafe_conillon.csv"},
]
# "Duro" was the initial pick for both Arabica pracas but LEM/Duro turned out
# to have a ~2-year block (2014-04-09 to 2016-02-29, 405 rows) of nonsensical
# near-zero prices (R$0.75-1.15/sc against a median of R$340) -- a sustained
# feed corruption, not a one-off typo. "Rio" has comparable coverage with no
# such block for either praca, so it replaces "Duro" as the canonical grade.

# Known bad points, confirmed by inspecting raw neighbours (see conversation
# notes) rather than guessed from the return threshold alone:
#  - 2023-08-31 is a single-day, source-wide SEAGRI-BA feed glitch: prices
#    for ALL THREE target series roughly triple then revert the next day,
#    and 01_audit_data.py separately found one-off phantom markets
#    (VARZEDO, WANDERLEY, ENTRE RIOS) that appear on that exact date only
#    and never again -- corroborating a feed-wide corruption, not a market
#    event.
#  - Jan 1st is a national holiday (no real trading); a handful of series
#    carry a priced quote on that date that reads as a stale/placeholder
#    copy rather than a genuine observation.
# Remaining |return_1d| > 40% points are left in place (flagged via
# is_extreme_return, not dropped): some are long-gap artifacts of the
# RV_{t,h} formula spanning a silent period, others may be genuine large
# moves -- and this is a volatility study, so real large moves must not be
# quietly removed.
DROP_MONTH_DAY = {(1, 1)}  # (month, day) dropped in every year
DROP_DATES = {pd.Timestamp("2023-08-31")}  # applies to every series (feed-wide glitch)

# Series-specific single-point typos: price spikes for exactly one day
# between two identical flat neighbours (e.g. 220, 220, 2215, 220, 220 --
# a classic decimal-point slip) and is dropped only for that series.
DROP_POINTS = {
    ("arabica", "VITORIA DA CONQUISTA"): {pd.Timestamp("2010-07-28")},
}


def log_return(price: pd.Series) -> pd.Series:
    return 100 * np.log(price / price.shift(1))


def load_local_series(spec: dict) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "seagri_ba" / spec["file"], encoding="utf-8-sig")
    df = df[(df["market"].str.strip() == spec["location"]) & (df["type"].str.strip() == spec["grade"])]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date")
    df = df.groupby("date", as_index=False)["price"].last()

    series_drop_dates = DROP_DATES | DROP_POINTS.get((spec["variety"], spec["location"]), set())
    drop_mask = df["date"].isin(series_drop_dates) | df["date"].apply(
        lambda d: (d.month, d.day) in DROP_MONTH_DAY
    )
    df = df.loc[~drop_mask].reset_index(drop=True)

    df["return_1d"] = log_return(df["price"])
    return df.reset_index(drop=True)


def add_trailing_and_forward_vol(df: pd.DataFrame) -> pd.DataFrame:
    r = df["return_1d"]
    for w in (5, 10, 20):
        df[f"hist_vol_{w}"] = np.sqrt(r.pow(2).rolling(w, min_periods=w).sum())
    r2 = r.pow(2)
    for h in (1, 5, 20):
        # sum_{i=1..h} r_{t+i}^2, i.e. the NEXT h returns after t (excludes r_t itself)
        fwd_sum = r2.shift(-1).rolling(h, min_periods=h).sum()
        df[f"target_rv_h{h}"] = np.sqrt(fwd_sum.shift(-(h - 1)))
    return df


def yahoo_return_series(fname: str, colname: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "yahoo_finance" / fname, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    df = df.groupby("date", as_index=False)["close"].last()
    out = pd.DataFrame({"date": df["date"]})
    out[f"{colname}_close"] = df["close"]
    out[f"{colname}_return"] = log_return(df["close"])
    return out


def fred_series(fname: str, colname: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fred" / fname, encoding="utf-8-sig")
    value_col = next(c for c in df.columns if c != "observation_date")
    df["date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df[colname] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=["date"])[["date", colname]].sort_values("date")


def bcb_series(fname: str, colname: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "bcb" / fname, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[colname] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date"])[["date", colname]].sort_values("date")


# National (CEPEA) benchmark, variety-specific: manually downloaded via
# browser on 2026-09-09 (automatic download is blocked by a Cloudflare
# bot challenge -- see 01_audit_data.py's failed_download_records).
CEPEA_FILE_BY_VARIETY = {
    "arabica": "cepea_cafe_arabica.csv",
    "conilon": "cepea_cafe_robusta.csv",
}


def cepea_series(fname: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cepea" / fname, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price_brl"]).sort_values("date")
    df = df.groupby("date", as_index=False)["price_brl"].last()
    out = pd.DataFrame({"date": df["date"]})
    out["cepea_close"] = df["price_brl"]
    out["cepea_return"] = log_return(df["price_brl"])
    return out


def build_exogenous_panel() -> pd.DataFrame:
    """One row per calendar date the union of sources reports on; each column
    computed from its own native series before any cross-source merging."""
    coffee = yahoo_return_series("yahoo_coffee_futures_kc_f.csv", "ice_coffee_c")
    soy = yahoo_return_series("yahoo_soybean_futures_zs_f.csv", "soy")
    cotton = yahoo_return_series("yahoo_cotton_futures_ct_f.csv", "cotton")
    cocoa = yahoo_return_series("yahoo_cocoa_futures_cc_f.csv", "cocoa")
    sp500 = yahoo_return_series("yahoo_sp500_gspc.csv", "sp500")[["date", "sp500_return"]]
    ibov = yahoo_return_series("yahoo_ibovespa_bvsp.csv", "ibovespa")[["date", "ibovespa_return"]]

    for cross_df, name in ((soy, "soy"), (cotton, "cotton"), (cocoa, "cocoa")):
        cross_df[f"{name}_vol_5"] = np.sqrt(
            cross_df[f"{name}_return"].pow(2).rolling(5, min_periods=5).sum()
        )

    usd_brl = bcb_series("bcb_usd_brl_venda.csv", "usd_brl")
    usd_brl["usd_brl_return"] = log_return(usd_brl["usd_brl"])
    usd_brl["usd_brl_vol_5"] = np.sqrt(usd_brl["usd_brl_return"].pow(2).rolling(5, min_periods=5).sum())

    selic = bcb_series("bcb_selic_anualizada_252.csv", "selic")

    vix = fred_series("fred_vix_vixcls.csv", "vix")
    vix["vix_change"] = vix["vix"].diff()

    brent = pd.read_csv(DATA_DIR / "eia" / "eia_brent_daily.csv", encoding="utf-8-sig")
    price_col = next(c for c in brent.columns if c != "date")
    brent["date"] = pd.to_datetime(brent["date"], errors="coerce")
    brent = brent.rename(columns={price_col: "brent"}).dropna(subset=["date", "brent"]).sort_values("date")
    brent["brent_return"] = log_return(brent["brent"])
    brent = brent[["date", "brent", "brent_return"]]

    dollar_index = fred_series("fred_broad_us_dollar_index_dtwexbgs.csv", "dollar_index")
    dollar_index["dollar_index_return"] = log_return(dollar_index["dollar_index"])

    treasury_10y = fred_series("fred_us_treasury_10y_dgs10.csv", "us_treasury_10y")
    fed_funds = fred_series("fred_fed_funds_effective_dff.csv", "fed_funds")

    panel = coffee[["date", "ice_coffee_c_close", "ice_coffee_c_return"]]
    for other in (
        soy[["date", "soy_return", "soy_vol_5"]],
        cotton[["date", "cotton_return", "cotton_vol_5"]],
        cocoa[["date", "cocoa_return", "cocoa_vol_5"]],
        sp500, ibov,
        usd_brl[["date", "usd_brl", "usd_brl_return", "usd_brl_vol_5"]],
        selic, vix[["date", "vix", "vix_change"]], brent, dollar_index[["date", "dollar_index", "dollar_index_return"]],
        treasury_10y, fed_funds,
    ):
        panel = panel.merge(other, on="date", how="outer")

    return panel.sort_values("date").reset_index(drop=True)


def build_other_variety_panel(local_series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """date, variety, other_variety_return/vol_5 -- the OTHER variety's
    signal on/near that date. Conilon's 'other' = mean of the two Arabica
    pracas; each Arabica praca's 'other' = Conilon/Eunapolis."""
    arabica_avg = (
        pd.concat([
            local_series["arabica_VITORIA DA CONQUISTA"][["date", "return_1d"]],
            local_series["arabica_LUIS EDUARDO MAGALHAES"][["date", "return_1d"]],
        ])
        .groupby("date", as_index=False)["return_1d"].mean()
        .rename(columns={"return_1d": "other_variety_return"})
        .sort_values("date")
    )
    arabica_avg["other_variety_vol_5"] = np.sqrt(
        arabica_avg["other_variety_return"].pow(2).rolling(5, min_periods=5).sum()
    )

    conilon = local_series["conilon_EUNAPOLIS"][["date", "return_1d"]].rename(
        columns={"return_1d": "other_variety_return"}
    ).sort_values("date")
    conilon["other_variety_vol_5"] = np.sqrt(
        conilon["other_variety_return"].pow(2).rolling(5, min_periods=5).sum()
    )
    return {"arabica": conilon, "conilon": arabica_avg}


def main() -> None:
    raw_local = {}
    for spec in TARGETS:
        key = f"{spec['variety']}_{spec['location']}"
        raw_local[key] = load_local_series(spec)

    other_variety_panels = build_other_variety_panel(raw_local)
    exogenous = build_exogenous_panel()
    cepea_by_variety = {
        variety: cepea_series(fname) for variety, fname in CEPEA_FILE_BY_VARIETY.items()
    }

    rows = []
    for spec in TARGETS:
        key = f"{spec['variety']}_{spec['location']}"
        df = add_trailing_and_forward_vol(raw_local[key].copy())
        df["target_variety"] = spec["variety"]
        df["target_location"] = spec["location"]
        df["grade"] = spec["grade"]

        df = pd.merge_asof(
            df.sort_values("date"), exogenous.sort_values("date"),
            on="date", direction="backward", tolerance=ASOF_TOLERANCE,
        )
        other_panel = other_variety_panels[spec["variety"]]
        df = pd.merge_asof(
            df.sort_values("date"), other_panel.sort_values("date"),
            on="date", direction="backward", tolerance=ASOF_TOLERANCE,
        )
        cepea_panel = cepea_by_variety[spec["variety"]]
        df = pd.merge_asof(
            df.sort_values("date"), cepea_panel.sort_values("date"),
            on="date", direction="backward", tolerance=ASOF_TOLERANCE,
        )
        rows.append(df)

    master = pd.concat(rows, ignore_index=True)
    # Flag, don't drop: a handful of |return_1d| > 40% look like raw SEAGRI-BA
    # data-entry errors (decimal-point typos, or a shared unit glitch across
    # all three local series on 2023-08-31/09-01) -- see conversation notes.
    # Left in place so the researcher decides how to treat them (section 23
    # "exclusao de periodos extremos" is a named robustness check, not a
    # default).
    master["is_extreme_return"] = master["return_1d"].abs() > 40

    ordered_cols = [
        "date", "target_variety", "target_location", "grade",
        "price", "return_1d", "is_extreme_return", "hist_vol_5", "hist_vol_10", "hist_vol_20",
        "target_rv_h1", "target_rv_h5", "target_rv_h20",
        "ice_coffee_c_close", "ice_coffee_c_return",
        "cepea_close", "cepea_return",
        "usd_brl", "usd_brl_return", "usd_brl_vol_5",
        "selic", "vix", "vix_change", "brent", "brent_return",
        "dollar_index", "dollar_index_return",
        "sp500_return", "ibovespa_return",
        "us_treasury_10y", "fed_funds",
        "other_variety_return", "other_variety_vol_5",
        "soy_return", "soy_vol_5", "cotton_return", "cotton_vol_5",
        "cocoa_return", "cocoa_vol_5",
    ]
    master = master[ordered_cols].sort_values(["target_variety", "target_location", "date"])
    master.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(master)} rows to {OUT_PATH}")
    print(master.groupby(["target_variety", "target_location"]).agg(
        n_obs=("price", "size"),
        start=("date", "min"),
        end=("date", "max"),
        pct_missing_ice_coffee=("ice_coffee_c_return", lambda s: round(100 * s.isna().mean(), 1)),
        pct_missing_cepea=("cepea_return", lambda s: round(100 * s.isna().mean(), 1)),
        pct_missing_usd_brl=("usd_brl_return", lambda s: round(100 * s.isna().mean(), 1)),
        pct_missing_target_rv_h20=("target_rv_h20", lambda s: round(100 * s.isna().mean(), 1)),
    ).to_string())


if __name__ == "__main__":
    main()
