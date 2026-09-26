"""Audit every raw commodity/financial series collected for the Bahia
volatility-forecasting project and write dados_commodities_bahia/data_audit.csv.

Implements the checklist from projeto_volatilidade_commodities_bahia_ml.md
(section 9 "Etapa 1 - Auditoria dos dados" and section 33 "Primeira tarefa
computacional"): for every series, report source, coverage, frequency,
missingness, duplicates, price staleness, gaps, unit stability and a first
pass "usable" flag, so the definitive list of commodities/sources (section
10) can be decided from evidence instead of assumption.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
DOWNLOAD_REPORT = DATA_DIR / "download_report.csv"
OUT_PATH = DATA_DIR / "data_audit.csv"

AUDIT_COLUMNS = [
    "source", "series", "commodity", "location", "type", "unit",
    "start_date", "end_date", "n_obs", "frequency",
    "missing_pct", "duplicate_pct", "unchanged_price_pct",
    "max_gap_days", "avg_obs_per_month", "business_days_pct",
    "unit_changes", "anomaly_pct", "usable", "notes",
]

# Series considered too thin to be anything but noise, regardless of other
# metrics (e.g. a market/type combo with a single stray quote).
MIN_OBS_FOR_REVIEW = 30
MIN_OBS_USABLE = 250
MAX_GAP_DAYS_USABLE = 365
MAX_MISSING_PCT_USABLE = 95


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "unknown"


def infer_frequency(gap_days: pd.Series) -> str:
    if gap_days.empty:
        return "unknown"
    median_gap = gap_days.median()
    if median_gap <= 1.5:
        return "daily"
    if median_gap <= 4:
        return "business_daily"
    if median_gap <= 9:
        return "weekly"
    if median_gap <= 35:
        return "monthly"
    return "irregular"


def audit_one_series(dates, prices, *, source, series, commodity, location,
                      series_type, unit, notes: str = "") -> dict:
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    p = pd.to_numeric(pd.Series(list(prices)), errors="coerce")
    raw = pd.DataFrame({"date": d, "price": p}).dropna(subset=["date"])
    n_raw = len(raw)

    base = {
        "source": source, "series": series, "commodity": commodity,
        "location": location, "type": series_type, "unit": unit,
    }

    if n_raw == 0:
        return {
            **base,
            "start_date": "", "end_date": "", "n_obs": 0, "frequency": "unknown",
            "missing_pct": np.nan, "duplicate_pct": np.nan, "unchanged_price_pct": np.nan,
            "max_gap_days": np.nan, "avg_obs_per_month": np.nan, "business_days_pct": np.nan,
            "unit_changes": 0, "anomaly_pct": np.nan, "usable": False,
            "notes": " | ".join(x for x in [notes, "no valid rows"] if x),
        }

    duplicate_pct = round(100 * (1 - raw["date"].nunique() / n_raw), 2)

    clean = raw.dropna(subset=["price"]).sort_values("date")
    clean = clean.groupby("date", as_index=False)["price"].last()
    n_obs = len(clean)

    notes_parts = [notes] if notes else []
    if n_obs == 0:
        return {
            **base,
            "start_date": "", "end_date": "", "n_obs": 0, "frequency": "unknown",
            "missing_pct": np.nan, "duplicate_pct": duplicate_pct, "unchanged_price_pct": np.nan,
            "max_gap_days": np.nan, "avg_obs_per_month": np.nan, "business_days_pct": np.nan,
            "unit_changes": 0, "anomaly_pct": np.nan, "usable": False,
            "notes": " | ".join(notes_parts + ["price column unparsable for every row"]),
        }

    start_date = clean["date"].min()
    end_date = clean["date"].max()
    span_days = (end_date - start_date).days

    gaps = clean["date"].diff().dt.days.dropna()
    max_gap_days = int(gaps.max()) if not gaps.empty else 0
    frequency = infer_frequency(gaps)

    bdays = max(len(pd.bdate_range(start_date, end_date)), 1)
    business_days_pct = round(100 * n_obs / bdays, 2)
    missing_pct = round(max(0.0, 100 - business_days_pct), 2)

    span_months = max(span_days / 30.44, 1 / 30.44)
    avg_obs_per_month = round(n_obs / span_months, 2)

    price_diff = clean["price"].diff()
    unchanged_price_pct = round(100 * (price_diff == 0).sum() / max(n_obs - 1, 1), 2)

    returns = np.log(clean["price"] / clean["price"].shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) >= 20 and returns.std(ddof=0) > 0:
        z = (returns - returns.mean()) / returns.std(ddof=0)
        anomaly_pct = round(100 * (z.abs() > 5).sum() / len(returns), 2)
    else:
        anomaly_pct = np.nan

    non_positive = int((clean["price"] <= 0).sum())
    if non_positive:
        notes_parts.append(f"{non_positive} non-positive prices")
    if duplicate_pct > 0:
        notes_parts.append(f"{duplicate_pct:.1f}% duplicate dates collapsed (kept last)")
    if n_obs < MIN_OBS_FOR_REVIEW:
        notes_parts.append("too few observations, likely noise")

    usable = bool(
        n_obs >= MIN_OBS_USABLE
        and max_gap_days <= MAX_GAP_DAYS_USABLE
        and missing_pct <= MAX_MISSING_PCT_USABLE
    )

    return {
        **base,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "n_obs": n_obs,
        "frequency": frequency,
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "unchanged_price_pct": unchanged_price_pct,
        "max_gap_days": max_gap_days,
        "avg_obs_per_month": avg_obs_per_month,
        "business_days_pct": business_days_pct,
        "unit_changes": 0,
        "anomaly_pct": anomaly_pct,
        "usable": usable,
        "notes": " | ".join(notes_parts),
    }


def audit_seagri(path: Path, commodity_label: str, note: str = "") -> list[dict]:
    """SEAGRI-BA files mix several praças (markets) and grades (types) per
    commodity in one CSV; each (market, type) pair is audited as its own
    candidate series, since only some will match the praças named in the
    project plan (section 8.1)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["market"] = df["market"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()

    records = []
    for (market, series_type), group in df.groupby(["market", "type"], dropna=False):
        unit_counts = group["unit"].value_counts()
        modal_unit = unit_counts.idxmax() if not unit_counts.empty else ""
        unit_changes = max(int(unit_counts.shape[0]) - 1, 0)

        rec = audit_one_series(
            group["date"], group["price"],
            source="SEAGRI-BA",
            series=f"seagri_{slugify(commodity_label)}_{slugify(market)}_{slugify(series_type)}",
            commodity=commodity_label,
            location=market,
            series_type=series_type,
            unit=modal_unit,
            notes=note,
        )
        rec["unit_changes"] = unit_changes
        if unit_changes:
            rec["notes"] = " | ".join(
                x for x in [rec["notes"], f"unit varies: {unit_counts.to_dict()}"] if x
            )
        records.append(rec)
    return records


def audit_yahoo(path: Path, commodity_label: str, dataset_key: str, note: str = "") -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    exchange = df["exchange"].dropna() if "exchange" in df else pd.Series(dtype=str)
    location = exchange.mode().iloc[0] if not exchange.empty else "international"
    return audit_one_series(
        df["date"], df["close"],
        source="Yahoo Finance (unofficial API)",
        series=f"yahoo_{dataset_key}",
        commodity=commodity_label,
        location=location,
        series_type="futures/index close",
        unit="see ticker (continuous/nearby contract, verify rollover)",
        notes=note,
    )


def audit_eia(path: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    price_col = next(c for c in df.columns if c != "date")
    return audit_one_series(
        df["date"], df[price_col],
        source="U.S. EIA",
        series="eia_brent_daily",
        commodity="Brent Crude Oil",
        location="international (spot)",
        series_type="spot",
        unit="USD/barrel",
    )


def audit_fred(path: Path, commodity_label: str, series_id: str, unit: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    value_col = next(c for c in df.columns if c != "observation_date")
    return audit_one_series(
        df["observation_date"], df[value_col],
        source="FRED",
        series=f"fred_{series_id.lower()}",
        commodity=commodity_label,
        location="international",
        series_type="index/rate",
        unit=unit,
    )


def audit_bcb(path: Path, commodity_label: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    unit = df["unit"].dropna().iloc[0] if "unit" in df and not df["unit"].dropna().empty else ""
    return audit_one_series(
        df["date"], df["value"],
        source="Banco Central do Brasil",
        series=f"bcb_{slugify(commodity_label)}",
        commodity=commodity_label,
        location="Brasil",
        series_type="official rate",
        unit=unit,
    )


def audit_cboe(path: Path) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    dates = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    return audit_one_series(
        dates, df["CLOSE"],
        source="CBOE",
        series="cboe_vix_history",
        commodity="VIX",
        location="international",
        series_type="index close",
        unit="index points",
    )


def audit_cepea(path: Path, commodity_label: str, series_key: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return audit_one_series(
        df["date"], df["price_brl"],
        source="CEPEA",
        series=f"cepea_{series_key}",
        commodity=commodity_label,
        location="Brasil (nacional)",
        series_type="a vista R$",
        unit="R$",
        notes="baixado manualmente via navegador em 2026-09-09 (download automatico bloqueado por Cloudflare)",
    )


CEPEA_MANUALLY_RECOVERED = {"soja_paranagua", "soja_parana", "cafe_arabica", "cafe_robusta"}


def failed_download_records() -> list[dict]:
    """CEPEA automatic download is blocked by a Cloudflare bot challenge
    (HTTP 403) that plain HTTP clients can't pass. 4 of 5 series were since
    recovered by manually downloading the .xls export via browser on
    2026-09-09 (see audit_cepea above); only algodao_8_dias is still
    missing. Keep the remaining gap visible as an unusable placeholder
    rather than silently dropping it. FRED/BCB/CBOE originally failed here
    too (SSL errors from the local Python `requests` stack, not the
    servers) but were re-fetched successfully via curl -- see
    audit_fred/audit_bcb/audit_cboe above."""
    if not DOWNLOAD_REPORT.exists():
        return []
    rep = pd.read_csv(DOWNLOAD_REPORT, encoding="utf-8-sig")
    failed = rep[
        (rep["status"].astype(str).str.upper() == "FAILED")
        & (rep["source"].astype(str).str.upper() == "CEPEA")
        & (~rep["dataset"].astype(str).isin(CEPEA_MANUALLY_RECOVERED))
    ]
    records = []
    for _, row in failed.iterrows():
        source = str(row.get("source", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        error_note = str(row.get("note", ""))[:200]
        records.append({
            "source": source,
            "series": f"{slugify(source)}_{slugify(dataset)}",
            "commodity": dataset,
            "location": "",
            "type": "",
            "unit": "",
            "start_date": "",
            "end_date": "",
            "n_obs": 0,
            "frequency": "unknown",
            "missing_pct": np.nan,
            "duplicate_pct": np.nan,
            "unchanged_price_pct": np.nan,
            "max_gap_days": np.nan,
            "avg_obs_per_month": np.nan,
            "business_days_pct": np.nan,
            "unit_changes": 0,
            "anomaly_pct": np.nan,
            "usable": False,
            "notes": f"download failed: {error_note}",
        })
    return records


SEAGRI_FILES = {
    "seagri_ba_soja.csv": ("Soja", "2001-2026: intermittent scraper gaps, see download_report.csv"),
    "seagri_ba_algodao.csv": ("Algodão", "2009: page 7 scrape failed, see download_report.csv"),
    "seagri_ba_cacau_ate_15_30h.csv": ("Cacau", "2016: page 7 scrape failed, see download_report.csv"),
    "seagri_ba_cafe_arabica.csv": ("Café Arábica", "multiple years: scrape failed, see download_report.csv"),
    "seagri_ba_cafe_conillon.csv": ("Café Conillon", "2019: page 3 scrape failed, see download_report.csv"),
}

YAHOO_FILES = {
    "yahoo_soybean_futures_zs_f.csv": ("Soja", "soybean_futures"),
    "yahoo_cocoa_futures_cc_f.csv": ("Cacau", "cocoa_futures"),
    "yahoo_coffee_futures_kc_f.csv": ("Café", "coffee_futures"),
    "yahoo_cotton_futures_ct_f.csv": ("Algodão", "cotton_futures"),
    "yahoo_corn_futures_zc_f.csv": ("Milho", "corn_futures"),
    "yahoo_sugar_11_futures_sb_f.csv": ("Açúcar", "sugar_futures"),
    "yahoo_brent_futures_bz_f.csv": ("Petróleo Brent", "brent_futures"),
    "yahoo_wti_futures_cl_f.csv": ("Petróleo WTI", "wti_futures"),
    "yahoo_gold_futures_gc_f.csv": ("Ouro", "gold_futures"),
    "yahoo_vix_vix.csv": ("VIX", "vix"),
    "yahoo_sp500_gspc.csv": ("S&P 500", "sp500"),
    "yahoo_ibovespa_bvsp.csv": ("Ibovespa", "ibovespa"),
    "yahoo_usd_brl_brl_x.csv": ("USD/BRL", "usd_brl"),
    "yahoo_us_dollar_index_dx_y_nyb.csv": ("Dollar Index", "us_dollar_index"),
}

FRED_FILES = {
    "fred_vix_vixcls.csv": ("VIX", "VIXCLS", "index points"),
    "fred_brent_spot_dcoilbrenteu.csv": ("Brent Crude Oil (spot)", "DCOILBRENTEU", "USD/barrel"),
    "fred_usd_brl_fed_dexbzus.csv": ("USD/BRL", "DEXBZUS", "BRL per USD"),
    "fred_broad_us_dollar_index_dtwexbgs.csv": ("Dollar Index (broad)", "DTWEXBGS", "index"),
    "fred_us_treasury_10y_dgs10.csv": ("US Treasury 10Y", "DGS10", "% p.a."),
    "fred_fed_funds_effective_dff.csv": ("Fed Funds Effective Rate", "DFF", "% p.a."),
}

BCB_FILES = {
    "bcb_usd_brl_venda.csv": "USD/BRL",
    "bcb_selic_anualizada_252.csv": "SELIC (anualizada 252)",
    "bcb_selic_percentual_dia.csv": "SELIC (% ao dia)",
}

CEPEA_FILES = {
    "cepea_cafe_arabica.csv": ("Café Arábica", "cafe_arabica"),
    "cepea_cafe_robusta.csv": ("Café Robusta", "cafe_robusta"),
    "cepea_soja_parana.csv": ("Soja", "soja_parana"),
    "cepea_soja_paranagua.csv": ("Soja", "soja_paranagua"),
}


def main() -> None:
    records: list[dict] = []

    for fname, (commodity, note) in SEAGRI_FILES.items():
        path = DATA_DIR / "seagri_ba" / fname
        if path.exists():
            records.extend(audit_seagri(path, commodity, note))
        else:
            records.append({
                "source": "SEAGRI-BA", "series": f"seagri_{slugify(commodity)}",
                "commodity": commodity, "location": "", "type": "", "unit": "",
                "start_date": "", "end_date": "", "n_obs": 0, "frequency": "unknown",
                "missing_pct": np.nan, "duplicate_pct": np.nan, "unchanged_price_pct": np.nan,
                "max_gap_days": np.nan, "avg_obs_per_month": np.nan, "business_days_pct": np.nan,
                "unit_changes": 0, "anomaly_pct": np.nan, "usable": False,
                "notes": f"file not found: {path}",
            })

    for fname, (commodity, key) in YAHOO_FILES.items():
        path = DATA_DIR / "yahoo_finance" / fname
        if path.exists():
            records.append(audit_yahoo(path, commodity, key))

    eia_path = DATA_DIR / "eia" / "eia_brent_daily.csv"
    if eia_path.exists():
        records.append(audit_eia(eia_path))

    for fname, (commodity, series_id, unit) in FRED_FILES.items():
        path = DATA_DIR / "fred" / fname
        if path.exists():
            records.append(audit_fred(path, commodity, series_id, unit))

    for fname, commodity in BCB_FILES.items():
        path = DATA_DIR / "bcb" / fname
        if path.exists():
            records.append(audit_bcb(path, commodity))

    cboe_path = DATA_DIR / "cboe" / "cboe_vix_history.csv"
    if cboe_path.exists():
        records.append(audit_cboe(cboe_path))

    for fname, (commodity, series_key) in CEPEA_FILES.items():
        path = DATA_DIR / "cepea" / fname
        if path.exists():
            records.append(audit_cepea(path, commodity, series_key))

    records.extend(failed_download_records())

    out = pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS)
    out = out.sort_values(
        ["usable", "source", "commodity", "location"],
        ascending=[False, True, True, True],
    )
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out["usable"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
