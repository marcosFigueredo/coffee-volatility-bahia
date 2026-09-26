"""Retry the specific SEAGRI-BA (product, year) chunks that failed during the
original scrape (see dados_commodities_bahia/download_report.csv notes) and
merge any recovered rows back into dados_commodities_bahia/seagri_ba/*.csv.

Diagnostic finding (2026-09-09): the SEAGRI cotacao endpoint itself works
fine today with both curl and Python requests -- the earlier "Response ended
prematurely" errors were transient. But direct re-querying also confirmed
that the *large* multi-hundred-day gaps found by 01_audit_data.py (e.g.
Soja/Barreiras has nothing between 2007-05-17 and 2009-01-14, a year that
was never flagged as failed) are NOT scraper artifacts -- SEAGRI's own
database has no rows for those windows. This script only recovers the
handful of pages (~50 rows each) that actually failed to download; it
cannot manufacture history the source doesn't have.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from download_commodities_bahia import (  # noqa: E402
    SEAGRI_URL,
    discover_select_options,
    find_select_name,
    make_session,
    match_options,
    parse_seagri_table,
    save_csv,
    slugify,
)
import pandas as pd  # noqa: E402

DATA_DIR = BASE_DIR / "dados_commodities_bahia"
CACHE_DIR = DATA_DIR / "_cache" / "seagri"
OUT_DIR = DATA_DIR / "seagri_ba"

# (slug, year) pulled from download_report.csv's failure notes.
FAILED_CHUNKS = [
    ("algodao", 2009),
    ("cacau_ate_15_30h", 2016),
    ("cafe_arabica", 2004),
    ("cafe_arabica", 2008),
    ("cafe_arabica", 2011),
    ("cafe_arabica", 2015),
    ("cafe_arabica", 2024),
    ("cafe_conillon", 2019),
]

MAX_PAGES = 500
PAGE_RETRIES = 5
DELAY = 0.35


def fetch_year(session, prod_name: str, prod_value: str, year: int) -> pd.DataFrame:
    a, b = date(year, 1, 1), date(year, 12, 31)
    frames = []
    previous_signature = None
    repeated = 0
    for page in range(MAX_PAGES):
        params = {
            prod_name: prod_value,
            "praca": "All",
            "tipo": "All",
            "data[min][date]": a.isoformat(),
            "data[max][date]": b.isoformat(),
            "order": "label_1",
            "sort": "asc",
            "page": page,
        }
        df = None
        last_err = None
        for attempt in range(PAGE_RETRIES):
            try:
                r = session.get(SEAGRI_URL, params=params, timeout=90)
                r.raise_for_status()
                df = parse_seagri_table(r.text)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        if df is None:
            raise RuntimeError(f"page {page}: exhausted {PAGE_RETRIES} retries: {last_err}")

        if df.empty:
            break

        signature_cols = [c for c in ["date", "product", "market", "type", "price_raw"] if c in df.columns]
        signature = (
            tuple(df[signature_cols].astype(str).head(3).to_numpy().ravel())
            if signature_cols else tuple(df.head(3).astype(str).to_numpy().ravel())
        )
        if signature == previous_signature:
            repeated += 1
            if repeated >= 1:
                break
        else:
            repeated = 0
        previous_signature = signature

        frames.append(df)
        time.sleep(DELAY)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def main() -> None:
    session = make_session()
    print("Discovering produto select options...")
    selects = discover_select_options(session, SEAGRI_URL)
    prod_name = find_select_name(selects, "produto")
    if not prod_name:
        raise SystemExit(f"could not find produto select; found: {list(selects)}")
    product_options = selects[prod_name]
    targets = match_options(product_options, ["soja", "algodao", "cacau", "cafe"])
    slug_to_value = {slugify(label): (label, value) for label, value in targets}
    print("Products:", {s: v[0] for s, v in slug_to_value.items()})

    touched_slugs: set[str] = set()
    for slug, year in FAILED_CHUNKS:
        if slug not in slug_to_value:
            print(f"SKIP {slug} {year}: no matching produto option found")
            continue
        label, prod_value = slug_to_value[slug]
        cache_path = CACHE_DIR / f"{slug}_{year}.csv"
        before_n = 0
        if cache_path.exists():
            try:
                before_n = len(pd.read_csv(cache_path))
            except Exception:
                before_n = 0

        print(f"Refetching {label} ({slug}) {year} (had {before_n} cached rows)...")
        try:
            df = fetch_year(session, prod_name, prod_value, year)
        except Exception as e:
            print(f"  FAILED again: {e}")
            continue

        if df.empty:
            print("  got 0 rows, leaving existing cache untouched")
            continue

        df["source_product_label"] = label
        df["source_product_value"] = prod_value
        save_csv(df, cache_path)
        print(f"  wrote {len(df)} rows to {cache_path} (was {before_n})")
        touched_slugs.add(slug)

    for slug in sorted(touched_slugs):
        year_files = sorted(CACHE_DIR.glob(f"{slug}_*.csv"))
        pieces = [pd.read_csv(p) for p in year_files]
        full = pd.concat(pieces, ignore_index=True).drop_duplicates()
        if "date" in full.columns:
            full = full.sort_values([c for c in ["date", "market", "type"] if c in full.columns])
        out_path = OUT_DIR / f"seagri_ba_{slug}.csv"
        save_csv(full, out_path)
        print(f"Rebuilt {out_path}: {len(full)} total rows from {len(year_files)} yearly caches")


if __name__ == "__main__":
    main()
