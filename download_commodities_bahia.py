#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download de dados para estudo de previsão de volatilidade de commodities
com foco empírico na Bahia.

Saídas: um CSV por conjunto de dados + relatórios de sucesso/falha.

Fontes tentadas automaticamente:
  1) SEAGRI-BA: cotações locais (scraping da tabela pública, com cache anual)
  2) CEPEA/ESALQ: séries históricas de soja, algodão e café
  3) Banco Central do Brasil (SGS): USD/BRL e Selic
  4) FRED: VIX, Brent, índice amplo do dólar, USD/BRL, Treasury 10Y e Fed Funds
  5) CBOE: VIX direto (quando o CSV público responder)
  6) Yahoo Finance (API pública não oficial): futuros contínuos e índices
  7) EIA: Brent direto em XLS (tentativa; FRED/EIA funciona como fallback)

Fontes oficiais NÃO baixadas automaticamente:
  - CME Group DataMine / históricos oficiais completos
  - ICE Data Services / históricos oficiais completos
Essas fontes normalmente exigem conta, produto de dados, API específica ou licença.
O script registra isso em manual_sources.csv.

Uso:
  python download_commodities_bahia.py

Opcional:
  python download_commodities_bahia.py --out dados_commodities_bahia \
      --seagri-start 2000-01-01 --global-start 1990-01-01

Dependências:
  pip install pandas requests beautifulsoup4 lxml openpyxl xlrd

Observações:
  - O downloader da SEAGRI é deliberadamente moderado (delay entre requisições)
    e usa cache por ano para permitir retomada.
  - Não preenche dados ausentes e não faz interpolação.
  - Mantém os dados brutos o mais próximo possível das fontes.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin

MISSING = []
try:
    import pandas as pd
except ImportError:
    MISSING.append("pandas")
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    MISSING.append("requests")
try:
    from bs4 import BeautifulSoup
except ImportError:
    MISSING.append("beautifulsoup4")

if MISSING:
    print("Dependências ausentes:", ", ".join(MISSING), file=sys.stderr)
    print(
        "Instale com: pip install pandas requests beautifulsoup4 lxml openpyxl xlrd",
        file=sys.stderr,
    )
    raise SystemExit(2)


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "AcademicResearchDownloader/1.0"
)


@dataclass
class Result:
    source: str
    dataset: str
    status: str
    rows: int = 0
    start_date: str = ""
    end_date: str = ""
    path: str = ""
    note: str = ""
    url: str = ""


RESULTS: list[Result] = []


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()


def slugify(s: str) -> str:
    x = norm_text(s)
    x = re.sub(r"[^a-z0-9]+", "_", x).strip("_")
    return x or "dataset"


def parse_iso(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def safe_date_range(df: "pd.DataFrame") -> tuple[str, str]:
    if df is None or df.empty:
        return "", ""
    candidates = [c for c in df.columns if norm_text(c) in {"date", "data", "data da cotacao"}]
    if not candidates:
        candidates = [c for c in df.columns if "data" in norm_text(c) or "date" in norm_text(c)]
    if not candidates:
        return "", ""
    c = candidates[0]
    s = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    s = s.dropna()
    if s.empty:
        return "", ""
    return s.min().date().isoformat(), s.max().date().isoformat()


def record(source: str, dataset: str, status: str, df: Optional["pd.DataFrame"] = None,
           path: Optional[Path] = None, note: str = "", url: str = "") -> None:
    rows = 0 if df is None else int(len(df))
    start, end = safe_date_range(df) if df is not None else ("", "")
    RESULTS.append(
        Result(
            source=source,
            dataset=dataset,
            status=status,
            rows=rows,
            start_date=start,
            end_date=end,
            path=str(path) if path else "",
            note=note[:1000],
            url=url,
        )
    )


def make_session() -> "requests.Session":
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7"})
    return session


def save_csv(df: "pd.DataFrame", path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def clean_numeric_br(x: Any) -> Any:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s or "sem cot" in norm_text(s):
        return None
    s = s.replace("R$", "").replace("US$", "").replace("U$$", "").replace("%", "").strip()
    # Padrão brasileiro: 1.234,56 -> 1234.56
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# SEAGRI-BA
# -----------------------------------------------------------------------------

SEAGRI_URL = "https://www.ba.gov.br/seagri/cotacao"


def discover_select_options(session: "requests.Session", url: str) -> dict[str, dict[str, str]]:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out: dict[str, dict[str, str]] = {}
    for sel in soup.find_all("select"):
        name = sel.get("name") or sel.get("id") or ""
        if not name:
            continue
        opts: dict[str, str] = {}
        for opt in sel.find_all("option"):
            label = opt.get_text(" ", strip=True)
            value = opt.get("value")
            if value is not None:
                opts[label] = value
        if opts:
            out[name] = opts
    return out


def find_select_name(selects: dict[str, dict[str, str]], keyword: str) -> Optional[str]:
    k = norm_text(keyword)
    for name in selects:
        if k in norm_text(name):
            return name
    # fallback por conteúdo típico
    for name, opts in selects.items():
        labels = " ".join(norm_text(x) for x in opts.keys())
        if k == "produto" and any(w in labels for w in ["soja", "algodao", "cacau", "cafe"]):
            return name
        if k == "praca" and any(w in labels for w in ["barreiras", "ilheus", "salvador"]):
            return name
    return None


def match_options(options: dict[str, str], keywords: Iterable[str]) -> list[tuple[str, str]]:
    keys = [norm_text(k) for k in keywords]
    matched: list[tuple[str, str]] = []
    for label, value in options.items():
        nl = norm_text(label)
        if not value or nl in {"todos", "todas", "all", "selecione o produto", "selecione"}:
            continue
        if any(k in nl for k in keys):
            matched.append((label, value))
    # remove duplicatas por value/label
    seen = set()
    unique = []
    for item in matched:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def parse_seagri_table(html: str) -> "pd.DataFrame":
    soup = BeautifulSoup(html, "html.parser")
    target = None
    target_headers: list[str] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        nh = [norm_text(h) for h in headers]
        if any("data" in h for h in nh) and any("produto" in h for h in nh) and any("preco" in h for h in nh):
            target = table
            target_headers = headers
            break
    if target is None:
        return pd.DataFrame()

    rows = []
    for tr in target.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if len(cells) < len(target_headers):
            cells += [None] * (len(target_headers) - len(cells))
        rows.append(cells[: len(target_headers)])
    if not rows:
        return pd.DataFrame(columns=target_headers)

    df = pd.DataFrame(rows, columns=target_headers)
    # Padronização leve
    rename = {}
    for c in df.columns:
        nc = norm_text(c)
        if "data" in nc:
            rename[c] = "date"
        elif "produto" in nc:
            rename[c] = "product"
        elif "praca" in nc:
            rename[c] = "market"
        elif "tipo" in nc:
            rename[c] = "type"
        elif "unidade" in nc:
            rename[c] = "unit"
        elif "preco" in nc:
            rename[c] = "price_raw"
    df = df.rename(columns=rename)
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["date"] = dt.dt.strftime("%Y-%m-%d")
    if "price_raw" in df.columns:
        df["price"] = df["price_raw"].map(clean_numeric_br)
    return df


def year_chunks(start: date, end: date):
    y = start.year
    while y <= end.year:
        a = max(start, date(y, 1, 1))
        b = min(end, date(y, 12, 31))
        yield y, a, b
        y += 1


def download_seagri(
    session: "requests.Session",
    outdir: Path,
    start: date,
    end: date,
    delay: float,
    max_pages: int,
) -> None:
    source = "SEAGRI-BA"
    log("SEAGRI-BA: descobrindo filtros de produto...")
    try:
        selects = discover_select_options(session, SEAGRI_URL)
        prod_name = find_select_name(selects, "produto")
        if not prod_name:
            raise RuntimeError(f"Não foi possível identificar o <select> de produto. Selects encontrados: {list(selects)}")
        product_options = selects[prod_name]
        targets = match_options(product_options, ["soja", "algodao", "cacau", "cafe"])
        if not targets:
            raise RuntimeError("Nenhuma opção de Soja/Algodão/Cacau/Café encontrada no filtro do portal.")
        log("SEAGRI-BA: produtos encontrados: " + ", ".join(label for label, _ in targets))
    except Exception as e:
        record(source, "cotacoes_bahia", "FAILED", note=str(e), url=SEAGRI_URL)
        log(f"SEAGRI-BA falhou na descoberta: {e}")
        return

    cache_dir = outdir / "_cache" / "seagri"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = outdir / "seagri_ba"
    data_dir.mkdir(parents=True, exist_ok=True)

    for label, prod_value in targets:
        slug = slugify(label)
        pieces: list[pd.DataFrame] = []
        failed_chunks = []
        for yr, a, b in year_chunks(start, end):
            cache_csv = cache_dir / f"{slug}_{yr}.csv"
            empty_marker = cache_dir / f"{slug}_{yr}.empty"
            if cache_csv.exists():
                try:
                    pieces.append(pd.read_csv(cache_csv))
                    continue
                except Exception:
                    pass
            if empty_marker.exists():
                continue

            log(f"SEAGRI-BA: {label} {yr}")
            chunk_frames: list[pd.DataFrame] = []
            previous_signature = None
            repeated = 0
            chunk_ok = True
            for page in range(max_pages):
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
                try:
                    r = session.get(SEAGRI_URL, params=params, timeout=75)
                    r.raise_for_status()
                    df = parse_seagri_table(r.text)
                except Exception as e:
                    failed_chunks.append(f"{yr}: page {page}: {e}")
                    chunk_ok = False
                    break

                if df.empty:
                    break

                # Proteção contra paginação que repita a última página.
                signature_cols = [c for c in ["date", "product", "market", "type", "price_raw"] if c in df.columns]
                signature = tuple(df[signature_cols].astype(str).head(3).to_numpy().ravel()) if signature_cols else tuple(df.head(3).astype(str).to_numpy().ravel())
                if signature == previous_signature:
                    repeated += 1
                    if repeated >= 1:
                        break
                else:
                    repeated = 0
                previous_signature = signature

                df["source_product_label"] = label
                df["source_product_value"] = prod_value
                chunk_frames.append(df)
                time.sleep(delay)
            else:
                failed_chunks.append(f"{yr}: atingiu max_pages={max_pages}; pode haver dados não coletados")
                chunk_ok = False

            if chunk_frames:
                chunk = pd.concat(chunk_frames, ignore_index=True)
                chunk = chunk.drop_duplicates()
                save_csv(chunk, cache_csv)
                pieces.append(chunk)
            elif chunk_ok:
                empty_marker.write_text("no rows\n", encoding="utf-8")

        if pieces:
            full = pd.concat(pieces, ignore_index=True).drop_duplicates()
            if "date" in full.columns:
                full = full.sort_values([c for c in ["date", "market", "type"] if c in full.columns])
            path = save_csv(full, data_dir / f"seagri_ba_{slug}.csv")
            status = "OK" if not failed_chunks else "PARTIAL"
            record(source, label, status, full, path, note=" | ".join(failed_chunks), url=SEAGRI_URL)
        else:
            record(source, label, "FAILED" if failed_chunks else "EMPTY", note=" | ".join(failed_chunks), url=SEAGRI_URL)


# -----------------------------------------------------------------------------
# CEPEA
# -----------------------------------------------------------------------------

CEPEA_SERIES = {
    "soja_paranagua": "https://cepea.org.br/br/indicador/series/soja.aspx?id=92",
    "soja_parana": "https://cepea.org.br/br/indicador/series/soja.aspx?id=12",
    "cafe_arabica": "https://cepea.org.br/br/indicador/series/cafe.aspx?id=23",
    "cafe_robusta": "https://cepea.org.br/br/indicador/series/cafe.aspx?id=24",
    "algodao_8_dias": "https://cepea.org.br/br/indicador/series/algodao.aspx?id=54",
}


def _flatten_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in tup if str(x) != "nan").strip("_") for tup in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def parse_cepea_content(content: bytes, content_type: str = "") -> "pd.DataFrame":
    errors = []
    # 1) Tenta Excel real (OLE xls ou xlsx zip)
    if content.startswith(b"\xd0\xcf\x11\xe0") or content.startswith(b"PK\x03\x04") or "excel" in content_type.lower():
        try:
            df = pd.read_excel(io.BytesIO(content))
            return _flatten_columns(df)
        except Exception as e:
            errors.append(f"read_excel: {e}")

    # 2) Muitos endpoints antigos servem HTML com MIME Excel.
    for enc in ("utf-8", "latin1"):
        try:
            text = content.decode(enc, errors="replace")
            tables = pd.read_html(io.StringIO(text), decimal=",", thousands=".")
            if tables:
                # Preferir a maior tabela com aparência de série temporal.
                tables = [_flatten_columns(t) for t in tables]
                tables.sort(key=lambda t: (len(t), t.shape[1]), reverse=True)
                return tables[0]
        except Exception as e:
            errors.append(f"read_html/{enc}: {e}")

    # 3) CSV/TSV disfarçado
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, encoding="latin1")
            if df.shape[1] > 1 and len(df) > 1:
                return _flatten_columns(df)
        except Exception as e:
            errors.append(f"read_csv/{sep}: {e}")
    raise RuntimeError("; ".join(errors[-6:]))


def normalize_cepea_dates(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    # A primeira coluna costuma ser data.
    candidate = None
    for c in out.columns:
        nc = norm_text(c)
        if "data" in nc or nc == "date":
            candidate = c
            break
    if candidate is None and len(out.columns):
        candidate = out.columns[0]
    if candidate is not None:
        parsed = pd.to_datetime(out[candidate], errors="coerce", dayfirst=True)
        # Só renomeia se uma parcela razoável for data.
        if parsed.notna().sum() >= max(2, int(0.2 * len(out))):
            out = out.rename(columns={candidate: "date"})
            out["date"] = parsed.dt.strftime("%Y-%m-%d")
            out = out[parsed.notna()].copy()
    return out


def download_cepea(session: "requests.Session", outdir: Path) -> None:
    data_dir = outdir / "cepea"
    raw_dir = outdir / "raw" / "cepea"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CEPEA_SERIES.items():
        log(f"CEPEA: {name}")
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            content = r.content
            ctype = r.headers.get("Content-Type", "")
            try:
                df = parse_cepea_content(content, ctype)
                df = normalize_cepea_dates(df)
                path = save_csv(df, data_dir / f"cepea_{name}.csv")
                record("CEPEA", name, "OK", df, path, url=url)
            except Exception as parse_error:
                # Preserva o retorno bruto para inspeção manual.
                ext = ".xls" if "excel" in ctype.lower() or content.startswith(b"\xd0\xcf") else ".bin"
                raw_path = raw_dir / f"cepea_{name}{ext}"
                raw_path.write_bytes(content)
                record("CEPEA", name, "FAILED_PARSE", path=raw_path,
                       note=f"Arquivo bruto preservado. {parse_error}", url=url)
        except Exception as e:
            record("CEPEA", name, "FAILED", note=str(e), url=url)


# -----------------------------------------------------------------------------
# Banco Central do Brasil - SGS
# -----------------------------------------------------------------------------

BCB_SERIES = {
    "usd_brl_venda": {"code": 1, "start": date(1984, 11, 28), "unit": "BRL por USD"},
    "selic_anualizada_252": {"code": 1178, "start": date(1986, 6, 4), "unit": "% a.a."},
    "selic_percentual_dia": {"code": 11, "start": date(1986, 6, 4), "unit": "% ao dia"},
}


def add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)


def bcb_chunks(start: date, end: date, years: int = 9):
    cur = start
    while cur <= end:
        nxt = min(end, add_years(cur, years) - timedelta(days=1))
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def download_bcb(session: "requests.Session", outdir: Path, end: date) -> None:
    data_dir = outdir / "bcb"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, meta in BCB_SERIES.items():
        code = meta["code"]
        url_base = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        log(f"BCB SGS {code}: {name}")
        frames = []
        errors = []
        for a, b in bcb_chunks(meta["start"], end, years=9):
            params = {
                "formato": "json",
                "dataInicial": a.strftime("%d/%m/%Y"),
                "dataFinal": b.strftime("%d/%m/%Y"),
            }
            try:
                r = session.get(url_base, params=params, timeout=90)
                r.raise_for_status()
                data = r.json()
                if data:
                    frames.append(pd.DataFrame(data))
            except Exception as e:
                errors.append(f"{a}..{b}: {e}")
        if frames:
            df = pd.concat(frames, ignore_index=True).drop_duplicates()
            if "data" in df.columns:
                df["date"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
                df = df.drop(columns=["data"])
            if "valor" in df.columns:
                df["value"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                df = df.drop(columns=["valor"])
            df["series_code"] = code
            df["unit"] = meta["unit"]
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            path = save_csv(df, data_dir / f"bcb_{name}.csv")
            status = "OK" if not errors else "PARTIAL"
            record("Banco Central do Brasil", name, status, df, path, note=" | ".join(errors), url=url_base)
        else:
            record("Banco Central do Brasil", name, "FAILED", note=" | ".join(errors), url=url_base)


# -----------------------------------------------------------------------------
# FRED - CSV público sem chave de API
# -----------------------------------------------------------------------------

FRED_SERIES = {
    "vix": "VIXCLS",
    "brent_spot": "DCOILBRENTEU",
    "usd_brl_fed": "DEXBZUS",
    "broad_us_dollar_index": "DTWEXBGS",
    "us_treasury_10y": "DGS10",
    "fed_funds_effective": "DFF",
}


def download_fred(session: "requests.Session", outdir: Path, start: date, end: date) -> None:
    data_dir = outdir / "fred"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, series_id in FRED_SERIES.items():
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {
            "id": series_id,
            "cosd": start.isoformat(),
            "coed": end.isoformat(),
        }
        log(f"FRED: {series_id} ({name})")
        try:
            r = session.get(url, params=params, timeout=90)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            if df.empty:
                raise RuntimeError("CSV vazio")
            df.columns = ["date" if norm_text(c) in {"observation_date", "date"} else c for c in df.columns]
            value_cols = [c for c in df.columns if c != "date"]
            if value_cols:
                df = df.rename(columns={value_cols[0]: "value"})
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["series_id"] = series_id
            path = save_csv(df, data_dir / f"fred_{name}_{series_id.lower()}.csv")
            record("FRED", name, "OK", df, path, url=r.url)
        except Exception as e:
            record("FRED", name, "FAILED", note=str(e), url=url)


# -----------------------------------------------------------------------------
# CBOE direto
# -----------------------------------------------------------------------------

def download_cboe_vix(session: "requests.Session", outdir: Path) -> None:
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    log("CBOE: VIX histórico direto")
    try:
        r = session.get(url, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        rename = {}
        for c in df.columns:
            if norm_text(c) == "date":
                rename[c] = "date"
            elif norm_text(c) in {"open", "high", "low", "close"}:
                rename[c] = norm_text(c)
        df = df.rename(columns=rename)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        path = save_csv(df, outdir / "cboe" / "cboe_vix_history.csv")
        record("CBOE", "VIX_history", "OK", df, path, url=url)
    except Exception as e:
        record("CBOE", "VIX_history", "FAILED", note=str(e), url=url)


# -----------------------------------------------------------------------------
# EIA direto - tentativa de XLS diário do Brent
# -----------------------------------------------------------------------------

def download_eia_brent(session: "requests.Session", outdir: Path) -> None:
    url = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"
    log("EIA: Brent diário direto")
    try:
        r = session.get(url, timeout=120)
        r.raise_for_status()
        try:
            xls = pd.ExcelFile(io.BytesIO(r.content))
            # Geralmente há uma aba 'Data 1'; escolhe a que contiver mais linhas.
            best_df = None
            for sheet in xls.sheet_names:
                tmp = pd.read_excel(xls, sheet_name=sheet)
                if best_df is None or len(tmp) > len(best_df):
                    best_df = tmp
            if best_df is None or best_df.empty:
                raise RuntimeError("XLS EIA sem dados")
            df = _flatten_columns(best_df)
            # tenta achar coluna data
            date_col = next((c for c in df.columns if "date" in norm_text(c)), df.columns[0])
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            df = df[parsed.notna()].copy()
            df = df.rename(columns={date_col: "date"})
            df["date"] = parsed[parsed.notna()].dt.strftime("%Y-%m-%d").values
            path = save_csv(df, outdir / "eia" / "eia_brent_daily.csv")
            record("U.S. EIA", "brent_daily", "OK", df, path, url=url)
        except Exception as e:
            raw = outdir / "raw" / "eia" / "RBRTEd.xls"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(r.content)
            record("U.S. EIA", "brent_daily", "FAILED_PARSE", path=raw,
                   note=f"XLS bruto preservado. {e}", url=url)
    except Exception as e:
        record("U.S. EIA", "brent_daily", "FAILED", note=str(e), url=url)


# -----------------------------------------------------------------------------
# Yahoo Finance - endpoint chart não oficial
# -----------------------------------------------------------------------------

YAHOO_TICKERS = {
    # Commodities alvo
    "soybean_futures": "ZS=F",
    "cocoa_futures": "CC=F",
    "coffee_futures": "KC=F",
    "cotton_futures": "CT=F",
    # Cross-commodity / energia
    "corn_futures": "ZC=F",
    "sugar_11_futures": "SB=F",
    "brent_futures": "BZ=F",
    "wti_futures": "CL=F",
    "gold_futures": "GC=F",
    # Financeiro
    "vix": "^VIX",
    "sp500": "^GSPC",
    "ibovespa": "^BVSP",
    "usd_brl": "BRL=X",
    "us_dollar_index": "DX-Y.NYB",
}


def unix_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def download_yahoo_chart(session: "requests.Session", ticker: str, start: date, end: date) -> "pd.DataFrame":
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": unix_ts(start),
        "period2": unix_ts(end + timedelta(days=1)),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    r = session.get(url, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()
    err = payload.get("chart", {}).get("error")
    if err:
        raise RuntimeError(str(err))
    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise RuntimeError("Yahoo retornou result vazio")
    obj = results[0]
    ts = obj.get("timestamp") or []
    qlist = obj.get("indicators", {}).get("quote") or []
    if not ts or not qlist:
        raise RuntimeError("Yahoo sem timestamps/OHLC")
    q = qlist[0]
    adjlist = obj.get("indicators", {}).get("adjclose") or []
    adj = adjlist[0].get("adjclose", []) if adjlist else []
    n = len(ts)
    def arr(key):
        v = q.get(key) or []
        return v + [None] * (n - len(v))
    adj = adj + [None] * (n - len(adj))
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d"),
        "open": arr("open"),
        "high": arr("high"),
        "low": arr("low"),
        "close": arr("close"),
        "adj_close": adj[:n],
        "volume": arr("volume"),
    })
    meta = obj.get("meta") or {}
    df["ticker"] = ticker
    df["currency"] = meta.get("currency")
    df["exchange"] = meta.get("exchangeName")
    df = df.dropna(subset=["close"], how="all").drop_duplicates(subset=["date"], keep="last")
    return df


def download_yahoo(session: "requests.Session", outdir: Path, start: date, end: date) -> None:
    data_dir = outdir / "yahoo_finance"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, ticker in YAHOO_TICKERS.items():
        log(f"Yahoo Finance: {ticker} ({name})")
        try:
            df = download_yahoo_chart(session, ticker, start, end)
            path = save_csv(df, data_dir / f"yahoo_{name}_{slugify(ticker)}.csv")
            record("Yahoo Finance (unofficial API)", name, "OK", df, path,
                   note="Continuous/nearby market series as provided by Yahoo; verify rollover conventions before publication.",
                   url=f"https://finance.yahoo.com/quote/{ticker}")
        except Exception as e:
            record("Yahoo Finance (unofficial API)", name, "FAILED", note=str(e),
                   url=f"https://finance.yahoo.com/quote/{ticker}")


# -----------------------------------------------------------------------------
# Relatórios
# -----------------------------------------------------------------------------

MANUAL_SOURCES = [
    {
        "source": "CME Group",
        "datasets": "Soybean futures official historical settlements / volume / open interest",
        "reason": "Histórico oficial completo normalmente depende de DataMine/produto de dados, conta ou licença. O script usa Yahoo como fallback para série de mercado.",
        "next_step": "Obter acesso CME DataMine/API e exportar os contratos/settlements necessários.",
        "url": "https://www.cmegroup.com/market-data/datamine-historical-data.html",
    },
    {
        "source": "ICE Data Services",
        "datasets": "Cocoa, Coffee C e Cotton No. 2 official historical futures",
        "reason": "Históricos oficiais completos e dados por contrato/volume/open interest podem exigir serviço/licença de dados.",
        "next_step": "Obter acesso ICE Data Services/Connect e exportar séries por contrato; usar Yahoo apenas como fallback exploratório.",
        "url": "https://www.ice.com/market-data",
    },
]


def write_reports(outdir: Path) -> None:
    report_path = outdir / "download_report.csv"
    pd.DataFrame([asdict(r) for r in RESULTS]).to_csv(report_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(MANUAL_SOURCES).to_csv(outdir / "manual_sources.csv", index=False, encoding="utf-8-sig")

    # Manifesto resumido por arquivo baixado
    manifest = []
    for r in RESULTS:
        if r.path:
            manifest.append({
                "source": r.source,
                "dataset": r.dataset,
                "status": r.status,
                "rows": r.rows,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "file": r.path,
                "source_url": r.url,
                "note": r.note,
            })
    pd.DataFrame(manifest).to_csv(outdir / "source_manifest.csv", index=False, encoding="utf-8-sig")

    ok = sum(r.status == "OK" for r in RESULTS)
    partial = sum(r.status in {"PARTIAL", "FAILED_PARSE"} for r in RESULTS)
    failed = sum(r.status in {"FAILED", "EMPTY"} for r in RESULTS)
    summary = [
        "DOWNLOAD DE DADOS - COMMODITIES / BAHIA",
        "=" * 48,
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        f"OK: {ok}",
        f"Parcial/parse pendente: {partial}",
        f"Falha/vazio: {failed}",
        "",
        "Veja download_report.csv para erros detalhados.",
        "Veja manual_sources.csv para CME/ICE e outras fontes que exigem acesso específico.",
        "",
        "IMPORTANTE PARA O PAPER:",
        "- Não trate Yahoo Finance como substituto metodológico automático do dado oficial por contrato.",
        "- Para futuros, documente convenção de rollover/continuous contract antes da modelagem final.",
        "- Preserve os CSVs brutos; faça limpeza/feature engineering em etapa separada.",
    ]
    (outdir / "README_downloads.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa dados para estudo ML de volatilidade de commodities na Bahia.")
    parser.add_argument("--out", default="dados_commodities_bahia", help="Diretório de saída")
    parser.add_argument("--seagri-start", default="2000-01-01", help="Início tentativa SEAGRI (YYYY-MM-DD)")
    parser.add_argument("--global-start", default="1990-01-01", help="Início séries globais/Yahoo/FRED (YYYY-MM-DD)")
    parser.add_argument("--end", default=date.today().isoformat(), help="Data final (YYYY-MM-DD)")
    parser.add_argument("--seagri-delay", type=float, default=0.30, help="Delay entre páginas SEAGRI, segundos")
    parser.add_argument("--seagri-max-pages", type=int, default=500, help="Máximo de páginas por produto/ano")
    parser.add_argument("--skip-seagri", action="store_true", help="Pula SEAGRI")
    parser.add_argument("--skip-cepea", action="store_true", help="Pula CEPEA")
    parser.add_argument("--skip-yahoo", action="store_true", help="Pula Yahoo Finance")
    args = parser.parse_args()

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seagri_start = parse_iso(args.seagri_start)
    global_start = parse_iso(args.global_start)
    end = parse_iso(args.end)
    if end < global_start:
        raise SystemExit("--end deve ser >= --global-start")

    session = make_session()
    log(f"Saída: {outdir}")

    if not args.skip_seagri:
        download_seagri(session, outdir, seagri_start, end, args.seagri_delay, args.seagri_max_pages)
    if not args.skip_cepea:
        download_cepea(session, outdir)

    download_bcb(session, outdir, end)
    download_fred(session, outdir, global_start, end)
    download_cboe_vix(session, outdir)
    download_eia_brent(session, outdir)

    if not args.skip_yahoo:
        download_yahoo(session, outdir, global_start, end)

    write_reports(outdir)

    log("Concluído. Resumo:")
    for r in RESULTS:
        log(f"{r.status:12s} | {r.source:28s} | {r.dataset:28s} | rows={r.rows}")
    log(f"Relatório: {outdir / 'download_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
