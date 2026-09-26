"""Consolidate all series runs and execute full experimental pipeline.

Merges separate series predictions into the canonical CSVs, then runs:
  - 07_compare_validated.py
  - 08_statistical_tests.py
  - 09_regime_analysis.py
  - 10_explainability_shap.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados_commodities_bahia"
SERIES_DIR = DATA_DIR / "ml_series_runs"


def consolidate_ml():
    # If series were run individually via run_ml_series.py, merge them
    pred_files = list(SERIES_DIR.glob("*_predictions.csv")) if SERIES_DIR.exists() else []
    res_files = list(SERIES_DIR.glob("*_results.csv")) if SERIES_DIR.exists() else []

    existing_pred = DATA_DIR / "ml_predictions_cafe.csv"
    existing_res = DATA_DIR / "ml_results_cafe.csv"

    all_preds = []
    if existing_pred.exists():
        all_preds.append(pd.read_csv(existing_pred))
    for f in pred_files:
        all_preds.append(pd.read_csv(f))

    all_res = []
    if existing_res.exists():
        all_res.append(pd.read_csv(existing_res))
    for f in res_files:
        all_res.append(pd.read_csv(f))

    if all_preds:
        merged_pred = pd.concat(all_preds, ignore_index=True).drop_duplicates(
            subset=["series", "horizon", "date", "experiment", "model"]
        )
        merged_pred.to_csv(existing_pred, index=False, encoding="utf-8-sig")
        print(f"Consolidated {len(merged_pred)} ML predictions across {merged_pred.series.nunique()} series: {merged_pred.series.unique().tolist()}")

    if all_res:
        merged_res = pd.concat(all_res, ignore_index=True).drop_duplicates(
            subset=["series", "horizon", "experiment", "model", "test_year"]
        )
        merged_res.to_csv(existing_res, index=False, encoding="utf-8-sig")
        print(f"Consolidated {len(merged_res)} ML fold results across {merged_res.series.nunique()} series.")


def run_pipeline():
    python_exe = sys.executable
    scripts = [
        "07_compare_validated.py",
        "08_statistical_tests.py",
        "09_regime_analysis.py",
        "10_explainability_shap.py",
    ]
    for s in scripts:
        script_path = BASE_DIR / s
        if script_path.exists():
            print(f"\n---> Running {s}...", flush=True)
            res = subprocess.run([python_exe, str(script_path)], cwd=str(BASE_DIR), capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(f"[STDERR]: {res.stderr}")


if __name__ == "__main__":
    consolidate_ml()
    run_pipeline()

