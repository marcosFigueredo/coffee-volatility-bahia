"""Optional independent series worker; same evaluation as script 05.

Usage: python run_ml_series.py 'conilon_EUNAPOLIS'
Writes separate artifacts, leaving the canonical combined outputs untouched.
"""
import importlib.util
from pathlib import Path
import sys

import pandas as pd

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ml_experiments', base / '05_ml_experiments.py')
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)
name = sys.argv[1]
master = pd.read_csv(ml.MASTER_PATH, encoding='utf-8-sig')
selected = master[master.target_variety + '_' + master.target_location == name]
if selected.empty:
    raise ValueError(f'Unknown series: {name}')
predictions = []
records = ml.evaluate_series(name, selected, predictions)
out = ml.DATA_DIR / 'ml_series_runs'
out.mkdir(exist_ok=True)
stem = name.replace(' ', '_')
pd.DataFrame(records).to_csv(out / f'{stem}_results.csv', index=False, encoding='utf-8-sig')
pd.concat(predictions, ignore_index=True).to_csv(out / f'{stem}_predictions.csv', index=False, encoding='utf-8-sig')
print(f'Completed {name}: {len(records)} evaluations', flush=True)
