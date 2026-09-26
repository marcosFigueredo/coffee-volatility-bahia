"""Compare saved out-of-sample forecasts on identical dates, by series/horizon."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata

import numpy as np
import pandas as pd
from temporal_validation import annual_split

DATA = Path(__file__).resolve().parent / 'dados_commodities_bahia'
KEY = ['series', 'horizon', 'date']
CONFIG = ['experiment', 'model']


def common_summary(rows):
    rows = rows.copy()
    rows['configuration'] = rows.experiment + '/' + rows.model
    valid = rows[rows.valid_prediction].copy()
    expected = rows.configuration.nunique()
    counts = valid.groupby(KEY).configuration.nunique()
    keys = counts[counts == expected].reset_index()[KEY]
    common = valid.merge(keys, on=KEY, validate='many_to_one')
    summary = common.groupby(['series', 'horizon'] + CONFIG).agg(
        n_test=('date','size'), qlike=('qlike','mean'),
        mae=('absolute_error','mean'), mse=('squared_error','mean')).reset_index()
    summary['rmse'] = np.sqrt(summary.pop('mse'))
    return summary, common


def main():
    master = pd.read_csv(DATA / 'master_dataset_cafe.csv')
    audit = []
    for (variety, location), g in master.groupby(['target_variety', 'target_location']):
        g = g.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(g.date)
        for h in [1, 5, 20]:
            for year in sorted(dates.dt.year.unique()):
                if year < dates.dt.year.min() + 3:
                    continue
                train, test = annual_split(g, year, h)
                if test.empty or len(train) < 250:
                    continue
                old = g[dates.dt.year < year].dropna(subset=[f'target_rv_h{h}', 'return_1d'])
                audit.append(dict(series=f'{variety}_{location}', horizon=h, test_year=int(year),
                    old_n_train=len(old), purged_n_train=len(train), removed_labels=len(old)-len(train),
                    train_last_origin=train.date.max(), train_last_target_end=train.target_end_date.max(),
                    cutoff=f'{year}-01-01', first_test_origin=test.date.min(), n_test=len(test)))
    pd.DataFrame(audit).to_csv(DATA / 'validated_fold_audit.csv', index=False)
    baseline = pd.read_csv(DATA / 'baseline_predictions_cafe.csv')
    ml = pd.read_csv(DATA / 'ml_predictions_cafe.csv')
    rows = pd.concat([baseline, ml], ignore_index=True)
    assert not rows.duplicated(KEY + CONFIG).any(), 'Duplicate predictions'
    supervised = ~rows.model.isin(['GARCH(1,1)', 'EGARCH(1,1,1)'])
    assert (pd.to_datetime(rows.loc[supervised, 'train_last_target_end']) <
            pd.to_datetime(rows.loc[supervised, 'train_cutoff'])).all(), 'Label leakage'
    assert (pd.to_datetime(rows.train_last_date) < pd.to_datetime(rows.train_cutoff)).all()
    assert (pd.to_datetime(rows.target_end_date) > pd.to_datetime(rows.date)).all()
    assert rows.groupby(KEY).actual_rv.nunique().max() == 1, 'Different targets'
    assert ml.model.nunique() == 3 and ml.experiment.nunique() == 4
    assert len(ml[CONFIG].drop_duplicates()) == 12
    assert rows['series'].nunique() == 3 and set(rows.horizon) == {1,5,20}
    coverage = rows.groupby(['series','horizon'] + CONFIG).agg(
        attempted=('date','size'), valid=('valid_prediction','sum')).reset_index()
    coverage['valid_pct'] = coverage.valid / coverage.attempted * 100
    coverage.to_csv(DATA / 'validated_coverage.csv', index=False)
    primary = rows[rows.model != 'EGARCH(1,1,1)']
    summary, common = common_summary(primary)
    summary.to_csv(DATA / 'validated_comparison_by_series.csv', index=False)
    pooled = common.groupby(['horizon'] + CONFIG).agg(
        n_test=('date','size'), qlike=('qlike','mean'),
        mae=('absolute_error','mean'), mse=('squared_error','mean')).reset_index()
    pooled['rmse'] = np.sqrt(pooled.pop('mse'))
    pooled.to_csv(DATA / 'validated_comparison_pooled.csv', index=False)
    all_summary, _ = common_summary(rows)
    all_summary.to_csv(DATA / 'validated_comparison_including_egarch.csv', index=False)
    own = summary[summary.experiment == 'E1_own'][['series','horizon','model','qlike','mae','rmse']]
    gains = summary[summary.experiment.isin(['E2_cross_commodity','E3_financial','E4_full'])].merge(
        own, on=['series','horizon','model'], suffixes=('','_own'), validate='many_to_one')
    for metric in ['qlike','mae','rmse']:
        gains[metric + '_gain_pct'] = 100 * (1 - gains[metric] / gains[metric + '_own'])
    gains.to_csv(DATA / 'validated_incremental_gains.csv', index=False)
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','arch','scikit-learn','lightgbm','xgboost']}
    files = ['03_build_master_dataset.py','04_baselines_garch_har.py','05_ml_experiments.py',
             'temporal_validation.py','07_compare_validated.py']
    hashes = {f: hashlib.sha256((DATA.parent/f).read_bytes()).hexdigest() for f in files}
    hashes['master_dataset_cafe.csv'] = hashlib.sha256((DATA/'master_dataset_cafe.csv').read_bytes()).hexdigest()
    (DATA / 'validated_run_manifest.json').write_text(json.dumps(
        {'python':platform.python_version(),'versions':versions,'sha256':hashes,
         'refit_cutoff':'January 1, strictly before cutoff',
         'origin':'after close at t; target next h observed returns',
         'qlike_variance_floor':1e-4,'egarch_seed':42,
         'n_prediction_rows':len(rows),'n_common_primary_rows':len(common)},indent=2),encoding='utf-8')
    print(pooled.round(4).to_string(index=False))
    print('\nVerified predictions:', len(rows), 'Common primary:', len(common))


if __name__ == '__main__':
    main()
