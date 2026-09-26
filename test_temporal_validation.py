import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


def module(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TemporalTests(unittest.TestCase):
    def test_invalid_variance_path_is_not_a_zero_forecast(self):
        baseline = module('04_baselines_garch_har.py')
        self.assertTrue(hasattr(baseline, 'aggregate_variance'))
        self.assertTrue(np.isnan(baseline.aggregate_variance(pd.Series([1., np.nan]))))
        self.assertTrue(np.isnan(baseline.aggregate_variance(pd.Series([np.inf]))))
        self.assertEqual(baseline.aggregate_variance(pd.Series([9., 16.])), 5.)

    def test_comparison_uses_same_dates_and_pooled_rmse(self):
        path = Path('07_compare_validated.py')
        self.assertTrue(path.exists(), 'comparison must align prediction dates')
        compare = module(str(path))
        rows = pd.DataFrame([
            {'series':'s','horizon':1,'date':d,'experiment':e,'model':m,
             'valid_prediction':True,'actual_rv':1.,'qlike':v,
             'absolute_error':v,'squared_error':v*v}
            for d,e,m,v in [('2020-01-01','E0','HAR-RV',3.),
                             ('2020-01-02','E0','HAR-RV',4.),
                             ('2020-01-01','E1_own','RandomForest',2.)]])
        result, common = compare.common_summary(rows)
        self.assertEqual(len(common), 2)
        har = result[result.model == 'HAR-RV'].iloc[0]
        self.assertEqual(har.n_test, 1)
        self.assertEqual(har.rmse, 3.)

    def test_training_labels_end_before_year_boundary(self):
        ml = module('05_ml_experiments.py')
        self.assertTrue(hasattr(ml, 'annual_split'), 'annual split must purge future labels')
        dates = pd.to_datetime(['2019-12-20', '2019-12-27', '2019-12-30',
                                '2020-01-03', '2020-02-10', '2020-02-11'])
        df = pd.DataFrame({'date': dates, 'return_1d': 1., 'target_rv_h2': [2.,2.,2.,2.,np.nan,np.nan]})
        train, test = ml.annual_split(df, 2020, 2)
        self.assertEqual(train.date.tolist(), [pd.Timestamp('2019-12-20')])
        self.assertEqual(test.date.tolist(), [pd.Timestamp('2020-01-03')])

    def test_garch_forecast_origin_matches_forward_target(self):
        baseline = module('04_baselines_garch_har.py')
        r = np.random.default_rng(42).normal(size=310)
        r[301] = 12
        df = pd.DataFrame({'return_1d': r})
        res = arch_model(df.return_1d.iloc[:300], mean='Zero', vol='GARCH', p=1, q=1).fit(disp='off')
        fixed = arch_model(df.return_1d, mean='Zero', vol='GARCH', p=1, q=1).fix(res.params)
        expected = np.sqrt(fixed.forecast(horizon=1, start=301).variance.loc[301].sum())
        actual = baseline.fit_garch_predict(df, 300, np.array([301]), 1)[0]
        self.assertAlmostEqual(actual, expected, places=10)


if __name__ == '__main__':
    unittest.main()
