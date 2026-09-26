# Forecasting Local Coffee Price Volatility in Bahia with Cross-Commodity and Financial Information

Code, data and manuscript for the study of out-of-sample volatility forecasting of local
Arabica (Vitória da Conquista, Luís Eduardo Magalhães) and Conilon (Eunápolis) coffee prices
in Bahia, Brazil.

## Structure

- `01_audit_data.py` … `14_robustness_checks.py` — pipeline, run in numerical order
  - 01–03: data audit, SEAGRI-BA refill, master dataset
  - 04–05: GARCH/HAR-RV baselines and ML experiments (E1–E4), expanding-window validation
  - 07–10: common-support comparison, DM-HLN tests, regimes, feature importance/SHAP
  - 12–13: figures; 14: robustness checks (QLIKE floor, Holm/BH, ex-ante regimes)
- `temporal_validation.py`, `test_temporal_validation.py` — annual split with label purging
- `dados_commodities_bahia/` — raw source files, master dataset and result CSVs
- `els-cas-templates/cas-sc-template.tex` — manuscript (Elsevier CAS single column), `figs/`, `references.bib`

## Environment

Python 3.12 with pandas, NumPy, arch, scikit-learn, LightGBM, XGBoost, shap (see
`dados_commodities_bahia/validated_run_manifest.json`). Plotting scripts require matplotlib.
Use `n_jobs=1` for scikit-learn estimators.

## Not versioned

`ml_predictions_cafe.csv` (88 MB) and `baseline_predictions_cafe.csv` (21 MB) are regenerated
by scripts 04 and 05; scripts 07–14 read them.

The manuscript compiles with `latexmk -xelatex cas-sc-template.tex`.
