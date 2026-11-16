# Final Project Audit — Favorita Forecasting and Inventory Planning

**Audit date:** 2026-11-16

**Workspace:** `/Users/mac/Documents/Codex/2026-11-14/files-pasted-by-the-user-project`
**Dataset:** Corporación Favorita Grocery Sales Forecasting (the M5 source files are not present)

## Executive result

The Favorita pipeline completed from the existing project. Naive Weekly, LightGBM, Prophet, and Darts ran and were compared on the same six series and the same untouched evaluation dates. The separate primary baseline run evaluates 116 series. Both use chronological selection, calibration, and evaluation periods. The model comparison dashboard reads the generated JSON/CSV outputs dynamically. The OR-Tools optimizer returned `OPTIMAL` for all 110 current recommendation rows. All 16 tests pass, including dashboard tab smoke checks.

TFT is implemented and was genuinely attempted with PyTorch Forecasting. Its dependencies imported, but the bounded CPU attempt did not return during the earlier audit run. Apple MPS is built into the Torch package but reports unavailable on this Mac. A minimal Google Colab GPU notebook and exact 116-series data export are now prepared; the Colab training has not been run, so TFT is still not scored or represented as successfully evaluated.

## Requirements audit

| Assignment requirement | Status | Evidence / result |
|---|---|---|
| Forecast models and actual validation metrics | **Complete with limitation** | Naive Weekly and LightGBM scored on the 116-series primary holdout. Prophet and Darts each ran in all benchmark windows and were scored with the same six-series common holdout as Naive/LightGBM. Colab TFT notebook/data handoff is prepared, but GPU training has not completed and has no score. |
| Chronological train/validation separation and leakage prevention | **Complete** | Selection: 2017-06-21–2017-07-04; calibration: 2017-07-05–2017-07-18; untouched evaluation: 2017-07-19–2017-08-15. Sales lags and store covariates use prior-day information; recursive forecasts use generated predictions. Promotion and calendar inputs are known-ahead data. |
| Hierarchical reconciliation | **Complete** | Generated bottom-up rollups at item-store, item, store, city, state, and overall levels. The hierarchy test passes; current output contains 2,512 rows over all six levels. |
| Prediction intervals and coverage | **Complete, empirical** | Absolute residual quantiles calibrated on the separate 14-day calibration period. Primary 116-series holdout coverage: 90.12% (nominal 90%, 3,248 observations). Common six-series selected-model coverage: 93.45% (168 observations). Coverage is measured only on the later untouched block. |
| OR-Tools optimization and all tests | **Complete** | 110/110 recommendations have `OPTIMAL` status. Ten inventory tests cover zero/sufficient stock, high demand, order and inventory bounds, cost tradeoffs, service levels, invalid inputs, and solver optimality versus brute force. All 16 project tests pass. |
| Six dashboard tabs | **Complete** | Home route and callback smoke checks pass for Executive Overview, Sales Trends, Demand Forecasts, Model Performance, Inventory Optimization, and Store and Item Analysis. This is Flask test-client/callback verification, not browser visual QA. |
| Dataset assumptions and limitations | **Complete** | Favorita is explicitly identified as the available alternative. The sample, no-stock/no-cost data limitations, proxy cost units, safety-stock assumptions, and interval limits are documented. |
| README and requirements | **Complete** | README covers project/problem, data, architecture, stack, setup, execution, model scores, inventory method, assumptions, and dashboard instructions. Requirements install in the checked environment; `pip check` passes. |
| Architecture diagram | **Complete** | README has a Mermaid diagram from raw data through evaluation, rollups, OR-Tools, outputs, and Dash. |
| GitHub readiness | **Complete locally** | Project-local Git repository initialized, `.gitignore` excludes datasets, generated outputs, credentials, model files, and virtual environments, and a clean initial source/documentation commit was created. No remote is configured and no push was performed. |

## Model metrics

### Primary pipeline validation

The primary one-day-ahead metrics are computed from held-out predictions for 116 item-store series, 3,248 observations, over 2017-07-19–2017-08-15. Training history for metric scaling ends before the holdout. LightGBM was selected on the earlier selection window, not on this scored holdout.

| Model | WAPE | RMSSE (mean per series) | RMSE |
|---|---:|---:|---:|
| Naive Weekly | 41.1774% | 0.704982 | 22.850307 |
| LightGBM | 29.9650% | 0.683098 | 15.713903 |

Primary interval: calibrated 2017-07-05–2017-07-18; evaluated 2017-07-19–2017-08-15; radius 20.421784 units; observed coverage 90.1170% over 3,248 rows.

### Common-sample model comparison

These scores come from the same six item-store series and 168 observations (28 dates) for every executed model. Selection and calibration each use 84 observations over 14 days. Model selection uses WAPE from 2017-06-21–2017-07-04; the final evaluation is 2017-07-19–2017-08-15.

| Model | WAPE | RMSSE (mean per series) | RMSE |
|---|---:|---:|---:|
| Naive Weekly | 24.7320% | 0.772054 | 51.084838 |
| LightGBM | 15.6490% | 0.470417 | 29.465924 |
| Prophet | 20.9164% | 0.620779 | 39.298949 |
| Darts Regression | 18.3903% | 0.552134 | 35.399453 |

LightGBM had the lowest selection WAPE and was therefore used for the calibrated interval. The 90% interval reached 93.4524% coverage on the untouched final period (168 rows). Prophet uses a per-series fit. Darts uses the Darts autoregressive `RegressionModel` with 28 lags and a shared Ridge estimator; Darts executed in selection, calibration, and evaluation. Darts 0.41 emitted a deprecation warning for `RegressionModel`, but the model ran successfully.

### TFT execution record

The local compact PyTorch Forecasting TFT path was attempted on six series, but it did not return during that audit; MPS was unavailable and no dependency exception was observed. For the final handoff, `notebooks/tft_colab.ipynb` now uses the exact primary 116-series split and a T4-sized architecture (hidden size 8, two heads, batch 32, up to 15 epochs with early stopping). Its dataset export retains real history, marks `unit_sales` as the unknown target, and evaluates the same rolling one-step validation observations. The Colab run has not yet occurred. TFT remains excluded from all scored comparisons; no TFT metric or prediction is claimed.

## Inventory optimization

Current run outputs (110 item-store recommendations):

| Measure | Actual output |
|---|---:|
| Solver status | 110 `OPTIMAL` |
| Recommended units | 56,580 |
| Estimated holding cost | 1,108.50 cost units |
| Estimated stockout cost | 5,020.00 cost units |
| Optimized total objective cost | 6,128.50 cost units |
| Current policy units | 85,375 |
| Current policy objective cost | 11,887.50 cost units |
| Modeled cost reduction vs current policy | 48.45% |

The OR-Tools objective balances expected holding and shortage costs under low/median/high demand scenarios (25%/50%/25%). Safety stock uses the configured 90% service level and calibration residual variation scaled by the square root of forecast horizon. Default on-hand is zero because Favorita does not provide inventory. Holding cost 0.20 per excess unit and stockout cost 2.00 per unmet unit are declared proxy cost units, not real currency or observed business costs. The modeled reduction is conditional on those assumptions.

## Test and verification results

- `python -m unittest discover -s tests -v`: **16 passed**, 0 failures (including 10 inventory tests, hierarchy/leakage tests, and 2 dashboard tests).
- Dashboard home endpoint: HTTP 200; all six tab callbacks returned content.
- `python -m compileall -q src dashboard app.py tests`: passed.
- `python -m pip check`: `No broken requirements found`.
- `git diff --check`: passed before initial commit.
- Pipeline rerun: completed; wrote 1,760 forecast rows for 110 item-store pairs and 16 official test dates; wrote 110 inventory recommendations.
- Dataset profile: 195,692 panel rows; 116 series, 39 items, 3 stores, 3 states; 2013-01-02–2017-08-15; zero missing values in the generated panel and zero duplicate series-date keys.
- Hierarchy levels in generated output: item-store, item, store, city, state, overall.

## Partial and missing requirements / limitations

1. **TFT evaluation remains incomplete.** Code and dependencies are present and a constrained CPU attempt was made. MPS is unavailable; the CPU call did not finish within this audit run, so no TFT metric is reported.
2. **M5-specific validation is unavailable.** M5 files were not supplied; this project uses the Favorita alternative and does not claim M5 results.
3. **Sample scope is limited.** The primary model metrics use 116 series (39 items across stores 1, 24, and 37); Prophet/Darts common comparison uses six series to keep per-series and CPU experiments manageable. Results are not full-dataset estimates.
4. **Intervals are marginal empirical residual intervals.** They have a separate calibration block and untouched evaluation block but do not guarantee conditional coverage by item, store, or forecast horizon.
5. **Inventory costs and stock are assumptions.** The dataset contains neither on-hand stock nor actual holding/stockout economics. Replace defaults before operational use.
6. **Darts API warning.** The Darts 0.41 `RegressionModel` API used in the successful run is deprecated upstream; the current run succeeded, but a future Darts upgrade may require migrating to `SKLearnModel`.
7. **GitHub publication is not complete by design.** The local initial commit exists; there is no remote URL, and nothing was pushed. Add a real repository URL and authorize a push when ready.
