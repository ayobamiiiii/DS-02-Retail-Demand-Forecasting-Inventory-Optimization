# Final Project Audit — Favorita Forecasting and Inventory Planning

**Audit date:** 2026-11-20
**Repository:** project checkout (commands and paths are relative to its root)
**Dataset:** Corporación Favorita Grocery Sales Forecasting (not Walmart M5)

## Executive result

The item and benchmark-series selectors now use only sales dated before the earliest configured evaluation window. For this run, the exclusive cutoff is **2017-06-21**; it precedes model selection (June 21–July 4), calibration (July 5–18), and held-out evaluation (July 19–August 15). A cache provenance manifest records the cutoff and selected IDs, and unmanifested or mismatched caches are rebuilt. Four regression tests cover the explicit cutoff, post-cutoff independence for item and series ranking, and derived cutoff dates.

The pipeline was rerun successfully from the corrected sample during this audit. Naive Weekly, LightGBM, Prophet, and Darts Regression have actual held-out predictions and refreshed metrics. Metrics were independently recomputed from the saved prediction CSVs using the project scoring function. OR-Tools returned `OPTIMAL` for all 110 recommendations. All 26 automated tests pass, including dashboard callback checks for all six tabs. TFT remains **pending GPU training and held-out evaluation**; no TFT prediction or score is claimed.

## Requirements audit

| Requirement | Status | Evidence / result |
|---|---|---|
| Forecast models and validation metrics | **Complete with scope limits** | Naive Weekly and LightGBM primary evaluation use 116 series/3,248 observations. Naive Weekly, LightGBM, Prophet and Darts use a common six-series/168-observation holdout. |
| Chronological separation and sample selection | **Complete for configured run** | Exclusive sample-selection cutoff 2017-06-21; selection June 21–July 4; calibration July 5–18; final evaluation July 19–August 15. Selectors and cache manifest enforce the boundary; tests verify future sales cannot affect membership. |
| Hierarchical rollups | **Complete** | Generated bottom-up rollups at item-store, item, store, city, state, and overall levels. Hierarchy tests pass. |
| Prediction intervals | **Complete, empirical** | Separate 14-day residual calibration followed by untouched 28-day evaluation. Primary 90% interval coverage is 90.4249% (3,248 rows); common-sample coverage is 93.4524% (168 rows). |
| OR-Tools inventory recommendations | **Complete for modeled assumptions** | 110/110 statuses are `OPTIMAL`; four order quantities reach the configured cap. Cost inputs and zero on-hand assumption are proxies. |
| Automated tests | **Complete** | 26 tests pass, including inventory edge cases, hierarchy, lag/covariate checks, cutoff tests, and dashboard view/callback checks. |
| Six-tab dashboard | **Complete locally** | Automated callback checks cover Executive Overview, Sales Trends, Demand Forecasts, Model Performance, Inventory Optimization, and Store and Item Analysis. Local app route and layout are also verified. |
| README, setup, architecture, data/API docs | **Complete locally** | README setup and execution, architecture diagram, data/API contracts, and audit report reflect this run. |
| TFT | **Pending training and held-out evaluation** | I attempted a Colab Tesla T4 run. A selection stage completed, but final refit/evaluation did not: the attempts encountered CUDA deterministic-operation and PyTorch Forecasting/Lightning scheduler configuration incompatibilities, and later retries were constrained by Colab GPU usage limits. The notebook was updated for scheduler compatibility, but no final checkpoint, TFT metrics, or predictions are claimed; TFT is excluded from all model comparisons. |
| GitHub repository | **Local removal prepared; remote update pending** | `origin` points to the intended repository; `main` matched `origin/main` at `b0db367` before this review. The local index now removes the reflection path and `.gitignore` excludes both reflection locations. Because no commit or push was made, the remote still contains the previously committed file until a future removal commit is pushed; earlier history was not rewritten. |

## Model metrics

### Primary rolling holdout: 116 series, 3,248 rows

| Model | WAPE | RMSSE (mean per series) | RMSE |
|---|---:|---:|---:|
| Naive Weekly | 40.9416% | 0.691895 | 21.499179 |
| LightGBM | 29.5346% | 0.648599 | 14.649477 |

The scored dates are 2017-07-19–2017-08-15. One-day-ahead predictions use actual target history through the previous date. LightGBM was selected using the earlier selection block, not the scored holdout. The primary interval was calibrated July 5–18, with radius 19.479636 sales units and 90.4249% empirical coverage on the later holdout.

### Common comparison: six series, 168 rows

| Model | WAPE | RMSSE (mean per series) | RMSE |
|---|---:|---:|---:|
| LightGBM | 15.6490% | 0.470417 | 29.465924 |
| Darts Regression | 18.3903% | 0.552134 | 35.399453 |
| Prophet | 20.9164% | 0.620779 | 39.298949 |
| Naive Weekly | 24.7320% | 0.772054 | 51.084838 |

All models use the same six series and 28 evaluation dates. LightGBM has the lowest metrics in this sample. The separate 90% interval has 93.4524% coverage. Common-sample and primary scores have different item-series scopes and should not be compared directly.

## Inventory optimization outputs

| Measure | Actual output |
|---|---:|
| Solver status | 110 `OPTIMAL` |
| Recommended order units | 54,814 |
| Estimated holding cost | 1,086.45 generic cost units |
| Estimated stockout cost | 4,942.50 generic cost units |
| Optimized objective cost | 6,028.95 generic cost units |
| Current policy order units | 82,453 |
| Current policy objective cost | 11,556.75 generic cost units |
| Modeled reduction vs current policy | 47.83% |

The CP-SAT objective weights low/median/high demand scenarios 25%/50%/25%; safety stock uses the configured 90% service level and calibration-residual uncertainty scaled by the square root of the 16-day horizon. Favorita contains no on-hand stock or actual cost inputs. The run assumes zero stock and proxy costs of 0.20 per leftover unit and 2.00 per unmet unit. The reduction is modeled only and is not realized business savings.

## Dataset and changed sample

The processed panel contains 195,692 rows from 2013-01-02 through 2017-08-15, 39 items, 116 item-store series, and stores 1, 24, and 37. The profile has zero missing values and duplicate series-day keys. The pre-cutoff item sample is selected using sales through 2017-06-20 only. Compared with the previous sample, item 1503847 was replaced by item 273528; the panel size and series count are unchanged. The local TFT CSV and tracked manifest were regenerated to match the corrected primary sample and preserve the same validation period.

## Test and verification results

- `./.venv/bin/python -m unittest discover -s tests -v`: **26 passed, 0 failed** (10 inventory, 3 lagged-covariate, 1 hierarchy, 8 dashboard, 4 sample-selection tests).
- The selection-specific suite includes four checks for explicit and derived cutoff behavior and post-cutoff independence.
- Dashboard launch: `./.venv/bin/python app.py` started on `http://127.0.0.1:8050/`; live HTTP GET returned 200. Automated callback tests for all six tabs passed.
- Pipeline: rerun on 2026-11-20 with `MPLBACKEND=Agg` and `MPLCONFIGDIR=.mplconfig`; completed successfully, generating 1,760 official-test forecast rows and 110 recommendations. The optional Darts API emitted its documented upstream deprecation warning but executed in all benchmark stages.
- `./.venv/bin/python -m compileall -q src dashboard app.py tests`: passed.
- `./.venv/bin/python -m pip check`: passed with no broken requirements.
- Saved primary and common-sample metrics were independently recomputed from held-out CSV predictions and agree with the JSON outputs to floating-point precision.
- TFT remains untrained; no metric or prediction artifact is reported.

## Repository hygiene and privacy

Raw and processed datasets, generated outputs other than this audit report, model artifacts, virtual environments, and common credential file patterns are excluded by `.gitignore`. `git ls-files` showed only the processed-data placeholder and this audit report under `data/` and `outputs/`; no dataset CSVs or credential files are tracked. The README contains only the owner-provided public profile links. No history rewrite was made.

## Limitations

- **TFT training and final held-out evaluation were not completed.** A Colab Tesla T4 selection run completed, but final training encountered CUDA deterministic-operation and PyTorch Forecasting/Lightning scheduler compatibility errors. Scheduler handling was adjusted, but subsequent attempts were constrained by Colab GPU usage limits. No TFT metrics or predictions are claimed.
- The evaluated sample is limited to 39 items and three Favorita stores; the Prophet/Darts benchmark is further limited to six series.
- Inventory quantities use zero-default on-hand stock and assumed generic costs. Business-approved stock, lead-time, capacity, holding, and shortage inputs are needed before operational interpretation.
- Prediction-interval coverage is empirical and limited.
