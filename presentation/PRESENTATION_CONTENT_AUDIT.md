# Presentation Content Audit

This record maps slide claims to repository evidence. The deck does not change forecasting code, model metrics, prediction files, or optimization outputs.

## Evidence used

| Slides | Claims checked | Primary project sources |
|---|---|---|
| 1–3 | Project name, bounded 39-item / three-store evaluation scope, owner, problem, and dataset context | README.md; outputs/final_audit_report.md; outputs/metrics/data_profile.json; config/default.yaml |
| 3 | Favorita source tables, bounded 39-item / 116-series / 3-store sample, panel rows and dates | outputs/metrics/data_profile.json; config/default.yaml; docs/DATA_AND_API.md |
| 4 | Pipeline stages, evaluation artifacts, hierarchy, CP-SAT, and dashboard | docs/ARCHITECTURE.md; src/pipeline.py; src/evaluation/model_comparison.py; src/forecasting/hierarchy.py; src/optimization/orders.py; dashboard/app.py |
| 5 | Selection cutoff, preprocessing, feature definitions, data-quality checks, observed sales by store | src/data/favorita.py; src/features/favorita.py; outputs/metrics/data_profile.json; outputs/forecasts/sales_history.csv |
| 6 | Executed model set, TFT status, chronological windows and forecast protocol | src/pipeline.py; src/evaluation/model_comparison.py; outputs/metrics/model_comparison.json; outputs/final_audit_report.md |
| 7 | Primary and common-sample held-out metrics, date windows, interval coverage | outputs/metrics/model_metrics.json; outputs/metrics/model_comparison.json; outputs/forecasts/validation.csv; outputs/forecasts/model_comparison_validation.csv |
| 8 | Solver statuses, current-policy and optimized units/costs, cost and quantity assumptions | outputs/recommendations/orders.csv; config/default.yaml; src/optimization/orders.py; outputs/final_audit_report.md |
| 9 | Six dashboard tabs, callback tests, local-only status, and stated limitations | dashboard/app.py; tests/test_dashboard_tabs.py; outputs/final_audit_report.md; README.md; notebooks/tft_colab.ipynb |
| 10 | Summary result, tests, and next steps | outputs/metrics/model_metrics.json; outputs/metrics/model_comparison.json; outputs/final_audit_report.md; tests/ |

## Metric and output verification

- Primary evaluation: 116 series and 3,248 held-out rows, 2017-07-19 through 2017-08-15; the table values come from model_metrics.json and the saved validation.csv.
- Common comparison: six series and 168 held-out rows over the same dates; all four model values come from model_comparison.json and model_comparison_validation.csv.
- Inventory totals are aggregated from the 110-row orders.csv file. All 110 saved solver statuses are OPTIMAL.
- The monthly chart uses only actual sales from sales_history.csv, grouped by store for January through August 15, 2017.
- The metric table distinguishes the 116-series primary evaluation from the six-series benchmark; their scores are not presented as directly comparable.
- The diagram, charts, and metric tables are native editable PowerPoint objects. The repository contains no dashboard screenshot file, so the dashboard slide lists the six verified tabs instead of presenting a fabricated screenshot.
- TFT has no reported metric or prediction. Notes repeat the documented Colab Tesla T4 attempt and the incomplete final refit/evaluation status.

## Presentation validation

The deck contains ten slides in the requested order, includes slide-by-slide speaking notes, and targets a ten-minute delivery. Final PPTX package, slide geometry, native chart/table placement, and PowerPoint import checks are recorded in the private validation receipt under .codex-presentation-build/nexvia-final/finalizer/.