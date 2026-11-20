# Nexvia Final Project Presentation — Speaker Notes

**Target duration:** 10 minutes total. Timing is approximate and includes brief pauses to read the charts and tables.

Each slide also contains these notes inside the PowerPoint file.

## Slide 1 — Title

**Approximate time:** 35 seconds

Welcome. I am Misbaudeen Abdurrahman. This presentation covers my nationwide demand forecasting and inventory optimization project for the Nexvia Technologies Data Science internship. The evaluation uses 39 selected items across three Favorita stores, not the full nationwide network. The project builds a local workflow from daily item-store demand through forecast evaluation and constrained replenishment recommendations. I will focus on what was implemented, what the saved evaluation shows, and where the evidence has limits.

**Sources:** README.md; outputs/final_audit_report.md

## Slide 2 — Problem Statement

**Approximate time:** 50 seconds

Retail demand changes by product, location, weekday, promotion, and calendar events. If a forecast is too low, a store can be short of product when customers want it. If it is too high, the business holds excess stock and incurs carrying costs, capacity pressure, or waste risk. The project links these two decisions: forecast daily unit sales for an item at a store, then use those demand estimates in an order recommendation. Since the dataset does not include real stock or financial costs, the recommendations are modeled decision support, not operating instructions.

**Sources:** README.md; docs/DATA_AND_API.md; config/default.yaml

## Slide 3 — Dataset and Business Understanding

**Approximate time:** 55 seconds

The source is the Corporación Favorita Grocery Sales competition dataset. It includes historical sales and promotions, item and store attributes, store transaction counts, oil prices, and holiday events. This run builds 195,692 daily panel rows for 39 selected items, 116 item-store series, and stores 1, 24, and 37, spanning three provinces. The panel covers January 2013 through August 15, 2017. The data is useful for studying demand with product and store context. The sample is intentionally bounded; it does not represent every Favorita store or all items, so I describe it as a nationwide demand forecasting project in context, not as a full nationwide model evaluation.

**Sources:** outputs/metrics/data_profile.json; config/default.yaml; docs/DATA_AND_API.md; src/data/favorita.py

## Slide 4 — System Architecture

**Approximate time:** 55 seconds

This is the end-to-end system. Favorita source tables are selected, aggregated, and joined into a daily panel. Feature engineering creates prior-sales lags and rolling statistics alongside calendar, promotion, holiday, metadata, and lagged auxiliary inputs. Naive Weekly and LightGBM form the primary evaluation; Prophet and Darts Regression run on a smaller common sample. Selection, calibration, and held-out evaluation use chronological windows. The pipeline also creates future forecasts, rolls item-store forecasts up through item, store, city, and state levels, and sends demand scenarios to an OR-Tools CP-SAT optimizer. Saved CSV and JSON outputs feed the locally run Dash application. The Colab TFT handoff remains separate and is not included among successful forecasts.

**Sources:** docs/ARCHITECTURE.md; src/pipeline.py; src/evaluation/model_comparison.py; src/forecasting/hierarchy.py; src/optimization/orders.py; dashboard/app.py

## Slide 5 — Data Preparation and Feature Engineering

**Approximate time:** 60 seconds

The data preparation first ranks candidate items using only nonnegative sales before the exclusive June 21 cutoff. That date comes before the model selection, calibration, and final evaluation periods. The selected item-store data is made into a daily panel. Duplicate date-store-item rows are aggregated, and days without a sales row are represented as zero. The saved panel profile reports no missing values and no duplicate series-days. The feature builder creates lags at 1, 7, 14, and 28 days, plus rolling averages and standard deviation from shifted prior values so the target date cannot contribute to its own predictors. Calendar, promotions, holidays, metadata, and prior-day oil and transactions add context. The chart shows observed Favorita sales by store for the sampled items from January through August 15, 2017; it contains actuals, not model predictions.

**Sources:** src/data/favorita.py; src/features/favorita.py; outputs/metrics/data_profile.json; outputs/forecasts/sales_history.csv

## Slide 6 — Forecasting Models

**Approximate time:** 60 seconds

Four approaches have completed evaluation. Naive Weekly provides the reference forecast from the previous week. LightGBM uses the project’s lagged demand and contextual features. Prophet and Darts Regression are included in a common benchmark that is deliberately smaller to bound runtime. Temporal Fusion Transformer was the intended sequence-model extension, but it is not a completed model result. The temporal protocol has three blocks: June 21 to July 4 for selection, July 5 to July 18 for interval calibration, then July 19 to August 15 for held-out scoring. The main 116-series metrics use rolling one-day-ahead predictions. The separate six-series model comparison forecasts the 28-day block recursively. Series membership is selected before the selection block begins.

**Sources:** src/pipeline.py; src/evaluation/model_comparison.py; src/models/baselines.py; src/models/lightgbm_model.py; src/models/prophet_model.py; src/models/darts_model.py; outputs/metrics/model_comparison.json

## Slide 7 — Model Evaluation and Results

**Approximate time:** 85 seconds

The primary rolling validation contains 116 series and 3,248 observations over July 19 through August 15, 2017. Naive Weekly produced 40.9416 percent WAPE, 0.691895 mean RMSSE, and 21.499179 RMSE. LightGBM produced 29.5346 percent WAPE, 0.648599 mean RMSSE, and 14.649477 RMSE. On this same primary sample, LightGBM lowers WAPE by about 11.41 percentage points relative to Naive Weekly. The separate model comparison uses six common series and 168 observations on the same evaluation dates: LightGBM 15.6490 percent WAPE, 0.470417 RMSSE, 29.465924 RMSE; Darts Regression 18.3903 percent, 0.552134, 35.399453; Prophet 20.9164 percent, 0.620779, 39.298949; and Naive Weekly 24.7320 percent, 0.772054, 51.084838. These tables must not be combined because the item-store sample differs. WAPE is total absolute error divided by total actual sales, RMSSE is averaged across per-series scaled errors, and RMSE remains in sales units per day. The primary nominal 90 percent interval had 90.4249 percent empirical coverage; the common-sample interval had 93.4524 percent coverage. Both are empirical results, not guarantees.

**Sources:** outputs/metrics/model_metrics.json; outputs/metrics/model_comparison.json; outputs/forecasts/validation.csv; outputs/forecasts/model_comparison_validation.csv; src/evaluation/metrics.py

## Slide 8 — Inventory Optimization

**Approximate time:** 85 seconds

The optimizer uses OR-Tools CP-SAT to select an integer order quantity that minimizes expected holding plus shortage costs across low, median, and high demand scenarios weighted 25, 50, and 25 percent. It uses forecast uncertainty for a 90 percent service level. Quantity is bounded by the 2,000-unit order limit and remaining capacity under the 5,000-unit maximum inventory. On-hand stock defaults to zero because Favorita does not provide inventory. Costs are assumed generic units: 0.20 per excess unit and 2.00 per unmet unit. In the saved output, all 110 recommendations solved OPTIMAL. The recommendations total 54,814 units, compared with 82,453 units under the current policy. The optimized estimated holding cost is 1,086.45 and estimated shortage cost is 4,942.50, for an objective of 6,028.95 generic cost units. The current policy objective is 11,556.75. The 47.83 percent reduction is modeled under these assumptions and is not a realized business saving. Four recommendations reach the order cap.

**Sources:** src/optimization/orders.py; src/pipeline.py; config/default.yaml; outputs/recommendations/orders.csv; outputs/final_audit_report.md

## Slide 9 — Dashboard, Challenges and Limitations

**Approximate time:** 70 seconds

The local Plotly Dash application has six tabs. Executive Overview shows high-level KPIs. Sales Trends draws actual history. Demand Forecasts separates historical actuals, saved validation predictions, and the future forecast. Model Performance compares saved held-out results. Inventory Optimization shows the recommendations and assumptions. Store and Item Analysis provides the filtered forecast table. The dashboard route and all six tab callbacks were verified, but no public deployment is claimed. The technical caveats matter: evaluation is limited to 39 items and three stores, and the Prophet/Darts comparison to six series. Inventory stock defaults to zero with generic cost assumptions. Interval coverage is empirical. For TFT, a Google Colab Tesla T4 selection run completed, but the final refit and held-out evaluation encountered CUDA deterministic-operation and PyTorch Forecasting/Lightning scheduler compatibility issues. After some compatibility changes, later attempts were constrained by Colab GPU usage limits. Therefore, the project claims no TFT predictions or scores.

**Sources:** dashboard/app.py; tests/test_dashboard_tabs.py; outputs/final_audit_report.md; notebooks/tft_colab.ipynb; README.md

## Slide 10 — Conclusion and Next Steps

**Approximate time:** 50 seconds

To close, the project demonstrates a working pipeline from item-store demand forecasts to constrained inventory recommendations. On the primary 116-series holdout, LightGBM achieved 29.53 percent WAPE compared with 40.94 percent for Naive Weekly, while also improving RMSSE and RMSE in the same sample. Four models completed evaluation in the smaller common benchmark. The dashboard and optimizer connect the model outputs to business-facing views, while the inventory economics remain assumed. The next work is to evaluate a broader range of items and stores, replace default stock and generic cost inputs with approved operational data, assess interval coverage over more rolling origins, and complete TFT training and held-out evaluation if GPU availability allows. Thank you, and I welcome your questions.

**Sources:** outputs/metrics/model_metrics.json; outputs/metrics/model_comparison.json; outputs/final_audit_report.md; tests/
