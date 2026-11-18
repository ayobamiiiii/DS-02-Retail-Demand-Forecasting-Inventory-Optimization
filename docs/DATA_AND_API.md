# Data and Application Interfaces

## Dataset and source files

The project uses the Corporación Favorita Grocery Sales Forecasting Kaggle dataset. It does not use Walmart M5 data. Raw data is supplied locally under `data/raw/favorita/` and is excluded from Git.

| Source CSV | Source columns used or retained | Purpose |
|---|---|---|
| `train.csv` | `id,date,store_nbr,item_nbr,unit_sales,onpromotion` | Historical sales target and promotion flag |
| `test.csv` | `id,date,store_nbr,item_nbr,onpromotion` | Official test dates and known promotion input |
| `items.csv` | `item_nbr,family,class,perishable` | Item attributes |
| `stores.csv` | `store_nbr,city,state,type,cluster` | Store and location attributes |
| `oil.csv` | `date,dcoilwtico` | Daily oil-price series |
| `holidays_events.csv` | `date,type,locale,locale_name,description,transferred` | Holiday/event attributes |
| `transactions.csv` | `date,store_nbr,transactions` | Daily store transaction counts |
| `sample_submission.csv` | `id,unit_sales` | Competition submission template |

## Preprocessing and features

The verified panel contains 195,692 daily rows from 2013-01-02 through 2017-08-15: 39 items, 3 stores, 116 item-store series, 153,766 positive-sales rows, no missing cells in the saved profile, and no duplicate series-date keys.

The ingestion code ranks item IDs by nonnegative sales strictly before the exclusive sample-selection cutoff. The cutoff is derived from the official test start and configured chronological windows; this run uses 2017-06-21, before selection, calibration, and scored evaluation. It then selects configured stores, aggregates duplicate date/store/item rows, builds a daily grid, and fills absent sales with zero. Promotion values are normalized to Boolean. Oil prices are forward-filled and leading gaps use a panel-level fallback. Transactions are forward-filled by store. The holiday feature includes National events that are not marked transferred. Product and store categories use integer codes.

The LightGBM feature set includes target lags at 1, 7, 14, and 28 days; prior-sales rolling mean over 7 and 28 days and standard deviation over 28 days; day-of-week, month, day, and week-of-year; promotion and holiday flags; lagged oil and transactions; and item/store metadata. Rolling features are shifted so the target date does not contribute to its predictors. Sample membership is now isolated from every configured selection, calibration, and evaluation window. The cutoff, sample size, store list, and selected item IDs are recorded in `data/processed/favorita_panel.manifest.json`; legacy caches without a matching manifest are rebuilt.

## Actual evaluation artifacts

The output JSON and prediction CSVs are in `outputs/metrics/` and `outputs/forecasts/`. The following figures are copied from those saved run outputs.

### Primary validation: 116 item-store series

| Model | WAPE | RMSSE, mean by series | RMSE |
|---|---:|---:|---:|
| Naive Weekly | 40.9416% | 0.691895 | 21.499179 |
| LightGBM | 29.5346% | 0.648599 | 14.649477 |

There are 3,248 observations from 2017-07-19 to 2017-08-15. Selection is 2017-06-21 to 2017-07-04; calibration is 2017-07-05 to 2017-07-18. The nominal 90% interval covered 90.4249% of the primary holdout.

### Common-model validation: six series

| Model | WAPE | RMSSE, mean by series | RMSE |
|---|---:|---:|---:|
| Naive Weekly | 24.7320% | 0.772054 | 51.084838 |
| LightGBM | 15.6490% | 0.470417 | 29.465924 |
| Prophet | 20.9164% | 0.620779 | 39.298949 |
| Darts Regression | 18.3903% | 0.552134 | 35.399453 |

Each model uses the same 168 held-out observations, 2017-07-19 to 2017-08-15. These metrics are not directly comparable to the primary table because the series sample differs. TFT is pending GPU training and has no score.

WAPE is the total absolute error divided by total absolute actual sales. RMSSE is calculated per series using the pre-holdout mean squared first difference as scale, then averaged across series. RMSE is calculated in unit-sales/day units.

## Inventory artifacts and assumptions

`outputs/recommendations/orders.csv` contains 110 item-store recommendations. All 110 solver statuses are `OPTIMAL`. Saved totals are 54,814 recommended units, 1,086.45 modeled holding-cost units, 4,942.50 modeled stockout-cost units, and 6,028.95 objective-cost units. The current order-up-to policy comparison is 82,453 units and 11,556.75 modeled cost units, a 47.83% modeled reduction under the same scenario assumptions.

Costs use generic cost units, not currency. The model assumes zero on-hand inventory unless an optional CSV with `item_nbr,store_nbr,on_hand` is configured. Default costs are 0.20 per leftover unit and 2.00 per unmet unit, and service level is 90%. This comparison is not realized business savings.

## Output contracts

| Path | Format and key fields |
|---|---|
| `outputs/forecasts/forecast.csv` | Forecast-level rows: `date,id,store_nbr,item_nbr,forecast,lower,upper` |
| `outputs/forecasts/validation.csv` | Primary held-out predictions: actual, LightGBM prediction, selected forecast and interval |
| `outputs/forecasts/model_comparison_validation.csv` | Common-sample actual and per-model prediction columns |
| `outputs/forecasts/hierarchical_forecast.csv` | Forecasts aggregated to item-store, item, store, city, state and overall |
| `outputs/forecasts/sales_history.csv` | Historical `date,store_nbr,item_nbr,sales` |
| `outputs/metrics/model_metrics.json` | Primary model metrics, selection scores, interval definition and coverage |
| `outputs/metrics/model_comparison.json` | Common-sample metrics, date windows, model execution records and interval coverage |
| `outputs/metrics/data_profile.json` | Dataset dimensions, date extent, missing-value counts and duplicate count |
| `outputs/recommendations/orders.csv` | Solver status, demand scenarios, bounds, order, stock and cost estimates |

## Dashboard/API behavior

This repository does **not** expose a standalone REST or prediction API. The application is a Plotly Dash web app:

- Start it from the repository root with `python app.py`.
- The browser UI is served at `http://127.0.0.1:8050/` by default.
- Dash serves its application layout and assets and uses internal Dash callback routes for filter and tab updates; those callback routes are an implementation detail, not a stable public API.
- The app reads the generated forecast, history, validation, recommendation CSVs and model comparison JSON at startup from relative `outputs/` paths. The held-out metrics panel also verifies saved predictions against `data/processed/favorita_panel.csv`.
- All six tabs use these files: Executive Overview, Sales Trends, Demand Forecasts, Model Performance, Inventory Optimization, and Store and Item Analysis.
- Store/item/date controls filter forecast views. The model comparison panel always describes its saved six-series, 28-day benchmark. Inventory orders cover the generated 16-day horizon; the dashboard's date selector does not recalculate or shorten the optimization horizon.

The dashboard expects the pipeline artifacts to exist. From a fresh clone, download Favorita files, install dependencies, run the pipeline, then launch the app. Large datasets and generated artifacts are Git-ignored, so they are not included in a clean public checkout.
