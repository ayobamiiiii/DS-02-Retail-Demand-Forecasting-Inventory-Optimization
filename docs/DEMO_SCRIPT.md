# Dashboard Screen-Recording Script

**Target duration:** 6 minutes

**Demo type:** Local screen recording. The dashboard is currently verified locally; no public deployment or recording file is included with this repository.

## Before recording

1. From the project root, activate the environment and start the app:
   ```bash
   source .venv/bin/activate
   python app.py
   ```
   On Windows, activate the environment using its Windows activation script.
2. Open `http://127.0.0.1:8050/` in a browser.
3. Use the default selection of stores/items. The global date picker spans history and the future forecast; its default begins one year before forecast origin. The saved future forecast covers August 16–31, 2017, for 110 item-store pairs.
4. Hide unrelated windows and browser bookmarks. Keep the app header and global Store, Item, and Date range filters visible.
5. Do not rerun the pipeline during the recording. The current generated outputs are the evidence shown below.

## Timed walkthrough

### 0:00–0:20 — Opening

**Click:** No click; show the dashboard at its initial page.

**Demonstrate:** App title and the Favorita subtitle.

**Say:**

“This is my Favorita retail demand forecasting and inventory optimization project. It forecasts daily item-store sales, compares forecasting models on held-out dates, and turns the forecast into constrained replenishment recommendations. The verified run covers 116 series across three stores, not the entire retailer network.”

### 0:20–1:05 — Executive Overview

**Click:** Click the **Executive Overview** tab.

**Demonstrate:** Forecast units, selected store count, recommended order units, and modeled inventory cost. The overview intentionally has no detailed chart; use the other tabs for historical and forecast curves. With the default filters, the saved forecast totals 51,840.2 units across 110 pairs and 16 dates; recommendations total 54,814 units.

**Say:**

“The overview keeps the decision summary compact. Across the selected 110 item-store pairs, the August 16–31, 2017 forecast sums to about 51,840 units. The optimizer recommends 54,814 units. The difference reflects safety stock and the configured demand scenarios; these are recommendation units, not additional realized sales.”

### 1:05–1:40 — Sales Trends

**Click:** Click **Sales Trends**.

**Demonstrate:** The actual-only daily sales line, average actual sales by weekday chart, store-comparison trend, and four historical KPIs. The default range is the latest year of observed sales plus the future forecast window; use the date picker to widen or narrow it. Point out visible weekday differences without claiming causality.

**Say:**

“Every value in this tab comes from observed Favorita sales, not model predictions. It reports total actual units, average actual units per day, the peak sales day, and store count. The charts show daily totals, average by weekday, and actual sales by selected store; absent item-store dates were zero-filled during panel preparation, so this is the prepared sample rather than a complete retailer-wide ledger.”

### 1:40–2:20 — Demand Forecasts

**Click:** Click **Demand Forecasts**.

**Demonstrate:** Historical actuals, actual versus saved model predictions over the July 19–August 15 validation period, and the August 16–31 future forecast with its lower/upper interval. Hover dates to distinguish actual, validation prediction, and future forecast values.

**Say:**

“This tab separates three things: historical actual sales before validation, saved actual and one-day-ahead predictions for July 19–August 15, and the 16-day recursive forecast for August 16–31. The future interval was calibrated using a separate 14-day residual period. Primary nominal 90% coverage was 90.42% on 3,248 held-out rows. It is marginal coverage on this sample, not a guarantee for each item or store. Future dates have predictions but no observed actuals.”

### 2:20–3:20 — Model Performance

**Click:** Click **Model Performance**.

**Demonstrate:** Best WAPE, RMSSE, and RMSE KPIs; three interactive comparison charts; scope label; metric-verification message; selection/calibration/evaluation periods. This tab uses the six-series common-sample output.

**Say:**

“This comparison uses the same six item-store series and 168 observations for every completed model. LightGBM has the lowest WAPE at 15.65%, RMSSE at 0.470, and RMSE at 29.466. Naive Weekly, Prophet, and Darts Regression are also shown. The dashboard recomputes the metrics from the saved held-out predictions and the pre-validation history; the status confirms that check passed. Separately, the primary 116-series LightGBM WAPE is 29.53%, compared with 40.94% for Naive Weekly. The two tables use different series samples, so I do not compare their values directly. TFT is not scored because GPU training is pending.”

### 3:20–4:35 — Inventory Optimization

**Click:** Click **Inventory Optimization**.

**Demonstrate:** Total recommended units, modeled holding cost, stockout cost, total objective cost, forecast/inventory/order chart, and current-policy versus optimized-policy cost chart. Show the recommendations table and point to solver status.

**Say:**

“The optimizer uses OR-Tools CP-SAT and returned OPTIMAL for all 110 recommendations. It recommends 54,814 units. Under the configured demand scenarios and proxy costs, estimated holding cost is 1,086.45, shortage cost is 4,942.50, and total objective cost is 6,028.95 cost units. The current order-up-to comparison is 82,453 units and 11,556.75 modeled cost units. The resulting 47.83% cost reduction is a modeled comparison, not realized savings. Favorita does not provide on-hand inventory or actual carrying and shortage costs; the default assumes zero stock and uses 0.20/2.00 proxy costs. The demand bar includes the safety-stock allowance.”

**Important interaction note:** The dashboard’s date selector filters forecast views, but it does not recompute order recommendations for a shorter horizon. Present the inventory figures as the full generated forecast-horizon policy.

### 4:35–5:20 — Store and Item Analysis

**Click:** Click **Store and Item Analysis**.

**Demonstrate:** Top forecast store-item combinations. Use the Store dropdown to clear the current selection, then select store **24**; show the list update. Optionally select item **1473474** in the Item dropdown to narrow further.

**Say:**

“This view ranks store-item combinations by forecast demand under the selected filters. The store and item controls make it possible to inspect a smaller planning slice without changing the saved model evaluation protocol.”

### 5:20–6:00 — Close and caveats

**Click:** No click; leave the analysis view visible.

**Demonstrate:** The bounded three-store sample, if the header is visible, or return to Model Performance if you want the metric evidence on screen.

**Say:**

“The main result is a working forecasting-to-inventory workflow with completed baseline, LightGBM, Prophet, and Darts evaluation. The main limitations are the small store/item sample, assumed inventory economics, and empirical interval coverage. Sample selection now uses sales only before June 21, 2017, ahead of every selection, calibration, and evaluation window. TFT remains pending a successful Colab GPU run. Next I would expand the series scope, add real inventory and cost inputs, and complete the GPU evaluation.”

## Verified figures shown

- Forecast: 1,760 rows, 110 pairs, 16 dates; total point forecast 51,840.2 units.
- Primary evaluation: 116 series, 3,248 rows, July 19–August 15, 2017.
- Common-model evaluation: six series, 168 rows, July 19–August 15, 2017.
- Optimizer: 110/110 statuses `OPTIMAL`; 54,814 recommended units; 6,028.95 modeled objective cost units.
- TFT: pending GPU training; no performance result.
