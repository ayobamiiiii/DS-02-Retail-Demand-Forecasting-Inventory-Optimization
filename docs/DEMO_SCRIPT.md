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
3. Use the default selection of stores/items and the full forecast date range. The dashboard's forecast artifact covers August 16–31, 2017, for 110 item-store pairs.
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

**Demonstrate:** Headline forecast total, store count, recommended order units, and the forecast-with-interval line chart. With the full default selection, the saved forecast has 51,840.2 total units across 110 pairs and 16 dates; the inventory recommendations total 54,814 units.

**Say:**

“The overview summarizes the current filter selection. This forecast covers August 16 through August 31, 2017. Across the 110 item-store pairs, predicted demand sums to about 51,840 units. The optimized policy recommends 54,814 units across the horizon; that can exceed the point forecast because the order model includes safety stock and scenario costs.”

### 1:05–1:40 — Sales Trends

**Click:** Click **Sales Trends**.

**Demonstrate:** Historical sales curve. Point out the overall pattern and any visible weekly variation; avoid claiming a causal seasonal effect from the chart alone.

**Say:**

“This view uses the processed historical sales file. It lets a reviewer inspect the selected items and stores before looking at future forecasts. The source panel is daily and zero-fills missing item-store dates, so the curve should be interpreted as the project’s prepared sample rather than a complete retailer-wide ledger.”

### 1:40–2:20 — Demand Forecasts

**Click:** Click **Demand Forecasts**.

**Demonstrate:** Forecast line and lower/upper interval lines; move the pointer over a date to show the plotted values.

**Say:**

“This tab shows the 16-day recursive forecast and the empirical interval bounds. The interval radius was calibrated on a separate 14-day residual period. On the primary 116-series holdout, nominal 90% coverage was 90.42%. That is marginal coverage on this sample, not a guarantee for every item or store.”

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
