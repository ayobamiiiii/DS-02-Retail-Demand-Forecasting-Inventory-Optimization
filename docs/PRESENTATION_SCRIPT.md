# 10-Minute Presentation Script

**Deck:** `docs/DS02_Final_Presentation.pptx`

**Presenter:** Misbaudeen Abdurrahman

**Target length:** 10 minutes at a deliberate pace of roughly 95–100 spoken words per minute, with pauses to let the charts and diagram register. The deck includes slide-by-slide notes; use this timed script for the full narration.

## Slide 1 — Project and scope (0:00–0:40)

“Hello, I’m Misbaudeen Abdurrahman. This project connects retail demand forecasting to inventory decision support using the Corporación Favorita grocery dataset. The title says nationwide demand forecasting, but the verified experiment is deliberately bounded: 39 items, 116 item-store series, and three stores across three provinces. Today I’ll cover the business problem, system design, measured model results, inventory assumptions, and what still needs work.”

## Slide 2 — Business problem (0:40–1:40)

“Retail planners have to estimate what will sell by product and store. Forecasts that are too low can contribute to shortages; forecasts that are too high can leave excess stock and tie up capacity. The project objective is to build a repeatable daily forecast, compare models on held-out data, show the uncertainty, and convert it into an order quantity with explicit capacity and cost assumptions. This is decision support, not an automated purchase system.”

## Slide 3 — Dataset and preparation (1:40–2:40)

“The data is Corporación Favorita, not Walmart M5. The source files include daily training sales, an official test table, item and store attributes, promotions, holidays, oil prices, and transactions. The current run filters to stores 1, 24, and 37 and 39 selected items. The processed panel has 195,692 daily rows. Duplicate series-days are aggregated, missing dates in each daily series are filled with zero, and features include prior sales lags, shifted rolling summaries, calendar fields, promotions, holidays, metadata, and prior-day oil and transaction values.”

“Item selection now ranks sales only before the exclusive June 21, 2017 cutoff, which is earlier than selection, calibration, and scored evaluation. The common benchmark also ranks series before its selection window. A provenance manifest invalidates a legacy panel cache, and tests verify that post-cutoff sales cannot affect either sample.”

## Slide 4 — Architecture (2:40–3:40)

“This diagram follows the data from raw CSVs through ingestion, features, chronological model evaluation, uncertainty calibration, test forecasting, hierarchy rollups, and inventory optimization. The command-line pipeline is in `src/pipeline.py`; the dashboard reads the generated files from `outputs/`. Bottom-up aggregation produces item-store, item, store, city, state, and overall totals. This repository has a Dash application, not a separately published REST prediction API.”

## Slide 5 — Models and evaluation design (3:40–4:50)

“The completed models are a previous-week Naive Weekly baseline, LightGBM, Prophet, and Darts Regression with a Ridge estimator. Prophet and Darts were run on the same six-series benchmark so their additional per-series runtime stayed manageable. The primary Naive and LightGBM evaluation covers 116 series.”

“For the common benchmark, selection uses June 21 through July 4, calibration uses July 5 through July 18, and evaluation uses July 19 through August 15. The interval is calibrated on absolute residuals from the calibration block, then measured on the later holdout. Primary forecasts are rolling one day ahead, with actual history through the previous date. The official test forecast is a separate recursive 16-day horizon. WAPE summarizes absolute error relative to total actual demand, RMSSE scales error against historical one-step changes, and RMSE remains in sales units. TFT has a Colab notebook and dataset handoff but no completed GPU training or score.”

## Slide 6 — Forecast results (4:50–6:00)

“On the primary 116-series holdout, Naive Weekly has WAPE 40.94%, RMSSE 0.692, and RMSE 21.499. LightGBM improves those to WAPE 29.53%, RMSSE 0.649, and RMSE 14.649.”

“In the separate common six-series comparison, LightGBM has WAPE 15.65%, RMSSE 0.470, and RMSE 29.466. Prophet records 20.92%, 0.621, and 39.299; Darts Regression 18.39%, 0.552, and 35.399; and Naive Weekly 24.73%, 0.772, and 51.085. LightGBM is lowest on all three metrics in this common sample. The two tables have different samples and should not be compared as if they were the same experiment. The primary nominal 90% interval covered 90.42% of its held-out observations.”

## Slide 7 — Inventory optimization (6:00–7:10)

“The optimizer uses OR-Tools CP-SAT to choose a bounded integer order quantity. It balances leftover inventory and unmet demand under low, median, and high demand scenarios weighted 25%, 50%, and 25%. Safety stock uses a 90% service-level normal quantile applied to forecast residual uncertainty, scaled by the square root of the 16-day horizon.”

“For the current 110 recommendations, every solver status is OPTIMAL. The system recommends 54,814 units, with modeled holding cost of 1,086.45 and stockout cost of 4,942.50, for total objective cost of 6,028.95. The baseline policy estimates 82,453 units and 11,556.75 cost units. The 47.83% modeled reduction depends on zero default on-hand stock and proxy costs of 0.20 holding and 2.00 shortage per unit. Those are not actual currency costs or realized savings.”

## Slide 8 — Dashboard (7:10–8:10)

“The local dashboard has six tabs: Executive Overview, Sales Trends, Demand Forecasts, Model Performance, Inventory Optimization, and Store and Item Analysis. Store, item, and date filters drive the forecast views. The model performance panel uses the six-series comparison output and verifies metrics from saved held-out predictions. The inventory panel compares the optimized recommendations with the current order-up-to policy. It does not recalculate order recommendations when the dashboard date filter changes; its figures cover the generated 16-day horizon. The Model Performance tab is intentionally tied to the saved common-sample evaluation, so filtering the forecast view does not silently change the benchmark.”

## Slide 9 — Challenges and limitations (8:10–9:05)

“The main challenges were keeping validation chronological, avoiding same-day sales and auxiliary-feature leakage, selecting series before evaluation windows, and making solver costs inspectable. The sample-selection cutoff is now isolated and tested. Important limitations remain: this is a small sample, intervals are constant-width empirical bands, inventory and costs are assumptions, and four orders reached the configured cap. The dashboard is locally verified, but the repository does not include a hosted deployment or screen recording. TFT remains pending GPU training.”

## Slide 10 — Next steps and close (9:05–10:00)

“My next steps are to expand the evaluation sample, add real stock, lead-time, and cost inputs, and check interval quality by store and item. I would then run the TFT notebook on Colab T4, evaluate only if the saved execution report confirms success, and prepare a screen recording or deployment for reviewers. The project demonstrates a complete local workflow, while its limitations set the boundary for the conclusions. Thank you.”

## Personal reflection prompt

For the internship submission, add a short personal example after Slide 10 or in the written reflection: one task you personally owned, one obstacle you encountered, and the evidence you used to decide the fix worked. Do not claim end-to-end authorship unless that matches your actual role.
