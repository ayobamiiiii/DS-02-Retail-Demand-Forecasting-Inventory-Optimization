"""Favorita training, rolling holdout evaluation, recursive forecast, and replenishment."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from math import ceil, sqrt
from src.data.favorita import load_favorita,make_daily_panel,add_lagged_covariates
from src.features.favorita import build,FEATURES
from src.models.lightgbm_model import create_model
from src.evaluation.metrics import score
from src.forecasting.intervals import residual_radius,add_interval
from src.forecasting.hierarchy import rollup_favorita
from src.optimization.orders import optimize_order
from src.evaluation.model_comparison import run_model_comparison

def _future_exog(testday,panel,items,stores,oil,holidays,transactions):
    """Build future-known test rows using the official Favorita test table."""
    x=testday[["date","store_nbr","item_nbr","onpromotion"]].copy()
    x["id"]=x.store_nbr.astype(str)+"_"+x.item_nbr.astype(str)
    known=set(panel.id.unique())
    x=x[x.id.isin(known)].copy()
    x["onpromotion"]=x.onpromotion.fillna(False).astype("int8")
    x=x.merge(items[["item_nbr","family","class","perishable"]],on="item_nbr",how="left")
    x=x.merge(stores[["store_nbr","state","type","cluster"]],on="store_nbr",how="left")
    # Apply category encoding learned from training data (same sorted vocabulary).
    for c,source in (("family",items.family),("state",stores.state),("type",stores.type)):
        cats=sorted(source.dropna().unique().tolist()); x[c]=pd.Categorical(x[c],categories=cats).codes.astype("int16")
    x["store_type"]=x.type.astype("int16")
    origin=panel.date.max()
    oil_known=oil[oil.date<=origin].sort_values("date").set_index("date").dcoilwtico.ffill()
    last_oil=float(oil_known.dropna().iloc[-1]) if oil_known.notna().any() else float(panel.oil_price.iloc[-1])
    # Later dates are forecast-origin unknown; hold the last observed oil value.
    x["oil_price"]=last_oil; x["oil_price_lag_1"]=last_oil
    hol=holidays[(holidays.locale=="National")&(~holidays.transferred.astype(bool))].groupby("date").size().gt(0).astype("int8").rename("is_holiday")
    x=x.join(hol,on="date"); x["is_holiday"]=x.is_holiday.fillna(0).astype("int8")
    last_tx=transactions[transactions.date<=origin].sort_values("date").groupby("store_nbr").tail(1)[["store_nbr","transactions"]].rename(columns={"transactions":"transactions_lag_1"})
    x=x.merge(last_tx,on="store_nbr",how="left"); x["transactions_lag_1"]=x.transactions_lag_1.fillna(0); x["unit_sales"]=0.0
    return x

def selection_cutoff_from_test_start(test_start,config):
    """Return the earliest exclusive date for all evaluation-window sample choice."""
    test_start=pd.Timestamp(test_start)
    primary_start=test_start-pd.Timedelta(days=int(config.get("validation_days",28)))
    comparison=config.get("model_comparison",{})
    comparison_start=test_start-pd.Timedelta(days=sum(int(comparison.get(k,default)) for k,default in
        (("evaluation_days",28),("calibration_days",14),("selection_days",14))))
    return min(primary_start,comparison_start)

def run(config):
    out=Path(config.get("output_dir","outputs")); root=Path(config.get("data_dir","data/raw/favorita"))
    for sub in ("figures","forecasts","metrics","recommendations","models"): (out/sub).mkdir(parents=True,exist_ok=True)
    cache=Path("data/processed/favorita_panel.csv")
    test_dates=pd.read_csv(root/"test.csv",usecols=["date"],parse_dates=["date"]).date
    test_start=pd.Timestamp(test_dates.min())
    selection_cutoff=selection_cutoff_from_test_start(test_start,config)
    panel_manifest=cache.with_suffix(".manifest.json")
    expected_manifest={"selection_cutoff_exclusive":str(selection_cutoff.date()),
      "sample_items":int(config.get("sample_items",40)),"stores":sorted(config.get("stores",[1,2,3]))}
    panel=None
    if cache.exists() and panel_manifest.exists() and config.get("reuse_processed_panel",True):
        cached=pd.read_csv(cache,parse_dates=["date"])
        requested=set(config.get("stores",[1,2,3]))
        try: prior_manifest=json.loads(panel_manifest.read_text())
        except (OSError,json.JSONDecodeError): prior_manifest={}
        if (set(cached.store_nbr.unique())==requested and
            all(prior_manifest.get(k)==v for k,v in expected_manifest.items()) and
            set(cached.item_nbr.unique())==set(prior_manifest.get("selected_items",[]))): panel=cached
    if panel is not None:
        test=pd.read_csv(root/"test.csv",usecols=["date","store_nbr","item_nbr","onpromotion"],parse_dates=["date"],dtype={"onpromotion":"string"})
        test["onpromotion"]=test.onpromotion.astype("string").str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False)
        test=test[test.item_nbr.isin(panel.item_nbr.unique())&test.store_nbr.isin(panel.store_nbr.unique())]
        items=pd.read_csv(root/"items.csv"); stores=pd.read_csv(root/"stores.csv")
        oil=pd.read_csv(root/"oil.csv",parse_dates=["date"]); holidays=pd.read_csv(root/"holidays_events.csv",parse_dates=["date"])
        transactions=pd.read_csv(root/"transactions.csv",parse_dates=["date"])
    else:
        train,test,items,stores,oil,holidays,transactions=load_favorita(root,config.get("sample_items",40),config.get("stores",[1,2,3]),config.get("chunksize",1_500_000),selection_cutoff)
        panel=make_daily_panel(train,items,stores,oil,holidays,transactions,config.get("stores",[1,24,37]))
    panel=add_lagged_covariates(panel,transactions)
    panel.to_csv(cache,index=False)
    panel_manifest.write_text(json.dumps({**expected_manifest,"selected_items":sorted(map(int,panel.item_nbr.unique()))},indent=2))
    profile={"dataset":"Corporación Favorita Grocery Sales Forecasting","nonzero_sales_panel_days":int((panel.unit_sales>0).sum()),"panel_rows":int(len(panel)),
      "series":int(panel.id.nunique()),"items":int(panel.item_nbr.nunique()),"stores":int(panel.store_nbr.nunique()),
      "states":int(panel.state.nunique()),"date_min":str(panel.date.min().date()),"date_max":str(panel.date.max().date()),
      "missing_values":{k:int(v) for k,v in panel.isna().sum().items()},"duplicate_series_days":int(panel.duplicated(["id","date"]).sum())}
    (out/"metrics"/"data_profile.json").write_text(json.dumps(profile,indent=2))
    history_daily=panel.groupby(["date","store_nbr","item_nbr"],as_index=False).unit_sales.sum().rename(columns={"unit_sales":"sales"})
    history_daily.to_csv(out/"forecasts"/"sales_history.csv",index=False)
    cutoff=panel.date.max()-pd.Timedelta(days=int(config.get("validation_days",28))-1)
    features=build(panel); feature_train=features[(features.date<cutoff)].dropna(subset=FEATURES).copy()
    validation=features[features.date>=cutoff].dropna(subset=FEATURES).copy()
    if feature_train.empty or validation.empty: raise ValueError("Insufficient Favorita history for requested validation window and 28-day lags")
    calibration_days=int(config.get("model_comparison",{}).get("calibration_days",14))
    selection_days=int(config.get("model_comparison",{}).get("selection_days",14))
    calibration_start=cutoff-pd.Timedelta(days=calibration_days)
    selection_start=calibration_start-pd.Timedelta(days=selection_days)
    selection_train=features[features.date<selection_start].dropna(subset=FEATURES).copy()
    selection=features[features.date.between(selection_start,calibration_start-pd.Timedelta(days=1))].dropna(subset=FEATURES).copy()
    if selection_train.empty or selection.empty: raise ValueError("Insufficient history for disjoint selection/calibration periods")
    selection_by_id={k:g.unit_sales.to_numpy() for k,g in panel[panel.date<selection_start].groupby("id",observed=True)}
    selection["prediction_naive"]=selection.lag_7.clip(lower=0)
    selection_scores={"naive_weekly":score(selection.rename(columns={"unit_sales":"sales","prediction_naive":"prediction"}),"prediction",selection_by_id)}
    selection_model=create_model(config.get("random_seed",41)); selection_model.fit(selection_train[FEATURES],selection_train.unit_sales)
    selection["prediction_lgbm"]=np.maximum(selection_model.predict(selection[FEATURES]),0)
    selection_scores["lightgbm"]=score(selection.rename(columns={"unit_sales":"sales","prediction_lgbm":"prediction"}),"prediction",selection_by_id)
    best_name=min(selection_scores,key=lambda k:selection_scores[k]["wape"])
    calibration=features[features.date.between(calibration_start,cutoff-pd.Timedelta(days=1))].dropna(subset=FEATURES).copy()
    if calibration.empty: raise ValueError("Calibration period has no lag-complete observations")
    calibration_residuals=calibration[["id"]].copy()
    if best_name=="lightgbm":
        calibration_train=features[features.date<calibration_start].dropna(subset=FEATURES)
        calibration_model=create_model(config.get("random_seed",41)); calibration_model.fit(calibration_train[FEATURES],calibration_train.unit_sales)
        calibration["prediction"]=np.maximum(calibration_model.predict(calibration[FEATURES]),0)
    else:
        calibration["prediction"]=calibration.lag_7.clip(lower=0)
    calibration_residuals["residual"]=calibration.unit_sales.to_numpy()-calibration.prediction.to_numpy()
    radius=residual_radius(calibration_residuals.residual,config.get("interval_level",.90))
    train_by_id={k:g.unit_sales.to_numpy() for k,g in panel[panel.date<cutoff].groupby("id",observed=True)}
    validation["prediction_naive"]=validation.lag_7.clip(lower=0)
    naive_frame=validation.rename(columns={"unit_sales":"sales","prediction_naive":"prediction"})
    metrics={"naive_weekly":score(naive_frame,"prediction",train_by_id)}
    model=create_model(config.get("random_seed",41)); model.fit(feature_train[FEATURES],feature_train.unit_sales)
    validation["prediction_lgbm"]=np.maximum(model.predict(validation[FEATURES]),0)
    lgb_frame=validation.rename(columns={"unit_sales":"sales","prediction_lgbm":"prediction"})
    metrics["lightgbm"]=score(lgb_frame,"prediction",train_by_id)
    chosen_prediction=validation["prediction_lgbm"] if best_name=="lightgbm" else validation.lag_7.clip(lower=0)
    residual=validation.unit_sales-chosen_prediction
    val_output=validation[["date","id","store_nbr","item_nbr","unit_sales","prediction_lgbm","lag_7"]].copy()
    val_output["forecast"]=validation.prediction_lgbm if best_name=="lightgbm" else validation.lag_7.clip(lower=0)
    val_output=val_output.rename(columns={"unit_sales":"actual"})
    val_output=add_interval(val_output,radius); val_output.to_csv(out/"forecasts"/"validation.csv",index=False)
    metrics["interval"]={"method":"selected-model absolute residual quantile; disjoint calibration and untouched evaluation periods","level":config.get("interval_level",.90),"radius_units":radius,
      "calibration_period_start":str(calibration_start.date()),"calibration_through":str((cutoff-pd.Timedelta(days=1)).date()),
      "coverage_period_from":str(cutoff.date()),"coverage_period_through":str(validation.date.max().date()),
      "coverage":float((np.abs(residual)<=radius).mean()),"coverage_rows":int(len(validation))}
    metrics["selected_model"]={"name":best_name,"selection_metric":"WAPE on disjoint selection period",
      "selection_period_start":str(selection_start.date()),"selection_period_end":str((calibration_start-pd.Timedelta(days=1)).date()),
      "selection_scores":selection_scores}
    residual_sigma=calibration_residuals.groupby("id",observed=True).residual.std(ddof=1)
    pooled_sigma=float(calibration_residuals.residual.std(ddof=1)) if len(calibration_residuals)>1 else 0.0
    # Use only IDs represented in official test rows and recursively extend actual training history.
    future_test=test.sort_values("date"); all_future=[]
    hist=panel.copy(); model.fit(build(hist).dropna(subset=FEATURES)[FEATURES],build(hist).dropna(subset=FEATURES).unit_sales)
    model.booster_.save_model(str(out/"models"/"lightgbm.txt"))
    for date,day in future_test.groupby("date",sort=True):
        exog=_future_exog(day,hist,items,stores,oil,holidays,transactions)
        hist=pd.concat([hist,exog],ignore_index=True,sort=False)
        today=build(hist); today=today[today.date.eq(date)].dropna(subset=FEATURES).copy()
        prediction=np.maximum(model.predict(today[FEATURES]),0) if best_name=="lightgbm" else today.lag_7.clip(lower=0).to_numpy()
        today["forecast"]=prediction
        by_id=dict(zip(today.id,prediction)); mask=hist.date.eq(date)
        hist.loc[mask,"unit_sales"]=hist.loc[mask,"id"].map(by_id).fillna(0).to_numpy()
        all_future.append(today[["date","id","store_nbr","item_nbr","forecast"]])
    forecast=pd.concat(all_future,ignore_index=True); forecast=add_interval(forecast,radius)
    forecast.to_csv(out/"forecasts"/"forecast.csv",index=False)
    rollup_favorita(forecast,stores).to_csv(out/"forecasts"/"hierarchical_forecast.csv",index=False)
    inv=config.get("inventory",{}); rec=[]
    on_hand_by_key={}
    on_hand_path=inv.get("on_hand_file")
    if on_hand_path and Path(on_hand_path).exists():
        stocks=pd.read_csv(on_hand_path)
        required={"item_nbr","store_nbr","on_hand"}
        if not required.issubset(stocks.columns): raise ValueError(f"on_hand_file must contain {sorted(required)}")
        stocks=stocks.groupby(["item_nbr","store_nbr"],as_index=False).on_hand.sum()
        on_hand_by_key={(int(r.item_nbr),int(r.store_nbr)):max(0,int(r.on_hand)) for r in stocks.itertuples()}
    service_level=float(inv.get("service_level",0.90)); max_order=int(inv.get("max_order",2000)); max_inventory=int(inv.get("max_inventory",5000))
    horizon_days=max(1,int(forecast.date.nunique()))
    for (item,store),g in forecast.groupby(["item_nbr","store_nbr"],observed=True):
        ident=g.id.iloc[0]; mean_demand=float(g.forecast.sum())
        daily_sigma=float(residual_sigma.get(ident,np.nan))
        if not np.isfinite(daily_sigma): daily_sigma=pooled_sigma
        horizon_sigma=max(0.0,daily_sigma*sqrt(horizon_days))
        on_hand=on_hand_by_key.get((int(item),int(store)),int(inv.get("default_on_hand",0)))
        # Reproduce the previous order-up-to policy for a like-for-like policy comparison.
        old_buffer=max(0,int(ceil(float(g.upper.sum())-mean_demand)))
        old_target=int(round(mean_demand))+old_buffer
        current_order=max(0,min(max_order,old_target-on_hand))
        rec.append({"item_nbr":int(item),"store_nbr":int(store),
          **optimize_order(mean_demand=mean_demand,demand_std=horizon_sigma,on_hand=on_hand,
            holding_cost=inv.get("unit_holding_cost",0.20),stockout_cost=inv.get("unit_stockout_cost",2.0),
            service_level=service_level,max_order=max_order,max_inventory=max_inventory,baseline_order=current_order)})
    pd.DataFrame(rec).to_csv(out/"recommendations"/"orders.csv",index=False)
    (out/"metrics"/"model_metrics.json").write_text(json.dumps(metrics,indent=2))
    # The multi-model experiment has its own sample, disjoint selection/calibration/
    # evaluation blocks, and execution log. Keep core pipeline outputs if an optional
    # framework fails on the host, and record that failure for the audit.
    try:
        comparison_metrics,_,_=run_model_comparison(panel,config,out)
        metrics["benchmark_execution"]="completed"
    except Exception as exc:
        metrics["benchmark_execution"]={"status":"failed","error":f"{type(exc).__name__}: {exc}"}
    (out/"metrics"/"model_metrics.json").write_text(json.dumps(metrics,indent=2))
    try:
        import matplotlib.pyplot as plt
        ax=history_daily.set_index("date").sales.plot(figsize=(12,4),title="Favorita sales in sampled series"); ax.set_ylabel("Units"); plt.tight_layout(); plt.savefig(out/"figures"/"sales_trend.png"); plt.close()
    except ImportError: pass
    report={"dataset":"Favorita","cutoff":str(cutoff.date()),"profile":profile,"metrics":metrics,"forecast_rows":int(len(forecast)),"recommendation_rows":len(rec)}
    print(json.dumps(report,indent=2)); return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/default.yaml"); a=ap.parse_args()
    with open(a.config) as f: run(yaml.safe_load(f))
if __name__=="__main__": main()
