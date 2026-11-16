"""Independent, common-sample benchmark for traditional and optional deep models."""
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.evaluation.metrics import score
from src.features.favorita import build,FEATURES
from src.forecasting.intervals import residual_radius,add_interval
from src.models.lightgbm_model import create_model
from src.models.prophet_model import predict_prophet
from src.models.darts_model import predict_darts
from src.models.tft import predict_tft

PREDICTION_COLUMNS={"naive_weekly":"prediction_naive_weekly","lightgbm":"prediction_lightgbm",
                    "prophet":"prediction_prophet","darts_regression":"prediction_darts_regression","tft":"prediction_tft"}

def _choose_common_ids(panel,limit):
    totals=panel.groupby(["store_nbr","id"],as_index=False,observed=True).unit_sales.sum()
    stores=sorted(totals.store_nbr.unique()); per_store=max(1,int(np.ceil(int(limit)/max(len(stores),1))))
    selected=[]
    for store in stores:
        top=totals[totals.store_nbr.eq(store)].nlargest(per_store,"unit_sales").id.tolist()
        selected.extend(top)
    return list(dict.fromkeys(selected))[:int(limit)]

def _future_rows(panel,series_ids,start,end,origin):
    cols=["id","date","store_nbr","item_nbr","onpromotion","is_holiday","oil_price_lag_1","transactions_lag_1",
          "family","class","perishable","state","cluster","type"]
    fut=panel[panel.id.isin(series_ids)&panel.date.between(start,end)][cols].copy()
    oil_known=panel[panel.date<=origin].sort_values("date").groupby("date",observed=True).oil_price.last()
    last_oil=float(oil_known.iloc[-1]) if len(oil_known) else 0.0
    tx=panel[panel.date<=origin].sort_values("date").groupby(["store_nbr","date"],as_index=False,observed=True).transactions.last()
    tx=tx.sort_values("date").groupby("store_nbr",observed=True).tail(1).set_index("store_nbr").transactions
    fut["oil_price_lag_1"]=last_oil
    fut["transactions_lag_1"]=fut.store_nbr.map(tx).fillna(0)
    # Only known calendar/exogenous values remain; held-out sales never enter model inputs.
    return fut.sort_values(["id","date"]).reset_index(drop=True)

def _weekly_forecast(history,future):
    values={k:g.sort_values("date").unit_sales.astype(float).tolist() for k,g in history.groupby("id",observed=True)}
    outputs=[]
    for day in sorted(future.date.unique()):
        rows=future[future.date.eq(day)]
        preds=[]
        for ident in rows.id:
            seq=values[str(ident)]
            p=max(0.0,seq[-7] if len(seq)>=7 else (seq[-1] if seq else 0.0))
            preds.append(p); seq.append(p)
        outputs.append(pd.DataFrame({"id":rows.id.to_numpy(),"date":rows.date.to_numpy(),"prediction":preds}))
    return pd.concat(outputs,ignore_index=True)

def _lightgbm_forecast(history,future,seed):
    train_features=build(history).dropna(subset=FEATURES)
    if train_features.empty: raise ValueError("No lag-complete rows for LightGBM")
    model=create_model(seed); model.fit(train_features[FEATURES],train_features.unit_sales)
    hist=history.copy(); outputs=[]
    for day in sorted(future.date.unique()):
        today=future[future.date.eq(day)].copy(); today["unit_sales"]=0.0
        hist=pd.concat([hist,today],ignore_index=True,sort=False)
        features=build(hist); target=features[features.date.eq(day)].dropna(subset=FEATURES)
        pred=np.maximum(model.predict(target[FEATURES]),0)
        target=target[["id","date"]].copy(); target["prediction"]=pred
        by_id=dict(zip(target.id,pred)); mask=hist.date.eq(day)
        hist.loc[mask,"unit_sales"]=hist.loc[mask,"id"].map(by_id).fillna(0).to_numpy()
        outputs.append(target)
    return pd.concat(outputs,ignore_index=True)

def _predict(name,history,future,config):
    dates=pd.DatetimeIndex(sorted(future.date.unique()))
    if name=="naive_weekly": return _weekly_forecast(history,future),{}
    if name=="lightgbm": return _lightgbm_forecast(history,future,config.get("random_seed",41)),{}
    if name=="prophet": return predict_prophet(history,dates,iterations=config.get("prophet_iterations",200)),{}
    if name=="darts_regression": return predict_darts(history,dates),{"library":"Darts RegressionModel","lags":28}
    if name=="tft": return predict_tft(history,future[["id","date"]],max_epochs=config.get("tft_max_epochs",1),
        batch_size=config.get("tft_batch_size",64),limit_train_batches=config.get("tft_train_batches",10))
    raise ValueError(f"Unknown model {name}")

def run_model_comparison(panel,config,output_dir="outputs"):
    """Select on early validation, calibrate on the next block, evaluate on a final holdout."""
    out=Path(output_dir); (out/"metrics").mkdir(parents=True,exist_ok=True); (out/"forecasts").mkdir(parents=True,exist_ok=True)
    cfg=config.get("model_comparison",{})
    evaluation_days=int(cfg.get("evaluation_days",28)); calibration_days=int(cfg.get("calibration_days",14)); selection_days=int(cfg.get("selection_days",14))
    max_date=panel.date.max(); eval_start=max_date-pd.Timedelta(days=evaluation_days-1)
    calibration_start=eval_start-pd.Timedelta(days=calibration_days)
    selection_start=calibration_start-pd.Timedelta(days=selection_days)
    selection_end=calibration_start-pd.Timedelta(days=1); calibration_end=eval_start-pd.Timedelta(days=1)
    if selection_start<=panel.date.min()+pd.Timedelta(days=60): raise ValueError("Insufficient history for requested selection/calibration/evaluation windows")
    ids=_choose_common_ids(panel,cfg.get("series_limit",12))
    models=["naive_weekly","lightgbm","prophet","darts_regression"]
    include_tft=cfg.get("include_tft",True)
    if include_tft: models.append("tft")
    execution={m:{"status":"pending","stages":{}} for m in models}
    if not include_tft:
        execution["tft"]={"status":"not_run","reason":"TFT was attempted separately; see the project audit for the runtime outcome."}
    selection_predictions={}; calibration_predictions={}; evaluation_predictions={}
    def save_progress():
        (out/"metrics"/"model_comparison_progress.json").write_text(json.dumps({"execution":execution},indent=2,default=str))
    stage_windows={"selection":(selection_start,selection_end),"calibration":(calibration_start,calibration_end),"evaluation":(eval_start,max_date)}
    for stage,(start,end) in stage_windows.items():
        origin=start-pd.Timedelta(days=1)
        history=panel[panel.id.isin(ids)&panel.date.le(origin)].copy()
        future=_future_rows(panel,ids,start,end,origin)
        if future.empty: raise ValueError(f"No future rows available for {stage} benchmark window")
        for name in models:
            if execution[name]["status"]=="failed": continue
            try:
                print(f"Independent benchmark: {stage} / {name}",flush=True)
                pred,diag=_predict(name,history,future,config)
                joined=pred.merge(panel[["id","date","unit_sales"]],on=["id","date"],how="left",validate="one_to_one")
                if joined.unit_sales.isna().any(): raise ValueError("Missing held-out actuals for benchmark predictions")
                joined=joined.rename(columns={"unit_sales":"actual"})
                if stage=="selection": selection_predictions[name]=joined
                elif stage=="calibration": calibration_predictions[name]=joined
                else: evaluation_predictions[name]=joined
                execution[name]["stages"][stage]={"status":"executed","rows":int(len(joined)),**diag}
                execution[name]["status"]="executed"
                save_progress()
            except Exception as exc:
                execution[name]["status"]="failed"
                execution[name]["stages"][stage]={"status":"failed","error":f"{type(exc).__name__}: {exc}"}
                save_progress()
    candidates=set(selection_predictions)&set(calibration_predictions)&set(evaluation_predictions)
    if not candidates: raise RuntimeError("No benchmark model completed selection, calibration, and final evaluation windows")
    selection_scale={k:g.sort_values("date").unit_sales.to_numpy() for k,g in panel[panel.id.isin(ids)&panel.date.lt(selection_start)].groupby("id",observed=True)}
    selection_scores={}
    for name in sorted(candidates):
        f=selection_predictions[name].rename(columns={"actual":"sales"})
        selection_scores[name]=score(f,"prediction",selection_scale)
    selected=min(candidates,key=lambda name:selection_scores[name]["wape"])
    calibration=calibration_predictions[selected]
    radius=float(np.quantile(np.abs(calibration.actual-calibration.prediction),config.get("interval_level",.90)))
    eval_scale={k:g.sort_values("date").unit_sales.to_numpy() for k,g in panel[panel.id.isin(ids)&panel.date.lt(eval_start)].groupby("id",observed=True)}
    metrics={}; combined=None
    for name in sorted(candidates):
        ev=evaluation_predictions[name]
        f=ev.rename(columns={"actual":"sales"})
        metrics[name]=score(f,"prediction",eval_scale)
        named=ev[["date","id","actual","prediction"]].rename(columns={"prediction":PREDICTION_COLUMNS[name]})
        combined=named if combined is None else combined.merge(named,on=["date","id","actual"],how="outer",validate="one_to_one")
    eval_dates=pd.DatetimeIndex(sorted(combined.date.unique()))
    actuals=combined[["date","id","actual"]].merge(panel[["date","id","store_nbr","item_nbr"]],on=["date","id"],how="left",validate="one_to_one")
    selected_col=PREDICTION_COLUMNS[selected]
    combined["prediction"]=combined[selected_col]
    combined["forecast"]=combined[selected_col]
    combined=add_interval(combined,radius)
    selected_eval=evaluation_predictions[selected]
    coverage=float(((selected_eval.actual>=np.maximum(0,selected_eval.prediction-radius))&(selected_eval.actual<=selected_eval.prediction+radius)).mean())
    validation=combined.merge(panel[["date","id","store_nbr","item_nbr"]],on=["date","id"],how="left",validate="one_to_one")
    validation.to_csv(out/"forecasts"/"model_comparison_validation.csv",index=False)
    calibration.to_csv(out/"forecasts"/"model_comparison_calibration.csv",index=False)
    meta={"evaluation_period":{"start":str(eval_start.date()),"end":str(max_date.date()),"days":evaluation_days},
      "selection_period":{"start":str(selection_start.date()),"end":str(selection_end.date()),"days":selection_days},
      "calibration_period":{"start":str(calibration_start.date()),"end":str(calibration_end.date()),"days":calibration_days},
      "series_count":len(ids),"series_ids":ids,"evaluation_rows":int(len(validation)),
      "common_series_and_dates":True,"selection_metric":"WAPE","selected_model":selected,
      "selection_scores":selection_scores,"execution":execution,
      "interval":{"method":"absolute selected-model calibration residual quantile","level":config.get("interval_level",.90),
        "radius_units":radius,"coverage_on_untouched_evaluation_period":coverage,
        "coverage_rows":int(len(selected_eval))}}
    metrics["model_comparison"]=meta
    (out/"metrics"/"model_comparison.json").write_text(json.dumps(metrics,indent=2,default=str))
    return metrics,validation,selected
