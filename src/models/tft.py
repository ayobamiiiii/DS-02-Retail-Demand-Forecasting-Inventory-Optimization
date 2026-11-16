"""Small global Temporal Fusion Transformer forecaster for M1-sized benchmarks."""
import pandas as pd

def predict_tft(history, future, max_epochs=1, batch_size=64, limit_train_batches=10):
    """Fit a compact global TFT and predict a common horizon for a series sample.

    Uses CPU when MPS is unavailable. Only time index/day-of-week are future-known
    covariates; unknown future targets are set to zero in the decoder input and are
    not used as model inputs.
    """
    try:
        import torch
        import pytorch_forecasting as pf
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from lightning.pytorch import Trainer
    except Exception as exc:
        raise RuntimeError(f"TFT dependencies failed to import: {type(exc).__name__}: {exc}") from exc
    torch.set_num_threads(min(4,torch.get_num_threads()))
    history=history[["id","date","unit_sales"]].copy().sort_values(["id","date"])
    future=future[["id","date"]].copy().sort_values(["id","date"])
    if future.empty: raise ValueError("TFT received an empty prediction horizon")
    horizon=int(future.groupby("id",observed=True).size().max())
    history["series_id"]=history.id.astype(str)
    future["series_id"]=future.id.astype(str); future["unit_sales"]=0.0
    frame=pd.concat([history,future],ignore_index=True).sort_values(["series_id","date"])
    origin=history.date.max()
    frame["time_idx"]=(frame.date-frame.date.min()).dt.days.astype("int32")
    frame["dow"]=frame.date.dt.dayofweek.astype(str)
    train=frame[frame.date<=origin].copy()
    encoder=min(56,max(14,int(train.date.nunique())-horizon-1))
    training=TimeSeriesDataSet(train,time_idx="time_idx",target="unit_sales",group_ids=["series_id"],
      max_encoder_length=encoder,min_encoder_length=min(14,encoder),max_prediction_length=horizon,
      time_varying_known_reals=["time_idx"],time_varying_known_categoricals=["dow"],
      time_varying_unknown_reals=["unit_sales"],target_normalizer=GroupNormalizer(groups=["series_id"]),
      add_relative_time_idx=True,add_target_scales=True,add_encoder_length=True,allow_missing_timesteps=False)
    predict_ds=TimeSeriesDataSet.from_dataset(training,frame,predict=True,stop_randomization=True)
    train_loader=training.to_dataloader(train=True,batch_size=batch_size,num_workers=0)
    predict_loader=predict_ds.to_dataloader(train=False,batch_size=batch_size,num_workers=0)
    model=TemporalFusionTransformer.from_dataset(training,learning_rate=.02,hidden_size=8,attention_head_size=2,
      dropout=.1,hidden_continuous_size=4,loss=pf.metrics.MAE(),log_interval=-1)
    accelerator="mps" if torch.backends.mps.is_available() else "cpu"
    trainer=Trainer(max_epochs=int(max_epochs),accelerator=accelerator,devices=1,logger=False,
      enable_progress_bar=False,enable_model_summary=False,enable_checkpointing=False,
      limit_train_batches=int(limit_train_batches),deterministic=True)
    trainer.fit(model,train_dataloaders=train_loader)
    predicted,index=model.predict(predict_loader,mode="prediction",return_index=True,
      trainer_kwargs={"accelerator":accelerator,"devices":1,"logger":False,"enable_progress_bar":False})
    values=predicted.detach().cpu().numpy()
    if values.ndim==3: values=values.squeeze(-1)
    index=index.reset_index(drop=True)
    outputs=[]
    for row_i,row in index.iterrows():
        ident=str(row["series_id"])
        dates=future[future.id.astype(str)==ident].date.sort_values().to_numpy()
        vals=values[row_i].reshape(-1)
        if len(vals)!=len(dates): raise ValueError(f"TFT returned {len(vals)} predictions for {len(dates)} dates ({ident})")
        outputs.append(pd.DataFrame({"id":ident,"date":dates,"prediction":vals.clip(min=0)}))
    diagnostics={"device":accelerator,"epochs":int(max_epochs),"series":int(history.id.nunique()),
                 "horizon":horizon,"torch_version":torch.__version__,"mps_available":bool(torch.backends.mps.is_available())}
    return pd.concat(outputs,ignore_index=True),diagnostics
