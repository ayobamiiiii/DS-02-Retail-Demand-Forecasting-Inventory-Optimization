"""Small-sample, per-series Prophet direct forecaster."""
import logging
import pandas as pd

def predict_prophet(history, future_dates, series_col="id", target_col="unit_sales", iterations=200):
    """Fit one compact Prophet model per series and forecast the requested dates."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise RuntimeError("Prophet is unavailable; install prophet in the project environment") from exc
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    dates=pd.DatetimeIndex(future_dates)
    outputs=[]
    for ident,group in history.groupby(series_col,sort=True,observed=True):
        frame=group[["date",target_col]].rename(columns={"date":"ds",target_col:"y"}).sort_values("ds")
        model=Prophet(weekly_seasonality=True,yearly_seasonality=len(frame)>=730,daily_seasonality=False,
                      n_changepoints=min(12,max(1,len(frame)//100)),changepoint_prior_scale=.05,
                      uncertainty_samples=0)
        model.fit(frame,iter=int(iterations))
        future=model.predict(pd.DataFrame({"ds":dates}))
        outputs.append(pd.DataFrame({"id":ident,"date":dates,"prediction":future.yhat.clip(lower=0).to_numpy()}))
    return pd.concat(outputs,ignore_index=True)
