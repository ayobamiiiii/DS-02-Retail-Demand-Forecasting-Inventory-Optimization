"""Darts global autoregressive regression forecaster."""
import pandas as pd

def predict_darts(history, future_dates, series_col="id", target_col="unit_sales"):
    """Train Darts RegressionModel on a common set of daily series and forecast direct blocks."""
    try:
        from darts import TimeSeries
        from darts.models import RegressionModel
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise RuntimeError("Darts integration requires u8darts and scikit-learn") from exc
    series=[]; ids=[]
    for ident,group in history.groupby(series_col,sort=True,observed=True):
        ts=TimeSeries.from_dataframe(group.sort_values("date"),time_col="date",value_cols=target_col,freq="D")
        series.append(ts); ids.append(ident)
    if not series: raise ValueError("Darts received no training series")
    model=RegressionModel(lags=28,model=Ridge(alpha=1.0),output_chunk_length=1,output_chunk_shift=0)
    model.fit(series,verbose=False)
    predictions=model.predict(n=len(future_dates),series=series,verbose=False)
    if not isinstance(predictions,list): predictions=[predictions]
    dates=pd.DatetimeIndex(future_dates); outputs=[]
    for ident,pred in zip(ids,predictions):
        values=pred.values(copy=False).reshape(-1)
        if len(values)!=len(dates): raise ValueError(f"Darts returned {len(values)} values for {len(dates)} dates")
        outputs.append(pd.DataFrame({"id":ident,"date":dates,"prediction":values.clip(min=0)}))
    return pd.concat(outputs,ignore_index=True)
