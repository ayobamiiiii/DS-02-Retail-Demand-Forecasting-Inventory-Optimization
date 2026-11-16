import numpy as np
class NaiveWeekly:
    def fit(self, frame): return self
    def predict(self, frame): return frame["lag_7"].to_numpy()

def prophet_forecast(series, horizon):
    try:
        from prophet import Prophet
    except ImportError as e:
        raise RuntimeError("Prophet is optional; install prophet to enable it") from e
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(series.rename_axis("ds").reset_index().rename(columns={"sales":"y"}))
    return m.predict(m.make_future_dataframe(periods=horizon))["yhat"].tail(horizon).to_numpy()
