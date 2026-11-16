import pandas as pd

FEATURES=["lag_1","lag_7","lag_14","lag_28","mean_7","mean_28","std_28","dow","month","day",
         "weekofyear","onpromotion","is_holiday","oil_price_lag_1","transactions_lag_1","item_nbr","store_nbr","family","class","perishable","state","cluster","store_type"]

def build(frame):
    x=frame.sort_values(["id","date"]).copy()
    g=x.groupby("id",sort=False,observed=True).unit_sales
    for lag in (1,7,14,28): x[f"lag_{lag}"]=g.shift(lag)
    prev=g.shift(1)
    x["mean_7"]=prev.groupby(x.id,sort=False).rolling(7,min_periods=7).mean().reset_index(level=0,drop=True)
    x["mean_28"]=prev.groupby(x.id,sort=False).rolling(28,min_periods=28).mean().reset_index(level=0,drop=True)
    x["std_28"]=prev.groupby(x.id,sort=False).rolling(28,min_periods=28).std().reset_index(level=0,drop=True)
    x["dow"]=x.date.dt.dayofweek.astype("int8"); x["month"]=x.date.dt.month.astype("int8")
    x["day"]=x.date.dt.day.astype("int8"); x["weekofyear"]=x.date.dt.isocalendar().week.astype("int16")
    x["store_type"]=x.type.astype("int16")
    return x
