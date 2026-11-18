"""Memory-bounded ingestion for the 3 GB Favorita training CSV."""
from pathlib import Path
import pandas as pd

FILES = ("train.csv", "test.csv", "items.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv")
USE = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]

def _require(root):
    missing=[name for name in FILES if not (root/name).exists()]
    if missing: raise FileNotFoundError(f"Favorita files missing from {root}: {', '.join(missing)}")

def choose_items(train_path, n_items=40, chunksize=1_500_000, before_date=None):
    """Rank items using only rows dated before the exclusive selection cutoff."""
    if before_date is None:
        raise ValueError("before_date is required to prevent evaluation-period sample-selection leakage")
    cutoff = pd.Timestamp(before_date)
    totals={}
    for chunk in pd.read_csv(train_path,usecols=["date","item_nbr","unit_sales"],parse_dates=["date"],chunksize=chunksize):
        chunk = chunk[chunk.date < cutoff]
        if chunk.empty:
            continue
        chunk["unit_sales"]=pd.to_numeric(chunk.unit_sales,errors="coerce").fillna(0).clip(lower=0)
        sums=chunk.groupby("item_nbr",sort=False).unit_sales.sum()
        for k,v in sums.items(): totals[k]=totals.get(k,0.0)+float(v)
    return [k for k,_ in sorted(totals.items(),key=lambda kv:kv[1],reverse=True)[:int(n_items)]]

def load_favorita(data_dir, n_items=40, stores=None, chunksize=1_500_000, selection_cutoff=None):
    root=Path(data_dir); _require(root)
    selected=choose_items(root/"train.csv",n_items,chunksize,selection_cutoff)
    stores=list(stores or [1,2,3])
    parts=[]
    for ch in pd.read_csv(root/"train.csv",usecols=USE,chunksize=chunksize,parse_dates=["date"],dtype={"onpromotion":"string"}):
        ch=ch[ch.item_nbr.isin(selected)&ch.store_nbr.isin(stores)]
        if ch.empty: continue
        ch["unit_sales"]=pd.to_numeric(ch.unit_sales,errors="coerce").fillna(0).clip(lower=0)
        ch["onpromotion"]=ch.onpromotion.astype("string").str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False)
        parts.append(ch.groupby(["date","store_nbr","item_nbr"],as_index=False).agg(unit_sales=("unit_sales","sum"),onpromotion=("onpromotion","max")))
    if not parts: raise ValueError("No sales rows found for selected Favorita items/stores")
    train=pd.concat(parts,ignore_index=True).groupby(["date","store_nbr","item_nbr"],as_index=False).agg(unit_sales=("unit_sales","sum"),onpromotion=("onpromotion","max"))
    test=pd.read_csv(root/"test.csv",parse_dates=["date"])
    test=test[test.item_nbr.isin(selected)&test.store_nbr.isin(stores)].copy()
    test["onpromotion"]=test.onpromotion.astype("string").str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False)
    items=pd.read_csv(root/"items.csv"); stores_df=pd.read_csv(root/"stores.csv")
    oil=pd.read_csv(root/"oil.csv",parse_dates=["date"]); holidays=pd.read_csv(root/"holidays_events.csv",parse_dates=["date"])
    transactions=pd.read_csv(root/"transactions.csv",parse_dates=["date"])
    return train,test,items,stores_df,oil,holidays,transactions

def make_daily_panel(train,items,stores,oil,holidays,transactions,store_ids):
    """Create zero-filled daily demand series and observed exogenous covariates."""
    min_date=train.date.min(); max_date=train.date.max()
    keys=train[["store_nbr","item_nbr"]].drop_duplicates()
    dates=pd.date_range(min_date,max_date,freq="D")
    pairs=keys.reset_index(drop=True)
    idx=pd.MultiIndex.from_product([range(len(pairs)),dates],names=["pair","date"])
    pairs["pair"]=pairs.index
    panel=idx.to_frame(index=False).merge(pairs,on="pair",how="left").drop(columns="pair")
    panel=panel.merge(train,on=["date","store_nbr","item_nbr"],how="left",validate="one_to_one")
    panel["unit_sales"]=panel.unit_sales.fillna(0).astype("float32")
    panel["onpromotion"]=panel.onpromotion.fillna(False).astype("int8")
    panel=panel.merge(items[["item_nbr","family","class","perishable"]],on="item_nbr",how="left",validate="many_to_one")
    panel=panel.merge(stores[["store_nbr","state","type","cluster"]],on="store_nbr",how="left",validate="many_to_one")
    oil_series=oil.sort_values("date").set_index("date").dcoilwtico.ffill()
    oil=oil_series.rename("oil_price")
    panel=panel.join(oil,on="date")
    oil_mean=float(panel.oil_price.mean()) if panel.oil_price.notna().any() else 0
    panel["oil_price"]=panel.oil_price.fillna(oil_mean)
    oil_lag=oil_series.reindex(pd.date_range(min(oil_series.index.min(),min_date),max_date,freq="D")).ffill().shift(1).rename("oil_price_lag_1")
    panel=panel.join(oil_lag,on="date"); panel["oil_price_lag_1"]=panel.oil_price_lag_1.fillna(oil_mean)
    # Overall Ecuador holiday indicator; transferred events are not treated as holidays.
    holiday=holidays[(holidays.locale=="National")&(~holidays.transferred.astype(bool))].groupby("date").size().gt(0).astype("int8").rename("is_holiday")
    panel=panel.join(holiday,on="date"); panel["is_holiday"]=panel.is_holiday.fillna(0).astype("int8")
    tx=transactions.pivot(index="date",columns="store_nbr",values="transactions").sort_index()
    tx=tx.reindex(dates).ffill().fillna(0)
    tx_daily=tx.stack().rename("transactions").rename_axis(index=["date","store_nbr"]).reset_index()
    panel=panel.merge(tx_daily,on=["date","store_nbr"],how="left",validate="many_to_one")
    panel["transactions"]=panel.transactions.fillna(0)
    panel["id"]=panel.store_nbr.astype(str)+"_"+panel.item_nbr.astype(str)
    for c in ["family","state","type"]: panel[c]=panel[c].astype("category").cat.codes.astype("int16")
    return panel.sort_values(["id","date"]).reset_index(drop=True)

def add_lagged_covariates(panel,transactions=None):
    """Use prior-day store transactions and oil prices, never same-day observations."""
    x=panel.drop(columns=["oil_price_lag_1","transactions_lag_1"],errors="ignore").copy()
    dates=pd.date_range(x.date.min(),x.date.max(),freq="D")
    if transactions is not None:
        tx_source=transactions.pivot(index="date",columns="store_nbr",values="transactions").sort_index()
        tx_source=tx_source.reindex(dates).ffill().fillna(0)
        tx_current=tx_source.stack().rename("transactions").rename_axis(index=["date","store_nbr"]).reset_index()
        x=x.drop(columns=["transactions"],errors="ignore").merge(tx_current,on=["date","store_nbr"],how="left",validate="many_to_one")
        x["transactions"]=x.transactions.fillna(0)
    daily_oil=x.groupby("date",observed=True).oil_price.first().reindex(dates).ffill().shift(1)
    fallback=float(x.oil_price.mean()) if x.oil_price.notna().any() else 0.0
    x["oil_price_lag_1"]=x.date.map(daily_oil).fillna(fallback)
    tx=x[["date","store_nbr","transactions"]].drop_duplicates(["date","store_nbr"])
    tx_wide=tx.pivot(index="date",columns="store_nbr",values="transactions").reindex(dates).ffill().shift(1)
    tx_lag=tx_wide.stack().rename("transactions_lag_1").rename_axis(index=["date","store_nbr"]).reset_index()
    x=x.merge(tx_lag,on=["date","store_nbr"],how="left",validate="many_to_one")
    x["transactions_lag_1"]=x.transactions_lag_1.fillna(0)
    return x.sort_values(["id","date"]).reset_index(drop=True)
