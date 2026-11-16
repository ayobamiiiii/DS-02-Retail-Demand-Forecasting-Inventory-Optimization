from pathlib import Path
import pandas as pd

FILES = ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")

def load_m5(data_dir):
    root = Path(data_dir)
    missing = [f for f in FILES if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"M5 files missing in {root}: {', '.join(missing)}. Download the official M5 data and place the files there.")
    sales = pd.read_csv(root / FILES[0])
    calendar = pd.read_csv(root / FILES[1])
    prices = pd.read_csv(root / FILES[2])
    return sales, calendar, prices

def to_long(sales, calendar, max_items=None):
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if max_items is not None:
        # Stable selection across runs: top selling items, which improves signal at modest scale.
        sales = sales.assign(_total=sales[day_cols].sum(axis=1)).nlargest(int(max_items), "_total").drop(columns="_total")
    id_cols = [c for c in sales.columns if c not in day_cols]
    long = sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    long["date"] = pd.to_datetime(long["date"])
    return long

def attach_prices(long, prices):
    out = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    out["sell_price"] = out.groupby(["store_id", "item_id"], observed=True)["sell_price"].ffill()
    return out

def profile(long):
    return {"rows": int(len(long)), "columns": list(long.columns), "missing": long.isna().sum().astype(int).to_dict(),
            "duplicate_rows": int(long.duplicated(["id", "date"]).sum()), "date_min": str(long.date.min().date()),
            "date_max": str(long.date.max().date()), "items": int(long.item_id.nunique()), "stores": int(long.store_id.nunique()),
            "states": int(long.state_id.nunique()), "sales_total": float(long.sales.sum())}
