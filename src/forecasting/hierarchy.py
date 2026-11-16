import pandas as pd

def rollup(bottom):
    """Bottom-up coherent totals: child item predictions define all parents."""
    dims=[("store",["store_id"]),("state",["state_id"]),("overall",[]),
          ("item_store",["item_id","store_id"]),("item_state",["item_id","state_id"])]
    outputs=[]
    for level, keys in dims:
        g=bottom.groupby(["date"]+keys, dropna=False, observed=True)[["forecast","lower","upper"]].sum().reset_index()
        g["level"]=level
        outputs.append(g)
    b=bottom.copy(); b["level"]="item_store"; b["item_id"]=b.id.str.split("_").str[:3].str.join("_")
    # Append item-store native bottom rows; store rollups remain directly additive.
    return pd.concat([b]+outputs, ignore_index=True, sort=False)

def rollup_favorita(bottom, stores):
    """Reconcile Favorita item/store forecasts bottom-up through city and state."""
    dims=stores[["store_nbr","city","state"]].drop_duplicates("store_nbr")
    frame=bottom.merge(dims,on="store_nbr",how="left",validate="many_to_one")
    if frame[["city","state"]].isna().any().any():
        raise ValueError("Every forecast store must map to a city and state")
    levels=[]
    for level,keys in (("item_store",["item_nbr","store_nbr"]),("item",["item_nbr"]),
                       ("store",["store_nbr"]),("city",["city"]),("state",["state"]),("overall",[])):
        grouped=frame.groupby(["date"]+keys,as_index=False,dropna=False,observed=True)[["forecast","lower","upper"]].sum()
        grouped["level"]=level
        levels.append(grouped)
    return pd.concat(levels,ignore_index=True,sort=False)
