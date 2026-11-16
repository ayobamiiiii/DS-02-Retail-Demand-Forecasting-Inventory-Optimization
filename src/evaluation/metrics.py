import numpy as np
import pandas as pd
def wape(y, p):
    denom = np.abs(y).sum()
    return float(np.abs(y-p).sum()/denom) if denom else float("nan")
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def rmsse(y,p,train):
    scale = np.mean(np.diff(np.asarray(train, dtype=float))**2) if len(train)>1 else 0
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)/scale)) if scale>0 else float("nan")
def score(frame, pred_col, train_by_id):
    vals=[]
    for ident,g in frame.groupby("id", observed=True):
        vals.append(rmsse(g.sales, g[pred_col], train_by_id.get(ident, [])))
    return {"wape":wape(frame.sales.to_numpy(), frame[pred_col].to_numpy()), "rmse":rmse(frame.sales,frame[pred_col]),
            "rmsse_mean":float(np.nanmean(vals)), "n_series":len(vals)}
