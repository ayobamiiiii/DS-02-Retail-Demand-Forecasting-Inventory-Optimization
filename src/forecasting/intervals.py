import numpy as np
def residual_radius(residuals, level=.90):
    a=np.abs(np.asarray(residuals, dtype=float)); a=a[np.isfinite(a)]
    return float(np.quantile(a, level)) if len(a) else 0.0
def add_interval(frame, radius):
    out=frame.copy(); out["lower"]=(out.forecast-radius).clip(lower=0); out["upper"]=out.forecast+radius
    return out
