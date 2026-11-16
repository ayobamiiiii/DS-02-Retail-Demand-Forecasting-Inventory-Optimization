def create_model(seed=41):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise RuntimeError("LightGBM is required. Install requirements.txt first.") from e
    return LGBMRegressor(n_estimators=350, learning_rate=0.05, num_leaves=31, max_depth=-1,
                         subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                         random_state=seed, n_jobs=-1, verbosity=-1)
