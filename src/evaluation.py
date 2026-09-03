import numpy as np
from src.config import GRID_MINUTES, HYPO_THRESHOLD

def walk_forward_splits(n_rows, n_splits, horizon_minutes):
    gap = horizon_minutes // GRID_MINUTES
    block = n_rows // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        boundary = block * (i + 1)
        # train stops 'gap' rows early so no training target reaching into test
        train = range(0, boundary - gap)
        test = range(boundary, boundary + block)
        splits.append((train, test))

    return splits

def persistence_baseline(df, horizon_minutes):
    return df['glucose']

def linear_baseline(df, horizon_minutes):
    return df['glucose'] + df['glucose_slope_15_min'] * horizon_minutes

def align( y_true, y_pred):
    ok = y_true.notna() & y_pred.notna()
    return y_true[ok], y_pred[ok]

def mae(y_true, y_pred):
    y_true, y_pred = align(y_true, y_pred)
    return np.abs(y_true - y_pred).mean()

def rmse(y_true, y_pred):
    y_true, y_pred = align(y_true, y_pred)
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def skill_score(model_mae, baseline_mae):
    return 1 - model_mae / baseline_mae

def hypo_counts(y_true, y_pred, threshold=HYPO_THRESHOLD):
    y_true, y_pred = align(y_true, y_pred)

    actual_low = y_true < threshold
    predicted_low = y_pred < threshold

    # identify true positives, false positives, true negatives, and false negatives
    tp = int((actual_low & predicted_low).sum())
    fp = int((~actual_low & predicted_low).sum())
    fn = int((actual_low & ~predicted_low).sum())
    tn = int((~actual_low & ~predicted_low).sum())

    # no lows in this window means sensitivity is undefined, not zero
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n_low": tp + fn,
    }


