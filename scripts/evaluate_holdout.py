import os
os.environ['KERAS_BACKEND'] = 'torch'

import joblib
import keras
import pandas as pd
from xgboost import XGBRegressor

from src.config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT, HORIZONS_MIN
from src.loading import load_cgm_csv, load_nightscout_events_csv
from src.pipeline import build_period
from src.evaluation import mae, rmse, skill_score, hypo_counts, persistence_baseline, linear_baseline
from src.models import predict_xgboost, predict_lstm

REPORTS_DIR = PROJECT_ROOT / 'reports'
VARIANTS = ['M1', 'M2', 'M3']

cgm = load_cgm_csv(RAW_DIR / 'clarity_cgm_holdout.csv', source='clarity')
events = load_nightscout_events_csv(RAW_DIR / 'nightscout_treatments_holdout.csv')

holdout = build_period(cgm, events, 'holdout')
holdout.to_parquet(PROCESSED_DIR / 'holdout.parquet')

# Inspect data before reporting results 
print('holdout rows ', len(holdout))
print('date range   ', holdout['timestamp'].min(), '->', holdout['timestamp'].max())
print('meals        ', int((holdout['meal_carbs'] > 0).sum()))
print('boluses      ', int((holdout['bolus_insulin'] > 0).sum()))

# Count transitions into a low, not rows below it -> a 30 minute low is one event
low = holdout['glucose'] < 70
print('hypo episodes', int((low & ~low.shift(1, fill_value=False)).sum()))
print()

idx = range(len(holdout))
rows = []

for horizon in HORIZONS_MIN:
    target_col = f'target_glucose_{horizon}'
    truth = holdout[target_col]

    # Gather baselines to compare to
    base_pred = persistence_baseline(holdout, horizon)

    predictions = {
        'persistence': base_pred,
        'linear': linear_baseline(holdout, horizon),
    }

    # Load and run each model for evaluating
    for variant in VARIANTS:
        xgb = XGBRegressor()
        xgb.load_model(str(MODELS_DIR / f'xgb_{variant}_h{horizon}.json'))
        predictions[f'xgboost_{variant}'] = predict_xgboost(holdout, xgb, idx)

        lstm = keras.models.load_model(MODELS_DIR / f'lstm_{variant}_h{horizon}.keras')

        # the scaler is loaded instead of refitted to prevent leaking holdout statistics into the predictions
        scaler = joblib.load(MODELS_DIR / f'lstm_{variant}_h{horizon}_scaler.joblib')
        predictions[f'lstm_{variant}'] = predict_lstm(
            holdout, lstm, scaler, idx, target_col
        )

    # Everything is restricted to rows where the target and all six models have value
    common = truth.dropna().index
    for pred in predictions.values():
        common = common.intersection(pred.dropna().index)

    if len(common) == 0:
        raise SystemExit(f'No scoreable rows at the {horizon} minute horizon.')

    truth_common = truth.loc[common]

    # The baseline must be measured on the same rows as the models it scores
    base_mae = mae(truth_common, predictions['persistence'].loc[common])

       # Creates a list of results which holds one row per model per horizon
    for name, pred in predictions.items():
        pred = pred.loc[common]

        model_mae = mae(truth_common, pred)
        hypo = hypo_counts(truth_common, pred)

        rows.append({
            'horizon': horizon,
            'model': name,
            'mae': round(model_mae, 3),
            'rmse': round(rmse(truth_common, pred), 3),
            'skill': round(skill_score(model_mae, base_mae), 3),
            'n_scored': len(common),
            'hypo_sensitivity': hypo['sensitivity'],
            'hypo_specificity': hypo['specificity'],
            'n_low': hypo['n_low'],
        })

res = pd.DataFrame(rows)
results_path = REPORTS_DIR / 'holdout_results.csv'
res.to_csv(results_path, index=False)

# ---------- print ----------
print(res.pivot(index='model', columns='horizon', values='mae'))
print()
print(res.pivot(index='model', columns='horizon', values='skill'))
print()
print(res[res['horizon'] == 30][['model', 'mae', 'skill', 'hypo_sensitivity', 'n_low']])
print()
print('written to', results_path)