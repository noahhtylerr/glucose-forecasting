import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR
from src.loading import load_cgm_csv, load_events_csv, load_nightscout_events_csv
from src.timeline import to_grid
from src.events import add_events
from src.physiology import add_iob_cob
from src.features import build_features

from src.config import HORIZONS_MIN
from src.evaluation import walk_forward_splits, mae, skill_score, persistence_baseline, linear_baseline
from src.models import train_xgboost, predict_xgboost, train_lstm, predict_lstm
from src.pipeline import build_period
N_SPLITS = 5

# ---------- CGM: both periods from Clarity, same loader ----------
april_cgm = load_cgm_csv(RAW_DIR / 'clarity_cgm_april_may.csv', source='clarity')
sept_cgm = load_cgm_csv(RAW_DIR / 'clarity_cgm_sept1-13.csv', source='clarity')

print('april cgm ', len(april_cgm), april_cgm['timestamp'].min(), '->', april_cgm['timestamp'].max())
print('sept cgm  ', len(sept_cgm), sept_cgm['timestamp'].min(), '->', sept_cgm['timestamp'].max())

cgm = pd.concat([april_cgm, sept_cgm], ignore_index=True)
cgm = cgm.sort_values('timestamp')
cgm = cgm.drop_duplicates(subset='timestamp', keep='last')
cgm = cgm.reset_index(drop=True)


# ---------- events: April CSV plus the September API pull ----------
april_events = load_events_csv(RAW_DIR / 'nightscout_treatments_dev.csv')
sept_events = load_nightscout_events_csv(RAW_DIR / 'nightscout_treatments_sept.csv')

events = pd.concat([april_events, sept_events], ignore_index=True)
events = events.sort_values('timestamp')
events = events.drop_duplicates(subset=['timestamp', 'event_type'], keep='last')
events = events.reset_index(drop=True)

print('events    ', len(events), events['timestamp'].min(), '->', events['timestamp'].max())
print()
print(events['event_type'].value_counts())

april_df = build_period(april_cgm, events, 'april')
sept_df = build_period(sept_cgm, events, 'sept')

df = pd.concat([april_df, sept_df], ignore_index=True)





results = []
for horizon in HORIZONS_MIN:
    target_col = f'target_glucose_{horizon}'
    truth = df[target_col]

    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(len(df), N_SPLITS, horizon)):
        model = train_xgboost(df, train_idx, target_col)
        pred = predict_xgboost(df, model, test_idx)

        test = df.iloc[test_idx]
        xgb_mae = mae(truth, pred)
        base_mae = mae(truth, persistence_baseline(test, horizon))
        lin_mae = mae(truth, linear_baseline(test, horizon))

        results.append({
            'horizon': horizon,
            'split': i,
            'xgboost': xgb_mae,
            'persistence': base_mae,
            'linear': lin_mae,
            'skill': skill_score(xgb_mae, base_mae),
        })

res = pd.DataFrame(results)

for horizon in HORIZONS_MIN:
    target_col = f'target_glucose_{horizon}'
    truth = df[target_col]

    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(len(df), N_SPLITS, horizon)):
        model, scaler = train_lstm(df, train_idx, target_col)
        pred = predict_lstm(df, model, scaler, test_idx, target_col)

        lstm_mae = mae(truth, pred)
        base_mae = mae(truth, persistence_baseline(df.iloc[test_idx], horizon))
        print(horizon, i, round(lstm_mae, 3), round(skill_score(lstm_mae, base_mae), 3))
        