import hashlib

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, MODELS_DIR, HORIZONS_MIN, RANDOM_SEED
from src.models import train_xgboost, train_lstm

HOLDOUT_START = '2026-09-14'

# load and verify the development datset for training
df = pd.read_parquet(PROCESSED_DIR / 'development.parquet')

print('development rows', len(df))
print('date range      ', df['timestamp'].min(), '->', df['timestamp'].max())
print()

# check that freeze is trained on development data
if df['timestamp'].max() >= pd.Timestamp(HOLDOUT_START, tz=df['timestamp'].dt.tz):
    raise SystemExit('development data reaches into the holdout window, stopping')

# the three variants set to their positional row indices
VARIANT_ROWS = {
    'M1': np.flatnonzero((df['period'] == 'april').to_numpy()),
    'M2': np.flatnonzero((df['period'] == 'sept').to_numpy()),
    'M3': np.arange(len(df)),
}

# verify that each variant is correct
for name, rows in VARIANT_ROWS.items():
    print(f'{name}  {len(rows)} rows  '
          f'{df["timestamp"].iloc[rows[0]]} -> {df["timestamp"].iloc[rows[-1]]}')
print()

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = []

for variant, train_rows in VARIANT_ROWS.items():
    for horizon in HORIZONS_MIN:
        target_col = f'target_glucose_{horizon}'

        # ---- XGBoost ----
        xgb = train_xgboost(df, train_rows, target_col)
        xgb_path = MODELS_DIR / f'xgb_{variant}_h{horizon}.json'
        xgb.save_model(str(xgb_path))

        # ---- LSTM, plus the scaler it was fit with ----
        lstm, scaler = train_lstm(df, train_rows, target_col)
        lstm_path = MODELS_DIR / f'lstm_{variant}_h{horizon}.keras'
        scaler_path = MODELS_DIR / f'lstm_{variant}_h{horizon}_scaler.joblib'
        lstm.save(lstm_path)
        joblib.dump(scaler, scaler_path)

        for path in (xgb_path, lstm_path, scaler_path):
            manifest.append({
                'variant': variant,
                'horizon': horizon,
                'file': path.name,
                'bytes': path.stat().st_size,
                'sha256': file_hash(path),
                'n_train_rows': len(train_rows),
                'seed': RANDOM_SEED,
            })

        print(f'{variant} h{horizon} saved')

# this file is written to reports/ -> holds the records of exactly what was frozen and when
man = pd.DataFrame(manifest)
man['frozen_at'] = pd.Timestamp.now(tz='America/New_York').isoformat()
man['dev_last_timestamp'] = df['timestamp'].max().isoformat()
man['dev_rows'] = len(df)

manifest_path = MODELS_DIR / 'freeze_manifest.csv'
man.to_csv(manifest_path, index=False)

print()
print(len(man), 'files written to', MODELS_DIR)
print('total size', round(man['bytes'].sum() / 1e6, 1), 'MB')
print('manifest  ', manifest_path)
