import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import RANDOM_SEED

GLUCOSE_FEATURES = [
    'glucose',
    'glucose_lag_5_min',
    'glucose_lag_10_min',
    'glucose_lag_15_min',
    'glucose_lag_30_min',
    'glucose_slope_5_min',
    'glucose_slope_10_min',
    'glucose_slope_15_min',
    'glucose_slope_30_min',
    'glucose_slope_60_min',
    'glucose_acceleration_10_min',
    'glucose_acceleration_15_min',
    'glucose_acceleration_30_min',
    'glucose_mean_15_min',
    'glucose_mean_30_min',
    'glucose_mean_60_min',
    'glucose_std_15_min',
    'glucose_std_30_min',
    'glucose_std_60_min',
]

INSULIN_FEATURES = ['iob']

CARB_FEATURES = ['meal_cob', 'correction_cob', 'total_cob']

EXERCISE_FEATURES = [
    'exercise_intensity',
    'exercise_effect',
    'is_walking',
    'is_lifting',
    'is_playing_basketball',
]

INTERACTION_FEATURES = [
    'iob_over_meal_cob',
    'iob_x_glucose',
    'cob_x_glucose',
    'exercise_x_iob',
]

TIME_FEATURES = ['time_sin', 'time_cos']

# targets, timestamp, source and is_interpolated are excluded -> targets would leak, the rest describe the data rather than the physiology
FEATURE_COLUMNS = (
    GLUCOSE_FEATURES
    + INSULIN_FEATURES
    + CARB_FEATURES
    + EXERCISE_FEATURES
    + INTERACTION_FEATURES
    + TIME_FEATURES
)

TIERS = {
    '1: glucose only':   GLUCOSE_FEATURES,
    '2: + insulin':      GLUCOSE_FEATURES + INSULIN_FEATURES,
    '3: + carbs':        GLUCOSE_FEATURES + INSULIN_FEATURES + CARB_FEATURES,
    '4: + exercise':     GLUCOSE_FEATURES + INSULIN_FEATURES + CARB_FEATURES + EXERCISE_FEATURES,
    '5: + interactions': GLUCOSE_FEATURES + INSULIN_FEATURES + CARB_FEATURES + EXERCISE_FEATURES + INTERACTION_FEATURES,
    '6: + time (full)':  FEATURE_COLUMNS,
}

def train_xgboost(df, train_idx, target_col, features=FEATURE_COLUMNS):
    train = df.iloc[train_idx]
    train = train.dropna(subset=features + [target_col])

    X_train = train[features]
    # predict the change from the current reading, not the level
    y_train = train[target_col] - train['glucose']

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=RANDOM_SEED
    )

    model.fit(X_train, y_train)

    return model

def predict_xgboost(df, model, idx, features=FEATURE_COLUMNS):
    test = df.iloc[idx]

    X_test = test[features].dropna()
    deltas = model.predict(X_test)

    # the model outputs a change so add the current reading back to get mg/dL
    preds = deltas + test.loc[X_test.index, 'glucose'].to_numpy()

    return pd.Series(preds, index=X_test.index)

LSTM_FEATURES = [
    'glucose',
    'bolus_insulin',
    'meal_carbs',
    'correction_carbs',
    'is_walking',
    'is_lifting',
    'is_playing_basketball',
    'exercise_intensity',
    'time_sin',
    'time_cos'
]

LSTM_WINDOW = 60

def create_sequences(df, idx, target_col, features=LSTM_FEATURES, window=LSTM_WINDOW):
    rows = df.iloc[idx]

    values = rows[features].to_numpy()
    targets = rows[target_col].to_numpy()

    complete = rows[features].notna().all(axis=1).to_numpy()
    has_target = rows[target_col].notna().to_numpy()
    period = rows['period'].to_numpy()

    X, y, last_rows = [], [], []

    for i in range(len(rows) - window + 1):
        end = i + window - 1

        # a window must not span the gap between collection periods
        if period[i] != period[end]:
            continue

        # skip any window containing an incomplete row, or with no target
        if not complete[i:end + 1].all():
            continue
        if not has_target[end]:
            continue

        X.append(values[i:end + 1])
        y.append(targets[end])
        last_rows.append(rows.index[end])

    return np.array(X), np.array(y), last_rows

def build_lstm(n_features, window):
    model = keras.Sequential([
        keras.layers.Input(shape=(window, n_features)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def train_lstm(df, train_idx, target_col, features=LSTM_FEATURES, window=LSTM_WINDOW):
    keras.utils.set_random_seed(RANDOM_SEED)

    train = df.iloc[train_idx].copy()

    # keep the real glucose values before scaling overwrites them
    current = train['glucose'].copy()

    # fitted on training rows only, and saved with the model
    scaler = StandardScaler()
    complete = train[features].notna().all(axis=1)
    scaler.fit(train.loc[complete, features])
    train[features] = scaler.transform(train[features])

    X, y, last_rows = create_sequences(train, range(len(train)), target_col, features, window)

    # predict the change from the current reading, not the level
    y = y - current.loc[last_rows].to_numpy()

    # last 15% of training sequences for early stopping -- never the test block
    split = int(len(X) * 0.85)

    model = build_lstm(len(features), window)
    model.fit(
        X[:split], y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=100,
        batch_size=64,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )

    return model, scaler

def predict_lstm(df, model, scaler, idx, target_col, features=LSTM_FEATURES, window=LSTM_WINDOW):
    test = df.iloc[idx].copy()

    current = test['glucose'].copy()

    test[features] = scaler.transform(test[features])

    X, _, last_rows = create_sequences(test, range(len(test)), target_col, features, window)
    deltas = model.predict(X, verbose=0).flatten()

    # the model outputs a change; add the current reading back to get mg/dL
    preds = deltas + current.loc[last_rows].to_numpy()
    return pd.Series(preds, index=last_rows)

