import pandas as pd
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

# targets, timestamp, source and is_interpolated are excluded:
# targets would leak, the rest describe the data rather than the physiology
FEATURE_COLUMNS = (
    GLUCOSE_FEATURES
    + INSULIN_FEATURES
    + CARB_FEATURES
    + EXERCISE_FEATURES
    + INTERACTION_FEATURES
    + TIME_FEATURES
)

TIERS = {
    'glucose only':   GLUCOSE_FEATURES,
    '+ insulin':      GLUCOSE_FEATURES + INSULIN_FEATURES,
    '+ carbs':        GLUCOSE_FEATURES + INSULIN_FEATURES + CARB_FEATURES,
    '+ exercise':     GLUCOSE_FEATURES + INSULIN_FEATURES + CARB_FEATURES + EXERCISE_FEATURES,
    '+ interactions': GLUCOSE_FEATURES + INSULIN_FEATURES + CARB_FEATURES + EXERCISE_FEATURES + INTERACTION_FEATURES,
    '+ time (full)':  FEATURE_COLUMNS,
}

def train_xgboost(df, train_idx, target_col, features=FEATURE_COLUMNS):
    train = df.iloc[train_idx]
    train = train.dropna(subset=features + [target_col])

    X_train = train[features]
    y_train = train[target_col]

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
    y_pred = model.predict(X_test)

    return pd.Series(y_pred, index=X_test.index)


