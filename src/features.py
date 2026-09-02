import numpy as np
import pandas as pd
from src.config import EXERCISE_DECAY_MIN, EXERCISE_EFFECT_MIN, HORIZONS_MIN, GRID_MINUTES


def build_features(df, events):
    df = df.copy()

    df = add_lags(df)
    df = add_slopes(df)
    df = add_acceleration(df)
    df = add_rolling(df)
    df = add_time_features(df)
    df = add_exercise(df, events)
    df = add_interaction_features(df)
    df = add_targets(df)

    return df

# add lags from previous glucose readings
def add_lags(df):
    df['glucose_lag_5_min'] = df['glucose'].shift(1)
    df['glucose_lag_10_min'] = df['glucose'].shift(2)
    df['glucose_lag_15_min'] = df['glucose'].shift(3)
    df['glucose_lag_30_min'] = df['glucose'].shift(6)
    return df

# add slope and acceleration of glucose for various windows
def add_slopes(df):
    df['glucose_slope_5_min'] = df['glucose'].diff(1) / 5
    df['glucose_slope_10_min'] = df['glucose'].diff(2) / 10
    df['glucose_slope_15_min'] = df['glucose'].diff(3) / 15
    df['glucose_slope_30_min'] = df['glucose'].diff(6) / 30
    df['glucose_slope_60_min'] = df['glucose'].diff(12) / 60
    return df

def add_acceleration(df):
    df['glucose_acceleration_10_min'] = df['glucose_slope_10_min'].diff(2) / 10
    df['glucose_acceleration_15_min'] = df['glucose_slope_15_min'].diff(3) / 15
    df['glucose_acceleration_30_min'] = df['glucose_slope_30_min'].diff(6) / 30
    return df

# add rolling metrics like mean and standard deviation for various windows
def add_rolling(df):
    df['glucose_mean_15_min'] = df['glucose'].rolling(window=3).mean()
    df['glucose_mean_30_min'] = df['glucose'].rolling(window=6).mean()
    df['glucose_mean_60_min'] = df['glucose'].rolling(window=12).mean()
    df['glucose_std_15_min'] = df['glucose'].rolling(window=3).std()
    df['glucose_std_30_min'] = df['glucose'].rolling(window=6).std()
    df['glucose_std_60_min'] = df['glucose'].rolling(window=12).std()
    return df

def add_time_features(df):
    hours = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    df['time_sin'] = np.sin(2 * np.pi * hours / 24)
    df['time_cos'] = np.cos(2 * np.pi * hours / 24)
    return df

def add_exercise(df, events):
    # convert exercise note into binary flag and numeric value
    EXERCISE_TYPES = {
        'walk': ('is_walking', 0.3),
        'workout': ('is_lifting', 0.7),
        'basketball': ('is_playing_basketball', 0.5)
    }

    df['exercise_intensity'] = 0.0
    df['exercise_effect'] = 0.0
    for col, _ in EXERCISE_TYPES.values():
        df[col] = 0

    exercise = events[events['event_type'] == 'Exercise']

    for _, row in exercise.iterrows():
        match = None
        for keyword, pair, in EXERCISE_TYPES.items():
            if keyword in row['notes']:
                match = pair
                break

        # skip unlabeled exercise, rather than mislabel it
        if match is None:
            continue

        col, intensity = match
        start = row['timestamp']
        end = start + pd.Timedelta(minutes=row['duration'])

        # during the session
        during = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        df.loc[during, "exercise_intensity"] += intensity
        df.loc[during, col] = 1

        # the effect on insulin sensitivity after the exercise session
        after = df["timestamp"] > end
        minutes_after = (df.loc[after, "timestamp"] - end).dt.total_seconds() / 60
        decay = np.exp(-minutes_after / EXERCISE_DECAY_MIN) * intensity
        decay[minutes_after >= EXERCISE_EFFECT_MIN] = 0
        df.loc[after, "exercise_effect"] += decay

    return df

def add_interaction_features(df):
    # avoid dividing by zero
    df["iob_over_meal_cob"] = np.where(
        df["meal_cob"] > 0,
        df["iob"] / df["meal_cob"],
        0.0,
    )
    df["iob_x_glucose"] = df["iob"] * df["glucose"]
    df["cob_x_glucose"] = df["total_cob"] * df["glucose"]
    df["exercise_x_iob"] = (df["is_walking"] * df["iob"]) + (df["is_lifting"] * df["iob"]) + (df["is_playing_basketball"] * df["iob"])
    return df

def add_targets(df):
    for minutes in HORIZONS_MIN:
        k = minutes // GRID_MINUTES
        target = df['glucose'].shift(-k)
        # ensures the value being predicted is real and not interpolated
        invented = df['is_interpolated'].shift(-k).fillna(True).astype(bool)
        df[f'target_glucose_{minutes}'] = target.where(~invented)

    return df

