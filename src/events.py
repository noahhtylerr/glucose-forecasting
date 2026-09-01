import pandas as pd
from src.config import GRID_MINUTES

def add_events(grid, events):
    ev = events.copy()

    ev['rounded_timestamp'] = ev['timestamp'].dt.round(f'{GRID_MINUTES}min')

    # Apply specific carb counts and insulin for different event types
    is_meal = ev['event_type'].isin(['Meal Bolus', 'Snack Bolus'])
    ev['meal_carbs'] = ev['carbs'].where(is_meal, 0)

    is_correction = ev['event_type'].isin(['Carb Correction'])
    ev['correction_carbs'] = ev['carbs'].where(is_correction, 0)
    ev['bolus_insulin'] = ev['insulin']

    # Combines any events that land in the same time slot
    amounts = ['meal_carbs', 'correction_carbs', 'bolus_insulin']
    ev = ev.groupby('rounded_timestamp')[amounts].sum().reset_index()

    ev = ev.rename(columns={'rounded_timestamp': 'timestamp'})

    # Left merge keeps all grid rows, attaching events where they exist
    df = grid.merge(ev, on='timestamp', how='left')

    # Fill carbs in rows without events with 0
    df[amounts] = df[amounts].fillna(0)

    # Ensures the merge does not fan out and add rows
    assert len(df) == len(grid)

    return df

