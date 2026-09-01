import pandas as pd
from src.config import GRID_MINUTES, MAX_INTERPOLATION_GAP_MIN

def to_grid(cgm):
    df = cgm.copy()
    df = df.sort_values('timestamp').set_index('timestamp')

    # Put readings on an even, 5 minute grid average any that share a timeslot
    df = df[['glucose']].resample(f'{GRID_MINUTES}min').mean()

    df['is_interpolated'] = df['glucose'].isna()

    # Only fill short gaps (4 slots) and only fill blank slots inside of readings
    limit = MAX_INTERPOLATION_GAP_MIN // GRID_MINUTES
    df['glucose'] = df['glucose'].interpolate(limit=limit, limit_area='inside')

    df['is_interpolated'] = df['is_interpolated'] & df['glucose'].notna()

    df = df.reset_index()
    return df


