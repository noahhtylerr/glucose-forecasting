from src.timeline import to_grid
from src.events import add_events
from src.physiology import add_iob_cob
from src.features import build_features


# Periods are built separately so lags, rolling windows, and targets never reach across the gap between them.
def build_period(cgm, events, label):
    grid = to_grid(cgm)
    grid = add_events(grid, events)
    grid = add_iob_cob(grid)

    df = build_features(grid, events)
    df['period'] = label

    return df