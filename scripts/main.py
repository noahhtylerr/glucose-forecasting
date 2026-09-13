import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR, LOCAL_TZ
from src.loading import load_cgm_csv, load_events_csv, load_nightscout_events_csv
from src.pipeline import build_period


# ---------- CGM: both periods from Clarity ----------
april_cgm = load_cgm_csv(RAW_DIR / 'clarity_cgm_april_may.csv', source='clarity')
sept_cgm = load_cgm_csv(RAW_DIR / 'clarity_cgm_sept1-13.csv', source='clarity')

print('april cgm ', len(april_cgm), april_cgm['timestamp'].min(), '->', april_cgm['timestamp'].max())
print('sept cgm  ', len(sept_cgm), sept_cgm['timestamp'].min(), '->', sept_cgm['timestamp'].max())

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

# nothing from the holdout window may enter the development set
DEV_END = pd.Timestamp('2026-09-14', tz=LOCAL_TZ)
sept_cgm = sept_cgm[sept_cgm['timestamp'] < DEV_END]
events = events[events['timestamp'] < DEV_END]
print('dev ends  ', sept_cgm['timestamp'].max())

april_df = build_period(april_cgm, events, 'april')
sept_df = build_period(sept_cgm, events, 'sept')

df = pd.concat([april_df, sept_df], ignore_index=True)
df.to_parquet(PROCESSED_DIR / 'development.parquet')
