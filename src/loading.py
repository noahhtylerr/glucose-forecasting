from pathlib import Path
import pandas as pd
from src.config import LOCAL_TZ 

# Load CGM readings from CSV with columns timestamp and glucose 
def load_cgm_csv(path: str | Path, source: str):
    df = pd.read_csv(path, encoding='utf-8-sig', usecols=['timestamp', 'glucose'])

    df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    df['source'] = source

    df = df.dropna(subset=['timestamp', 'glucose'])

    # Duplicate timestamps mean a later upload corrected an earlier one
    df = df.drop_duplicates(subset='timestamp', keep='last')

    return df.sort_values('timestamp').reset_index(drop=True)

# Load events data from Nightscout CSV including columns: timestamp, event_type,\
# carbs, insulin, duration, and notes
def load_events_csv(path: str | Path):
    raw = pd.read_csv(path, encoding='utf-8-sig')

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
                       .dt.tz_convert(LOCAL_TZ),
        "event_type": raw["eventType"].astype("string").str.strip(),
        # Blank carbs on a bolus means zero carbs, not unknown
        "carbs": pd.to_numeric(raw["carbs"], errors="coerce").fillna(0.0),
        "insulin": pd.to_numeric(raw["insulin"], errors="coerce").fillna(0.0),
        "duration": pd.to_numeric(raw["duration"], errors="coerce").fillna(0.0),
        # Exercise features key off note text, so normalize case 
        "notes": raw["notes"].astype("string").fillna("").str.strip().str.lower(),
    })

    df = df.dropna(subset=["timestamp"])

    return df.sort_values("timestamp").reset_index(drop=True)


