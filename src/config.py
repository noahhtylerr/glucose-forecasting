from pathlib import Path
from zoneinfo import ZoneInfo

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'
SAMPLE_DIR = DATA_DIR / 'sample'

MODELS_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Time
LOCAL_TZ = ZoneInfo('America/New_York')
GRID_MINUTES = 5

# Phsiology
INSULIN_DIA_MIN = 240
MEAL_CARB_DIA_MIN = 180
CORRECTION_CARB_DIA_MIN = 60

# Data Quality
MAX_INTERPOLATION_GAP_MIN = 20
GLUCOSE_MIN_VALID = 20
GLUCOSE_MAX_VALID = 600
HYPO_THRESHOLD = 70

# Prediction
HORIZONS_MIN = [15, 30, 45, 60]
RANDOM_SEED = 42


