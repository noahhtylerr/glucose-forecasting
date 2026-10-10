from src.config import RAW_DIR
from src.nightscout_scraper import fetch_range

START = '2026-09-24'
END = '2026-10-09'

treatments = fetch_range('treatments', 'created_at', START, END)
treatments.to_csv(RAW_DIR / 'nightscout_treatments_holdout.csv', index=False)

print()
print('treatments', len(treatments))
print(treatments.columns.tolist())