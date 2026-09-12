from src.config import RAW_DIR
from src.nightscout_scraper import fetch_range

# A day wide on each end -- timezone offsets make the boundaries fuzzy
START = '2026-08-31'
END = '2026-09-14'

treatments = fetch_range('treatments', 'created_at', START, END)
treatments.to_csv(RAW_DIR / 'nightscout_treatments_sept.csv', index=False)

print()
print('treatments', len(treatments))
print(treatments.columns.tolist())