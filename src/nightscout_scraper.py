import hashlib
import os
import time

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.environ.get('NIGHTSCOUT_URL', '').rstrip('/')
API_SECRET = os.environ.get('NIGHTSCOUT_API_SECRET', '')


def _headers():
    return {'api-secret': hashlib.sha1(API_SECRET.encode()).hexdigest()}


def _get(endpoint, params):
    if not BASE_URL or not API_SECRET:
        raise RuntimeError('Set NIGHTSCOUT_URL and NIGHTSCOUT_API_SECRET first')

    # Pull logs from nightscout
    url = f'{BASE_URL}/api/v1/{endpoint}.json'
    response = requests.get(url, params=params, headers=_headers(), timeout=30)
    response.raise_for_status()

    return response.json()


# Pulls one day at a time to avoid silently dropping old data in a wide window
def fetch_day(endpoint, date_field, day):
    # start on the day being pulled and retrieve 24 hours of logs
    start = day.strftime('%Y-%m-%d')
    end = (day + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # pull all records within the time frame
    params = {
        'count': 10000,
        f'find[{date_field}][$gte]': start,
        f'find[{date_field}][$lt]': end,
    }

    return _get(endpoint, params)


def fetch_range(endpoint, date_field, start_date, end_date):
    days = pd.date_range(start_date, end_date, freq='D')

    records = []
    for day in days:
        batch = fetch_day(endpoint, date_field, day)
        print(day.date(), len(batch)) # identify any days with zero entries
        records.extend(batch)
        time.sleep(0.2)

    return pd.DataFrame(records)