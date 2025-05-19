import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Your Frost API client ID
CLIENT_ID = 'f8cf8726-6617-4696-90a4-5b89a668073a'

# Station IDs
stations = ['SN50500']  # Florida SN50540, Flesland SN50500

# Parameters
element = 'cloud_area_fraction'
resolution = 'PT10M'
start_date = '2015-06-01'
end_date = '2025-05-01'

# Function to build request
def fetch_data(station_id):
    url = 'https://frost.met.no/observations/v0.jsonld'
    headers = {'accept': 'application/json'}

    params = {
        'sources': station_id,
        'elements': element,
        'referencetime': f'{start_date}/{end_date}',
        'timeresolutions': resolution,
        'fields': 'referenceTime,value',
        'limit': 1000
    }

    response = requests.get(url, auth=(CLIENT_ID, ''), params=params)
    response.raise_for_status()
    data = response.json()

    observations = data.get('data', [])
    records = []
    for item in observations:
        time_utc = datetime.fromisoformat(item['referenceTime'].replace('Z', '+00:00'))
        value = item['observations'][0]['value']
        records.append((time_utc, value))

    df = pd.DataFrame(records, columns=['UTC_time', 'cloud_area_fraction'])
    return df

# Convert and filter to local time (13–15)
def filter_by_local_time(df):
    oslo = pytz.timezone('Europe/Oslo')
    df['Local_time'] = df['UTC_time'].dt.tz_localize('UTC').dt.tz_convert(oslo)
    df_filtered = df[(df['Local_time'].dt.hour >= 12) & (df['Local_time'].dt.hour < 16)]
    return df_filtered

# Fetch and filter data
all_data = []
for station in stations:
    print(f'Fetching data for station {station}...')
    df = fetch_data(station)
    df_filtered = filter_by_local_time(df)
    df_filtered['station'] = station
    all_data.append(df_filtered)

# Combine and save
final_df = pd.concat(all_data)
final_df.to_csv('cloud_cover_12_16.csv', index=False)
print("Done! Data saved to cloud_cover_12_16.csv.")
