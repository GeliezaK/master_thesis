# Download 1M resolution GHI data from florida and flesland via Frost API 

import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Your Frost API client ID
CLIENT_ID = 'f8cf8726-6617-4696-90a4-5b89a668073a'

# Station IDs and names mapping
stations = {
    #'SN50539': 'Florida_UiB',
    'SN50500': 'Flesland'
}

# Parameters
element = 'mean(surface_downwelling_shortwave_flux_in_air PT1M)'
resolution = 'PT1M'
start_date = '2016-01-01'
end_date = '2024-12-31'

# Timezone for Bergen
bergen_tz = pytz.timezone('Europe/Oslo')

# Frost API base URL
BASE_URL = 'https://frost.met.no/observations/v0.jsonld'

def daterange(start_dt, end_dt):
    """Generate first day of each month between start_dt and end_dt"""
    current = start_dt.replace(day=1)
    while current <= end_dt:
        yield current
        # move to first day of next month
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)
            
def get_value(obs_list):
    if isinstance(obs_list, list) and len(obs_list) > 0 and isinstance(obs_list[0], dict):
        return obs_list[0].get('value')
    return None

def download_monthly_data(station_id, start_dt, end_dt):
    """Download data for one station in monthly chunks"""
    all_data = []
    for month_start in daterange(start_dt, end_dt):
        # Calculate month end (last day of month)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year+1, month=1) - timedelta(days=1)
        else:
            month_end = month_start.replace(month=month_start.month+1) - timedelta(days=1)

        # Format times as ISO8601 date strings
        time_range = f"{month_start.strftime('%Y-%m-%d')}/{month_end.strftime('%Y-%m-%d')}"

        # Build query parameters
        params = {
            'sources': station_id,
            'referencetime': time_range,
            'elements': element,
            #'timeresolutions': resolution,
            #'timeoffsets': 'PT0S',  # no offset, raw time
            'limit': 100000,  # max per request (Frost default)
        }

        print(f"Downloading {station_id} data from {time_range}...")

        try:
            response = requests.get(BASE_URL, params=params, auth=(CLIENT_ID, ''))
            response.raise_for_status()
            data_json = response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error for {station_id} {time_range}: {http_err}")
            continue
        except Exception as err:
            print(f"Unexpected error for {station_id} {time_range}: {err}")
            continue

        # Extract observations
        observations = data_json.get('data', [])

        if not observations:
            print(f"No data for {station_id} in {time_range}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(observations)

        # Some API responses have 'observations' nested — flatten if needed
        if 'observations' in df.columns:
            df['value'] = df['observations'].apply(get_value)
            df = df.drop(columns=["observations"])

        # Extract needed columns - 'referenceTime' and 'observations' value
        # Here 'referenceTime' is the timestamp, and 'observations' field may have 'value'
        if 'referenceTime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['referenceTime'])
        else:
            print("No 'referenceTime' column in data")
            continue

        # The value is typically nested inside 'observations' dict - try to extract
        if 'value' in df.columns:
            df['value'] = df['value']
        else:
            # Try to find 'value' inside nested dict, fallback if missing
            if 'observations' in observations[0]:
                df['value'] = df['0'].apply(lambda x: x.get('value') if isinstance(x, dict) else None)
            else:
                df['value'] = None

        # Filter timestamps by local Bergen time 12:00-16:00
        df['local_time'] = df['timestamp'].dt.tz_convert(bergen_tz)
        df_filtered = df[(df['local_time'].dt.hour >= 12) & (df['local_time'].dt.hour < 16)].copy()

        # Select relevant columns
        df_filtered = df_filtered[['local_time', 'value']]

        all_data.append(df_filtered)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['local_time', 'value'])

def main():
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)

    all_stations_data = []

    for station_id in stations.keys():
        df_station = download_monthly_data(station_id, start_dt, end_dt)
        all_stations_data.append(df_station)

    # Combine all stations data
    df_all = pd.concat(all_stations_data, ignore_index=True)

    # Save to CSV
    outpath='data/processed/frost_ghi_1M_Flesland_filtered.csv'
    df_all.to_csv(outpath, index=False)
    print(f"Saved filtered data to {outpath}.")

if __name__ == '__main__':
    main()
