# Download 1M resolution GHI data from florida and flesland via Frost API 


import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm


# Your Frost API client ID
CLIENT_ID = 'f8cf8726-6617-4696-90a4-5b89a668073a'

# Station IDs and names mapping
stations = {
    'SN50539': "Florida",
    'SN50500': 'Flesland',
    'SN50540' : "Florida",
}

# Parameters
element = 'mean(surface_downwelling_shortwave_flux_in_air PT1M)'
resolution = 'PT1M'
start_date_flesland = '2015-08-03' # First obs in flesland
start_date_florida_1 = '2016-02-11'
end_date_florida_1 = '2025-01-31'
start_date_florida_2 = '2025-01-21'
end_date = '2025-08-31'

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

def filter_utc_window(df):
    """Keep only times between 10:30 and 11:30 UTC"""
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    return df[((df['hour'] == 10) & (df['minute'] >= 30)) |
              ((df['hour'] == 11) & (df['minute'] < 30))].copy()


def download_monthly_data(station_id, start_dt, end_dt):
    """Download data for one station in monthly chunks"""
    all_data = []
    station_name = stations[station_id]
    for month_start in tqdm(daterange(start_dt, end_dt), desc=f"Downloading from {station_id}..."):
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

        tqdm.write(f"Downloading {station_id}:{station_name} data from {time_range}...")

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
            tqdm.write(f"No data for {station_id} in {time_range}")
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

        # Filter timestamps by UTC time 10:30-11:30
        df_filtered = filter_utc_window(df)
        
        # Add station column
        df_filtered['station'] = station_name
        df_filtered['station_id'] =station_id
        
        # Select relevant columns
        df_filtered = df_filtered[['timestamp', 'value', 'station', 'station_id']]
        
        all_data.append(df_filtered)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['timestamp', 'value', 'station', 'station_id'])
    
def main():
    # Flesland
    #  "sourceId": "SN50500:0",
    # "validFrom": "2015-08-03T00:00:00.000Z",
    # Florida 1 
    # "sourceId": "SN50539:0",
    #  "validFrom": "2016-02-11T00:00:00.000Z",
    #  "validTo": "2025-01-31T00:00:00.000Z",
    # Florida 2
    # "sourceId": "SN50540:0",
    # "validFrom": "2025-01-21T00:00:00.000Z",
    start_dt_flesland = datetime.strptime(start_date_flesland, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    start_dt_florida_1 = datetime.strptime(start_date_florida_1, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    end_dt_florida_1 = datetime.strptime(end_date_florida_1, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    start_dt_florida_2 = datetime.strptime(start_date_florida_2, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)

    all_stations_data = []

    # Flesland full range
    df_flesland = download_monthly_data('SN50500', start_dt_flesland, end_dt)
    all_stations_data.append(df_flesland)

    # Florida until 2025-01-31
    df_florida_1 = download_monthly_data('SN50539', start_dt_florida_1, end_dt_florida_1)
    all_stations_data.append(df_florida_1)

    # Florida from 2025-01-21 onward
    df_florida_2 = download_monthly_data('SN50540', start_dt_florida_2, end_dt)
    all_stations_data.append(df_florida_2)

    # Combine
    df_all = pd.concat(all_stations_data, ignore_index=True)

    # Save to CSV
    outpath = f'data/processed/frost_ghi_1M_Flesland_Florida_10:30-11:30UTC.csv'
    df_all.to_csv(outpath, index=False)
    print(f"Saved filtered data to {outpath}.")



if __name__ == '__main__':
    main()
