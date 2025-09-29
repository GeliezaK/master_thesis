import pandas as pd
import numpy as np
from netCDF4 import Dataset, num2date
from cftime import DatetimeGregorian

def merge_by_closest_timestamp(obs_table_path, frost_table_path):
    """Add GHI measurements from Florida and Flesland stations to obs_table based on closest timestamp.
    
    Only matches observations if the closest timestamp difference is <= 5 minutes.
    """

    # Read tables
    obs_table = pd.read_csv(obs_table_path)
    stations_data = pd.read_csv(frost_table_path)
    print(list(obs_table.columns.values))

    # Convert timestamps
    # obs_table: ms → datetime
    obs_table['obs_datetime'] = pd.to_datetime(obs_table['system:time_start_large'], unit='ms', utc=True)
    # stations_data: already datetime strings → datetime
    stations_data['timestamp'] = pd.to_datetime(stations_data['timestamp'], utc=True)

    # Prepare columns for merged data
    obs_table['Florida_ghi_1M'] = np.nan
    obs_table['Flesland_ghi_1M'] = np.nan

    # Maximum allowed time difference: 5 minutes
    max_delta = pd.Timedelta(minutes=5)

    # Separate stations
    florida_data = stations_data[stations_data['station'] == 'Florida'].copy()
    flesland_data = stations_data[stations_data['station'] == 'Flesland'].copy()

    # Function to find closest timestamp
    def closest_value(obs_time, station_df, ghi_col='value'):
        """Return GHI value from station_df with timestamp closest to obs_time,
        but only if within max_delta, else NaN."""
        if station_df.empty:
            return np.nan
        time_diffs = np.abs(station_df['timestamp'] - obs_time)
        idx_min = time_diffs.idxmin()
        if time_diffs.loc[idx_min] <= max_delta:
            return station_df.loc[idx_min, ghi_col]
        else:
            return np.nan

    # Apply for each observation
    obs_table['Florida_ghi_1M'] = obs_table['obs_datetime'].apply(lambda t: closest_value(t, florida_data))
    obs_table['Flesland_ghi_1M'] = obs_table['obs_datetime'].apply(lambda t: closest_value(t, flesland_data))
    obs_table = obs_table.drop('obs_datetime', axis=1)
    return obs_table


def extract_pixel_by_location(nc_filepath, pixel_lat, pixel_lon): 
    """Return the time series for a pixel given its location from a .nc file with dimensions [time, lat, lon]."""
    # Open your NetCDF file
    nc = Dataset(nc_filepath)

    ghi = nc.variables["GHI_total"]   # shape: (time, lat, lon)
    lats = nc.variables["lat"][:]
    lons = nc.variables["lon"][:]
    
    # Find nearest index for location
    ilat = np.abs(lats - pixel_lat).argmin()
    ilon = np.abs(lons - pixel_lon).argmin()
    
    print(f"Pixel ilat: {ilat}, lat: {lats[ilat]}; ilon: {ilon}, lon: {lons[ilon]}")
    
    # Extract time dimension
    time_var = nc.variables["time"]
    times = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

    # Extract pixel values
    ghi_timeseries = ghi[:, ilat, ilon]
    return times, ghi_timeseries
    
def convert_cftime_to_datetime(val):
    if isinstance(val, DatetimeGregorian):
        # Convert to Python datetime via pd.Timestamp
        return pd.Timestamp(val.isoformat())
    return val

def add_simulated_florida_flesland_ghi_by_date(obs_table_path, df, out_path):
    """
    Merge Florida and Flesland GHI into Sentinel-2 observations by matching on date (UTC).
    If no match exists for a given date, the GHI columns remain NaN.
    """
    # Load Sentinel-2 table
    obs = pd.read_csv(obs_table_path)

    # Convert to datetime
    obs["date"] = pd.to_datetime(obs["date"]).dt.date  

    # Ensure df["time"] is datetime and extract date
    df["time"] = df["time"].apply(convert_cftime_to_datetime)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date
    
    # Drop duplicates (if they exist, values are identical anyway)
    df_unique = df.drop_duplicates(subset="date")[["date", "Florida_ghi_sim_ECAD", "Flesland_ghi_sim_ECAD"]]

    # Left join on date → preserves all obs rows, NaN if no match
    merged = obs.merge(df_unique, on="date", how="left")

    # Save to CSV
    merged.to_csv(out_path, index=False)

    return merged



if __name__ == "__main__": 
    obs_table_path = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    frost_table_path = "data/processed/frost_ghi_1M_Flesland_Florida_10:30-11:30UTC.csv"
    model_ghi_path = "data/processed/simulated_ghi.nc"
    obs_with_stations_data_path = "data/processed/s2_cloud_cover_table_small_and_large_with_simulated_florida_flesland_ghi.csv"
    
    # Step 1: Merge Stations data with cloud_cover aso observations table
    """ obs_table = merge_by_closest_timestamp(obs_table_path, frost_table_path)
    print(obs_table.head())
    print(obs_table[["Florida_ghi_1M", "Flesland_ghi_1M"]].describe())
    print(obs_table.dtypes)
    
    obs_table.to_csv("data/processed/s2_cloud_cover_table_small_and_large_with_stations_data.csv", index=False)"""
    
    # Step 2: Extract Florida and Flesland pixels from model output
    # Florida and Flesland coordinates source: frost.met.no
    # frost: SN50539: 60.3837, 5.332 # Florida 1 until 2025, Plotting source unknown: 60.3833, 5.3333
    #frost: SN50540: 60.383, 5.3327 (typo?) # Florida 2 since 2025, previously 60.3833, 5.3333 # ECAD: 60.38306, 5.33306
    # Flesland, frost: 60.2892, 5.2265; ECAD: 60.28917, 5.22639, plotting source unknown: 60.292792, 5.222689
    Florida_lat, Florida_lon = 60.38306, 5.33306 # source: ECAD
    Flesland_lat, Flesland_lon = 60.28917, 5.22639

    times, Florida_ghi = extract_pixel_by_location(model_ghi_path, Florida_lat, Florida_lon)
    times, Flesland_ghi = extract_pixel_by_location(model_ghi_path, Flesland_lat, Flesland_lon)
    df = pd.DataFrame({
        "time": times,
        "Florida_ghi_sim_ECAD": Florida_ghi,
        "Flesland_ghi_sim_ECAD": Flesland_ghi
    })
    out_path = "data/processed/s2_cloud_cover_table_small_and_large_with_simulated_florida_flesland_ghi.csv"
    merged = add_simulated_florida_flesland_ghi_by_date(obs_with_stations_data_path, df, out_path)

    
