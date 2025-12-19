# ====================================================================
# Simulate instantaneous Global Horizontal Irradiance 
# ====================================================================

import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
from netCDF4 import Dataset, date2num
import xarray as xr
import os
import numpy as np
import pandas as pd
from src.model import MIXED_THRESHOLD, OVERCAST_THRESHOLD, COARSE_RESOLUTIONS
from src.model import FLESLAND_LAT, FLESLAND_LON, FLORIDA_LAT, FLORIDA_LON


def get_closest_lut_entry(lut, unique_values, doy, hour, albedo, altitude_km,
                          cloud_top_km=None, cot=None, cloud_phase=None, verbose=False):
    """
    Return LUT entry closest to the given parameters.
    
    Parameters
    ----------
    lut : pd.DataFrame
        LUT table with all parameters and irradiance outputs.
    unique_values : dict
        Dictionary of unique values per LUT variable.
    doy, hour, albedo, altitude_km : float
        Mandatory input parameters.
    cloud_top_km, cot, cloud_phase : float, optional
        Cloud parameters; if None, returns clear-sky values only.
    
    Returns
    -------
    dict
        {'direct_clear': ..., 'diffuse_clear': ..., 'direct_cloudy': ..., 'diffuse_cloudy': ...}
        If cloud parameters are not provided, cloudy values are set to None.
    """
    
    # Assert input params are correct
    if cloud_top_km is not None:
        assert cloud_top_km <= 50, f"Cloud Top Height is larger than 50 km ({cloud_top_km})! Did you convert m to km?"
        
    if cloud_phase is not None:
        assert isinstance(cloud_phase, str), f"Cloud phase should be a string!"
        assert (cloud_phase == "water") or (cloud_phase == "ice"), f"Cloud phase should be either 'water', 'ice' or None! Current value: {cloud_phase}."
    
    if cot is not None: 
        assert 0.0 <= cot <= 300, f"COT is expected to be within range 0-300 (unitless). Current value: {cot}."
        
    assert 1 <= doy <= 366, f"Doy {doy} is out of range! Expected values between 1-366."
    assert 0 <= hour <= 24, f"Hour of day {hour} is out of range! Expected values between 0 and 24. Remember hour is UCT. "
    assert 0.0 <= albedo <= 1.0, f"Albedo {albedo} is out of range! Values should be between 0 and 1."
    assert 0.0 <= altitude_km <= 5.0, f"Altitude {altitude_km} is out of range! Expected values between 0 and 5.0. Did you convert m to km? "
    
    # Find closest values in LUT bins
    def closest(value, array):
        array = np.array(array)

        # If value is a string → look for exact match
        if isinstance(value, str):
            matches = array[array == value]
            if len(matches) == 0:
                raise ValueError(f"No exact match found for string '{value}' in array")
            return matches[0]

        # Otherwise → assume numeric → closest value
        else:
            idx = np.argmin(np.abs(array - value))
            return array[idx]
    
    doy_bin = closest(doy, unique_values['doy'])
    hour_bin = closest(hour, unique_values['hour'])
    albedo_bin = closest(albedo, unique_values['albedo'])
    alt_bin = closest(altitude_km, unique_values['altitude_km'])
    
    # Optional cloud parameters
    cloud_top_bin = closest(cloud_top_km, unique_values['cloud_top_km']) if cloud_top_km is not None else None
    tau_bin = closest(cot, unique_values['cot']) if cot is not None else None
    cloud_type_bin = closest(cloud_phase, unique_values['cloud_phase']) if cloud_phase is not None else None
    
    if verbose: 
        print(f"Closest LUT bin found: {doy_bin}, {hour_bin}, {albedo_bin}, {alt_bin}, cth: {cloud_top_bin:.3f}, cot: {tau_bin}, cph: {cloud_type_bin}")

    # Filter LUT by closest bins
    df_filtered = lut[
        (lut['doy'] == doy_bin) &
        (lut['hour'] == hour_bin) &
        (lut['albedo'] == albedo_bin) &
        (lut['altitude_km'] == alt_bin)
    ]
    
    # If cloud parameters are provided, further filter
    if cloud_top_bin is not None:
        df_filtered = df_filtered[
            (df_filtered['cloud_top_km'] == cloud_top_bin) &
            (df_filtered['cot'] == tau_bin) &
            (df_filtered['cloud_phase'] == cloud_type_bin)
        ]
        
    # Return the irradiance values
    if len(df_filtered) == 0:
        # No match found, return None
        return {'direct_clear': None,
                'diffuse_clear': None,
                'direct_cloudy': None,
                'diffuse_cloudy': None}
    
    # Take the first row (or you could take mean if multiple matches)
    row = df_filtered.iloc[0]
    
    direct_clear = row['direct_clear']
    diffuse_clear = row['diffuse_clear']
    
    if cloud_top_bin is not None:
        direct_cloudy = row['direct_cloudy']
        diffuse_cloudy = row['diffuse_cloudy']
    else:
        direct_cloudy = None
        diffuse_cloudy = None
    
    res_dict = {'direct_clear': direct_clear,
            'diffuse_clear': diffuse_clear,
            'direct_cloudy': direct_cloudy,
            'diffuse_cloudy': diffuse_cloudy}
    
    return res_dict


def build_ghi_clear_lut(path = "data/processed/LUT/claas3/LUT.csv"):
    """Build full LUT for clear sky entries as np.array (faster than multiple
    .csv file reads or keeping the full LUT in memory)."""
    
    ghi_clear_table = np.full((365, 24), np.nan)
    lut = pd.read_csv(path)
    variables = ["doy", "hour", "albedo", "altitude_km", "cloud_top_km", "cot", "cloud_phase"]
    unique_values = {var: lut[var].unique() for var in variables if var in lut.columns}
    surface_albedo = 0.129
    altitude = 0.08 
    
    for doy in range(1, 366):
        for hour in range(24):
            res = get_closest_lut_entry(lut, unique_values, doy, hour, surface_albedo, altitude)
            if res["direct_clear"] is not None:
                ghi_clear_table[doy-1, hour] = (
                    res["direct_clear"] + res["diffuse_clear"]
                )
    return ghi_clear_table


def idx_from_time(times_array, timestamp, by_time_of_year=False, max_delta=1.0, verbose=False):
    """
    Given a timestamp, find the index of the closest observation in times_array.
    
    Parameters
    ----------
    times_array : np.ndarray
        Array of np.datetime64 timestamps.
    timestamp : datetime-like
        Target time to match.
    by_time_of_year : bool, optional
        If True, match by day-of-year first (±16 days allowed), then by hour-of-day (within max_delta hours).
    max_delta : float, optional
        Maximum allowed time difference in hours (for hour-of-day comparison).
    verbose : bool, optional
        Print additional information.
        
    Returns 
    -------
    idx : int
        Index of the closest observation to timestamp in times_array.
    """
    
    # Ensure timestamp is timezone-free and numpy datetime64
    if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    timestamp = np.datetime64(timestamp, "ns")

    if by_time_of_year:
        ts = pd.Timestamp(timestamp)
        target_doy = ts.day_of_year
        target_hour = ts.hour + ts.minute / 60.0

        pd_times = pd.to_datetime(times_array)
        doys = pd_times.day_of_year
        hours = pd_times.hour + pd_times.minute / 60.0

        # --- Step 1: Find closest day of year ---
        doy_diff = np.abs(doys - target_doy)
        doy_diff = np.minimum(doy_diff, 365 - doy_diff)  # handle wrap-around (Dec/Jan)
        doy_idx = np.argmin(doy_diff)
        min_doy_diff = doy_diff[doy_idx]

        if verbose:
            print(f"[by_time_of_year] Closest DOY: {doys[doy_idx]} (ΔDOY={min_doy_diff:.1f} days)")

        if min_doy_diff > 8:
            raise ValueError(
                f"No close match found (by_time_of_year): nearest DOY {doys[doy_idx]} "
                f"is {min_doy_diff:.1f} days from target {target_doy}."
            )

        # --- Step 2: Among entries with that DOY, find closest hour ---
        same_doy_mask = doy_diff == min_doy_diff
        hour_diff = np.abs(hours[same_doy_mask] - target_hour)
        hour_idx_local = np.argmin(hour_diff)
        min_hour_diff = hour_diff[hour_idx_local]

        # Map back to global index
        global_indices = np.where(same_doy_mask)[0]
        idx = global_indices[hour_idx_local]

        if verbose:
            print(f"[by_time_of_year] Closest hour: {hours[idx]:.2f} (Δhour={min_hour_diff:.2f} h) "
                  f"→ Closest time: {times_array[idx]}")

        if min_hour_diff > max_delta:
            raise ValueError(
                f"No close hour match: nearest timestamp {times_array[idx]} is "
                f"{min_hour_diff:.2f} hours from target hour {target_hour:.2f}."
            )

    else:
        # Standard direct timestamp matching
        idx = np.argmin(np.abs(times_array - timestamp))
        closest_time = times_array[idx]
        delta = np.abs(closest_time - timestamp).astype("timedelta64[m]").astype(float) / 60.0

        if verbose:
            print(f"Closest time to {timestamp} is {closest_time} ({delta:.3f} hours apart).")

        if delta >= max_delta:
            raise ValueError(
                f"No close match found: nearest timestamp {closest_time} is {delta:.2f} hours from {timestamp}."
            )

    return idx


def get_cloud_properties(row, monthly_medians, month, verbose=False):
    """
    Get representative cloud optical and physical properties for a single observation.

    This function extracts cloud optical thickness (COT), cloud top height (CTH),
    and cloud phase (CPH) from a pandas DataFrame row. If small-roi (region-of-interest)
    values are missing, large-roi values are used as a fallback.
    If both are unavailable, monthly median values are used.

    Parameters
    ----------
    row : pandas.Series
        A row from a pandas DataFrame containing cloud property statistics.
        Expected fields include:
        - 'cot_median_small', 'cot_median_large'
        - 'cth_median_small', 'cth_median_large'
        - 'cph_median_small', 'cph_median_large'

    monthly_medians : pandas.DataFrame
        DataFrame containing monthly median cloud properties used as fallback.
        Must include the columns:
        - 'month'
        - 'cot_median_small'
        - 'cth_median_small'
        - 'cph_median_small'

    month : int
        Month index (1–12) used to select fallback values from `monthly_medians`.

    verbose : bool, optional
        If True, print the selected cloud properties to stdout.
        Default is False.

    Returns
    -------
    COT : float
        Cloud Optical Thickness (dimensionless).

    CTH : float
        Cloud Top Height in kilometers.

    CPH : str
        Cloud Phase, either "water" or "ice".
    """
    
    # --- COT ---
    if not pd.isna(row["cot_median_small"]):
        COT = row["cot_median_small"]
    elif not pd.isna(row["cot_median_large"]):
        COT = row["cot_median_large"]
    else:
        COT = monthly_medians.loc[monthly_medians['month'] == month, "cot_median_small"].values[0]
        
    # --- CTH ---
    if not pd.isna(row["cth_median_small"]):
        CTH = row["cth_median_small"]
    elif not pd.isna(row["cth_median_large"]):
        CTH = row["cth_median_large"]
    else:
        CTH = monthly_medians.loc[monthly_medians['month'] == month, "cth_median_small"].values[0]
    
    CTH = CTH / 1000.0  # convert to km

    # --- CPH ---
    if not pd.isna(row["cph_median_small"]):
        CPH = "water" if row["cph_median_small"] <= 1.5 else "ice"
    elif not pd.isna(row["cph_median_large"]):
        CPH = "water" if row["cph_median_large"] <= 1.5 else "ice"
    else:
        monthly_cph = monthly_medians.loc[monthly_medians['month'] == month, "cph_median_small"].values[0]
        CPH = "water" if monthly_cph <= 1.5 else "ice"

    if verbose:
        print(f"cloud props: COT {COT:.2f}, CTH {CTH:.2f}, CPH {CPH}")
    return COT, CTH, CPH


def total_ghi_from_sat_imgs(cloud_shadow_path, cloud_cover_filepath, LUT_filepath,
                               sw_cor_path, out_nc_file,
                               mixed_threshold, overcast_threshold, verbose=False):
    """
    Compute total surface Global Horizontal Irradiance (GHI) from satellite-derived
    cloud information and radiative transfer lookup tables, and store the results
    incrementally in a NetCDF file.

    The function processes a time series of satellite observations and classifies
    each timestep into clear-sky, overcast, or mixed cloud conditions based on
    large-scale cloud cover thresholds. For each timestep, direct and diffuse
    irradiance components are obtained from a precomputed radiative transfer
    lookup table (LUT) and combined with spatial correction factors and cloud
    shadow information to derive total GHI fields.

    Results are written incrementally to a NetCDF file, allowing safe interruption
    and restart without recomputation of previously processed timesteps.

    Parameters
    ----------
    cloud_shadow_path : str
        Path to a NetCDF file containing time-resolved cloud shadow masks
        (boolean or binary), including dimensions ``time``, ``lat``, and ``lon``.

    cloud_cover_filepath : str
        Path to a CSV file containing per-timestep cloud properties and statistics
        derived from satellite imagery. Must include observation timestamps,
        cloud cover fractions, cloud optical properties, and surface albedo
        information.

    LUT_filepath : str
        Path to a CSV file containing a radiative transfer lookup table (LUT)
        with clear-sky and cloudy-sky direct and diffuse irradiance components.

    sw_cor_path : str or None
        Path to a NetCDF file containing shortwave direct irradiance correction
        factors accounting for topography or terrain shading. If ``None``, no
        correction is applied (output variable is global horizontal irradiance) 
        and clear-sky timesteps are skipped.

    out_nc_file : str
        Path to the output NetCDF file where total GHI fields are written.
        The file is created if it does not exist and appended to otherwise.

    mixed_threshold : float
        Cloud cover fraction below which a timestep is classified as clear-sky.
        Values between ``mixed_threshold`` and ``overcast_threshold`` are treated
        as mixed sky type. Values lower than ``mixed_threshold`` are 
        classified as clear sky type.

    overcast_threshold : float
        Cloud cover fraction above which a timestep is classified as overcast.

    verbose : bool, optional
        If True, print progress information and diagnostic messages during
        processing. Default is False.

    Outputs
    -------
    NetCDF file
        A NetCDF file written to ``out_nc_file`` containing:
        - ``time`` : time dimension in hours since a reference date
        - ``lat``  : latitude coordinates
        - ``lon``  : longitude coordinates
        - ``GHI_total`` : total surface Global Horizontal Irradiance
          (W m⁻²) for each timestep and grid cell

    Notes
    -----
    - Each timestep is processed independently and written immediately to disk,
      enabling efficient handling of large datasets.
    - Existing timesteps in the output NetCDF file are detected and skipped to
      avoid duplication.
    - Cloud optical thickness, cloud top height, and cloud phase are inferred
      per timestep with fallback to monthly climatological medians when necessary.
    """

    # -----------------------------------------------------------------------------
    # Initialize
    # -----------------------------------------------------------------------------
    cloud_props = pd.read_csv(cloud_cover_filepath)
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    cloud_props["month"] = cloud_props["date"].dt.month

    # Default monthly medians
    monthly_medians = (
        cloud_props.groupby("month")[["cot_median_small","cth_median_small","cph_median_small"]]
        .median().reset_index()
    )

    # Surface albedo
    surface_albedo = cloud_props["blue_sky_albedo_median"].mean()

    # Read LUT and unique values
    lut = pd.read_csv(LUT_filepath)
    variables = ["doy", "hour", "albedo", "altitude_km", "cloud_top_km", "cot", "cloud_phase"]
    unique_values = {var: lut[var].unique() for var in variables if var in lut.columns}

    # SW correction factors
    if sw_cor_path is not None:
        ds_dir_cor = xr.open_dataset(sw_cor_path)
        dir_cor_times = ds_dir_cor["time"].values
        dir_cor_lat = ds_dir_cor["lat"].values
        dir_cor_lon = ds_dir_cor["lon"].values
        shape_dir_cor_lat = len(dir_cor_lat)
        shape_dir_cor_lon = len(dir_cor_lon)
    
    # Cloud shadows
    ds_cloud_shadow = xr.open_dataset(cloud_shadow_path)
    cloud_shadow_times = ds_cloud_shadow["time"].values
    cloud_shadow_lat = ds_cloud_shadow["lat"].values
    cloud_shadow_lon = ds_cloud_shadow["lon"].values
    shape_cloud_shadow_lat = len(cloud_shadow_lat)
    shape_cloud_shadow_lon = len(cloud_shadow_lon)

    # -----------------------------------------------------------------------------
    # Create NetCDF file if it does not exist
    # -----------------------------------------------------------------------------
    if sw_cor_path is not None: 
        shape_lat = shape_dir_cor_lat
        shape_lon = shape_dir_cor_lon
        lat = dir_cor_lat
        lon = dir_cor_lon
    else: 
        shape_lat = shape_cloud_shadow_lat
        shape_lon = shape_cloud_shadow_lon
        lat = cloud_shadow_lat
        lon = cloud_shadow_lon
        
    if verbose: 
        print(f"Lat: {shape_lat}, lon:{shape_lon}")
        
    if not os.path.exists(out_nc_file):
        os.makedirs(os.path.dirname(out_nc_file), exist_ok=True)
        ncfile = Dataset(out_nc_file, mode="w")
        ncfile.createDimension("time", size=None)
        ncfile.createDimension("lat", size=shape_lat)
        ncfile.createDimension("lon", size=shape_lon)

        # Time variable
        nc_time = ncfile.createVariable("time", "f8", ("time",))
        nc_time.units = "hours since 2015-01-01 00:00:00"
        nc_time.calendar = "gregorian"

        # Lat/Lon
        nc_lat = ncfile.createVariable("lat", "f4", ("lat",))
        nc_lat[:] = lat
        nc_lat.units = "degree"
        nc_lon = ncfile.createVariable("lon", "f4", ("lon",))
        nc_lon[:] = lon
        nc_lon.units = "degree"

        # GHI variable
        nc_ghi = ncfile.createVariable("GHI_total", "f4", ("time", "lat", "lon"), zlib=True)
        nc_ghi.units = "W/m2"
        nc_ghi.long_name = "Total surface GHI including cloud effects"

        ncfile.close()

    # -----------------------------------------------------------------------------
    # Open NetCDF in append mode and get existing times
    # -----------------------------------------------------------------------------
    ncfile = Dataset(out_nc_file, mode="a")
    existing_times = ncfile.variables["time"][:]
    # Convert existing numeric times to datetime (tz-naive)
    existing_datetimes = pd.to_datetime(existing_times, origin=pd.Timestamp("2015-01-01 00:00:00"), unit='h')
    existing_dates = existing_datetimes.date  # numpy array of datetime.date
    ncfile.close()
    
    
    print(f"Number of observations saved in {out_nc_file}: {len(existing_times)}")
    
    # -----------------------------------------------------------------------------
    # Iterate and write per timestep
    # -----------------------------------------------------------------------------
    for idx, row in tqdm(cloud_props.iterrows(), total=len(cloud_props), desc="Processing rows"):
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)
        
        # Skip if date already in NetCDF
        # Make dt tz-naive for comparison with existing_datetimes
        dt_naive = dt.tz_convert(None) if hasattr(dt, 'tz') else dt
        obs_date = dt_naive.date()  

        # Check if the date already exists
        if obs_date in existing_dates:
            if verbose:
                tqdm.write(f"Skipping {dt}, date {obs_date} already exists in NetCDF.")
            continue
        
        month = row["month"]
        doy = dt.timetuple().tm_yday
        hour = dt.hour + round(dt.minute/60)
        altitude = 0.08  # km

        cloud_cover_large = row["cloud_cover_large"]

        # Initialize GHI_total
        GHI_total = None

        # ---------------- Clear Sky ----------------
        if cloud_cover_large <= mixed_threshold:
            # Get direct correction factor, or skip clear days
            if sw_cor_path is not None: 
                idx_sw = idx_from_time(dir_cor_times, dt,by_time_of_year=True, max_delta=1, verbose=verbose)
                dir_cor = ds_dir_cor["sw_dir_cor"].isel(time=idx_sw).values
            else : 
                continue
            
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude)
            direct_clear = res_dict["direct_clear"]
            diffuse_clear = res_dict["diffuse_clear"]
            if direct_clear is None or diffuse_clear is None:
                tqdm.write(f"Skip {dt}: no data returned. Direct clear: {direct_clear}, Diffuse clear: {diffuse_clear}.")
                tqdm.write(f"Inputs to LUT: DOY {doy}, hour {hour}.")
                continue
                
            GHI_total = dir_cor * direct_clear + diffuse_clear

        # ---------------- Overcast ----------------
        elif cloud_cover_large >= overcast_threshold:
            # Get direct correction factor, or skip overcast days
            if sw_cor_path is not None:
                idx_sw = idx_from_time(dir_cor_times, dt, by_time_of_year=True, max_delta=1, verbose=verbose)
                dir_cor = ds_dir_cor["sw_dir_cor"].isel(time=idx_sw).values
            else : 
                continue
            
            COT, CTH, CPH = get_cloud_properties(row, monthly_medians,month, verbose=verbose)
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude,
                                             CTH, COT, CPH, verbose=verbose)
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            if direct_cloudy is None or diffuse_cloudy is None:
                tqdm.write(f"Skip {dt}: no data returned. Direct cloudy: {direct_cloudy}, Diffuse cloudy: {diffuse_cloudy}.")
                tqdm.write(f"Inputs to LUT: DOY {doy}, hour {hour}, CTH {CTH}, COT {COT}, CPH {CPH}.")
                continue
     
            GHI_total = dir_cor * direct_cloudy + diffuse_cloudy

        # ---------------- Mixed ----------------
        else:
            try:
                # Read cloud shadow
                idx_cloud_shadow = idx_from_time(cloud_shadow_times, dt, max_delta=1/60, verbose=verbose)
            except ValueError as e: 
                # Skip if date is not found 
                tqdm.write(f"⏩ Skipping {dt}: {e}")
                continue 
            
            cloud_shadow_mask = ds_cloud_shadow["shadow_mask"].isel(time=idx_cloud_shadow).values
            
            # Get direct correction factor and match shapes
            if sw_cor_path is not None: 
                idx_sw = idx_from_time(dir_cor_times, dt, by_time_of_year=True, max_delta=1, verbose=verbose)
                dir_cor = ds_dir_cor["sw_dir_cor"].isel(time=idx_sw).values
                nrows, ncols = cloud_shadow_mask.shape
                target_rows, target_cols = dir_cor.shape
                
                # Assert only small shape differences and shadow mask smaller than sw_dir mask
                assert 0 <= (target_rows - nrows) < 5, f"Shape differences between cloud shadow : ({nrows},{ncols}) and sw dir cor : ({target_rows}, {target_cols})"
                assert 0 <= (target_cols - ncols) < 5, f"Shape differences between cloud shadow : ({nrows},{ncols}) and sw dir cor : ({target_rows}, {target_cols})"
                
                # Fill missing rows in cloud shadow with "clear" cells (no cloud shadow at border)
                pad_rows = target_rows - nrows
                pad_cols = target_cols - ncols
                cloud_shadow_mask = np.pad(cloud_shadow_mask, ((0,pad_rows),(0,pad_cols)),
                                 mode="constant", constant_values=False)
            else: 
                dir_cor = np.full(shape=(len(cloud_shadow_lat), len(cloud_shadow_lon)), fill_value=1.0)
               
            # get LUT entries
            COT, CTH, CPH = get_cloud_properties(row, monthly_medians, month, verbose=verbose)
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude,
                                             CTH, COT, CPH, verbose=verbose)
            
            direct_clear = res_dict["direct_clear"]
            diffuse_clear = res_dict["diffuse_clear"]
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            
            if (direct_clear is None or diffuse_clear is None or
                direct_cloudy is None or diffuse_cloudy is None):
                tqdm.write(f"Skip {dt}: no data returned. Direct clear: {direct_clear}," \
                           f"Diffuse clear: {diffuse_clear}, Direct cloudy: {direct_cloudy}, " \
                           f"Diffuse cloudy: {diffuse_cloudy}.")
                tqdm.write(f"Inputs to LUT: DOY {doy}, hour {hour}, CTH {CTH}, COT {COT}, CPH {CPH}.")
                continue
            
            # Compute Ghi total
            GHI_total = np.where(cloud_shadow_mask, dir_cor*direct_cloudy + diffuse_cloudy,
                                 dir_cor*direct_clear + diffuse_clear)
        
        if GHI_total is None:
            print(f"Skipped at {dt} because GHI_total is None. res_dict: {res_dict}")

        # Write to NetCDF and close after each iteration
        if GHI_total is not None and GHI_total.shape == (shape_lat, shape_lon):
            ncfile = Dataset(out_nc_file, mode="a")
            t_var = ncfile.variables["time"]
            i = len(t_var)  # next index
            t_var[i] = date2num(dt.to_pydatetime(), units=t_var.units, calendar=t_var.calendar)
            ncfile.variables["GHI_total"][i,:,:] = GHI_total
            ncfile.close()

    if sw_cor_path is not None:
        ds_dir_cor.close()
    ds_cloud_shadow.close()
    print(f"GHI maps saved incrementally to {out_nc_file}")
   
   
def simulate_stations_pixels_corrected(cloud_cover_filepath, lut_filepath, cloud_shadow_path, verbose=False):
    """Correct the simulation results for Flesland and Florida pixels (set direct shortwave correction factor
    to 1.0, i.e. calculate GHI). Add to the table in cloud_cover_filepath new columns florida_diffuse, 
    florida_direct, flesland_diffuse and flesland_direct to store results for these two pixels explicitely."""
    
    sim_vs_obs = pd.read_csv(cloud_cover_filepath)
    sim_vs_obs["date"] = pd.to_datetime(sim_vs_obs["date"], format="%Y-%m-%d")
    sim_vs_obs["month"] = sim_vs_obs["date"].dt.month
    # Prepare new columns for GHI components
    sim_vs_obs["total_clear_sky"] = np.nan
    sim_vs_obs["florida_direct"] = np.nan
    sim_vs_obs["florida_diffuse"] = np.nan
    sim_vs_obs["flesland_direct"] = np.nan
    sim_vs_obs["flesland_diffuse"] = np.nan
    sim_vs_obs["flesland_cloud_shadow"] = np.nan
    sim_vs_obs["flesland_cloud_shadow"] = np.where(
    sim_vs_obs["cloud_cover_large"] >= OVERCAST_THRESHOLD, 1, # always shadow for overcast 
        np.where(sim_vs_obs["cloud_cover_large"] <= MIXED_THRESHOLD, 0, np.nan) # never shadow for clear sky
    )
    sim_vs_obs["florida_cloud_shadow"] = sim_vs_obs["flesland_cloud_shadow"] # Copy 

    # Default monthly medians
    monthly_medians = (
        sim_vs_obs.groupby("month")[["cot_median_small","cth_median_small","cph_median_small"]]
        .median().reset_index()
    )

    # Surface albedo
    surface_albedo = sim_vs_obs["blue_sky_albedo_median"].mean()

    # Read LUT and unique values
    lut = pd.read_csv(lut_filepath)
    variables = ["doy", "hour", "albedo", "altitude_km", "cloud_top_km", "cot", "cloud_phase"]
    unique_values = {var: lut[var].unique() for var in variables if var in lut.columns}
    
    # Cloud shadows
    ds_cloud_shadow = xr.open_dataset(cloud_shadow_path)
    cloud_shadow_times = ds_cloud_shadow["time"].values
    lats = ds_cloud_shadow["lat"].values
    lons = ds_cloud_shadow["lon"].values

    flesland_ilat = np.abs(lats - FLESLAND_LAT).argmin()
    flesland_ilon = np.abs(lons - FLESLAND_LON).argmin()
    florida_ilat = np.abs(lats - FLORIDA_LAT).argmin()
    florida_ilon = np.abs(lons - FLORIDA_LON).argmin()
    
    if verbose: 
        print(f"Flesland ilat: {flesland_ilat}, lat: {lats[flesland_ilat]}; ilon: {flesland_ilon}, lon: {lons[flesland_ilon]}")
        print(f"Florida ilat: {florida_ilat}, lat: {lats[florida_ilat]}; ilon: {florida_ilon}, lon: {lons[florida_ilon]}")

    # For each observation in sim vs obs dataframe, get GHI values from LUT and append to df. 
    for idx, row in tqdm(sim_vs_obs.iterrows(), total=len(sim_vs_obs), desc="Processing rows"):
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)
        month = row["month"]
        doy = dt.timetuple().tm_yday
        hour = dt.hour + round(dt.minute/60)
        altitude = 0.08  # km
        cloud_cover_large = row["cloud_cover_large"]
        
        # Fill theoretical clear-sky values 
        res_dict_clear_sky = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude)
        direct_clear = res_dict_clear_sky["direct_clear"]
        diffuse_clear = res_dict_clear_sky["diffuse_clear"]
        sim_vs_obs.at[idx,"total_clear_sky"] = direct_clear + diffuse_clear
        
        # ---------------- Clear Sky ----------------
        if cloud_cover_large <= MIXED_THRESHOLD:
            if not direct_clear is None: 
                sim_vs_obs.at[idx,"florida_direct"] = direct_clear
                sim_vs_obs.at[idx,"flesland_direct"] = direct_clear
            if not diffuse_clear is None: 
                sim_vs_obs.at[idx,"florida_diffuse"] = diffuse_clear            
                sim_vs_obs.at[idx,"flesland_diffuse"] = diffuse_clear            
            
        # ---------------- Overcast ----------------
        elif cloud_cover_large >= OVERCAST_THRESHOLD:
            COT, CTH, CPH = get_cloud_properties(row, monthly_medians,month, verbose=verbose)
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude,
                                             CTH, COT, CPH, verbose=verbose)
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            
            if not direct_cloudy is None: 
                sim_vs_obs.at[idx,"florida_direct"] = direct_cloudy
                sim_vs_obs.at[idx,"flesland_direct"] = direct_cloudy
            if not diffuse_cloudy is None: 
                sim_vs_obs.at[idx,"florida_diffuse"] = diffuse_cloudy
                sim_vs_obs.at[idx,"flesland_diffuse"] = diffuse_cloudy
            
        # ---------------- Mixed ----------------
        else:
            try:
                # Read cloud shadow
                idx_cloud_shadow = idx_from_time(cloud_shadow_times, dt, max_delta=1/60, verbose=verbose)
            except ValueError as e: 
                # Skip if date is not found 
                tqdm.write(f"⏩ Skipping {dt}: {e}")
                continue 
            
            cloud_shadow_mask = ds_cloud_shadow["shadow_mask"].isel(time=idx_cloud_shadow).values
            # Extract pixel values at Flesland and Florida
            flesland_cloud_shadow = cloud_shadow_mask[flesland_ilat, flesland_ilon]
            florida_cloud_shadow = cloud_shadow_mask[florida_ilat, florida_ilon]

            if verbose: 
                print(f"Flesland shadow value: {flesland_cloud_shadow}")
                print(f"Florida shadow value: {florida_cloud_shadow}")
            
            sim_vs_obs.at[idx,"florida_cloud_shadow"] = florida_cloud_shadow
            sim_vs_obs.at[idx,"flesland_cloud_shadow"] = flesland_cloud_shadow
            
            # get LUT entries
            COT, CTH, CPH = get_cloud_properties(row, monthly_medians, month, verbose=verbose)
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude,
                                             CTH, COT, CPH, verbose=verbose)
            
            direct_clear = res_dict["direct_clear"]
            diffuse_clear = res_dict["diffuse_clear"]
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            
            if florida_cloud_shadow == 1: 
                sim_vs_obs.at[idx,"florida_direct"] = direct_cloudy
                sim_vs_obs.at[idx,"florida_diffuse"] = diffuse_cloudy
            elif florida_cloud_shadow == 0: 
                sim_vs_obs.at[idx,"florida_direct"] = direct_clear
                sim_vs_obs.at[idx,"florida_diffuse"] = diffuse_clear
            else : 
                print(f"Warning! Model not implemented for cloud mask value {florida_cloud_shadow}. (set at Florida)")

            if flesland_cloud_shadow == 1: 
                sim_vs_obs.at[idx,"flesland_direct"] = direct_cloudy
                sim_vs_obs.at[idx,"flesland_diffuse"] = diffuse_cloudy
            elif flesland_cloud_shadow == 0: 
                sim_vs_obs.at[idx,"flesland_direct"] = direct_clear
                sim_vs_obs.at[idx,"flesland_diffuse"] = diffuse_clear
            else : 
                print(f"Warning! Model not implemented for cloud mask value {flesland_cloud_shadow}. (set at Flesland)")

    sim_vs_obs["florida_ghi_sim_horizontal"] = sim_vs_obs["florida_direct"] + sim_vs_obs["florida_diffuse"]
    sim_vs_obs["flesland_ghi_sim_horizontal"] = sim_vs_obs["flesland_direct"] + sim_vs_obs["flesland_diffuse"]
    
    sub_new_vars = sim_vs_obs[["date", "cloud_cover_large", "total_clear_sky"]]
    print(f"Head of df with new variables: \n{sub_new_vars.head()}")
    print(f"Summary of df with new variables: \n{sub_new_vars.describe()}")
    print(f"Summary of df with new variables: \n{sub_new_vars.describe()}")
        
    return sim_vs_obs



if __name__ == "__main__": 
    # Paths
    cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_stations_data.csv"
    DSM_filepath = "data/processed/bergen_dsm_10m_epsg4326.tif"
    sw_cor_filepath = "data/processed/sw_cor/sw_cor_bergen.nc" # direct sw correction factor for mean reducer dsm
    s2_cloud_mask_folderpath = "data/raw/S2_cloud_mask_large"
    lut_filepath = "data/processed/LUT/claas3/LUT.csv" # LUT with Claas3 cloud properties
    cloud_shadow_filepath = "data/processed/cloud_shadow_thresh40.nc"
    cloud_props = pd.read_csv(cloud_cover_table_filepath) 
    outpath_new_florida_flesland_sim = "data/processed/s2_cloud_cover_with_stations_with_pixel_sim.csv"
    
    for res in COARSE_RESOLUTIONS:
        print(f"Resolution: {res}m.")
        coarse_cloud_shadow_filepath = f"data/processed/cloud_shadow_{res}m.nc"
        total_ghi_from_sat_imgs(coarse_cloud_shadow_filepath, cloud_cover_table_filepath, lut_filepath,
                                sw_cor_path=None, out_nc_file=f"data/processed/simulated_ghi_without_terrain_only_mixed_{res}m.nc",
                                mixed_threshold=1, overcast_threshold=99, verbose=False)

    df = simulate_stations_pixels_corrected(cloud_cover_filepath=cloud_cover_table_filepath,
                                       lut_filepath=lut_filepath, 
                                       cloud_shadow_path=cloud_shadow_filepath,
                                       verbose=False)
    df.to_csv(outpath_new_florida_flesland_sim, index=False) 