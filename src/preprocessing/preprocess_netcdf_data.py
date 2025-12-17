# =============================================================================
# This script contains functions for preprocessing steps for netcdf files,
# like the ones retrieved from CLAAS-3 and CLARA-A 3.0 datasets from CM SAF
# Web Interface. 
# CLAAS-3 cloud properties are retrieved from here: 
# https://wui.cmsaf.eu/safira/action/viewProduktDetails?eid=22218_22239&fid=38 (COT, CPH, CGH)
# and here: 
# https://wui.cmsaf.eu/safira/action/viewProduktDetails?eid=22223_22244&fid=38 (CTH)
# CLARA-A 3.0 surface albedo is retrieved from here: 
# https://wui.cmsaf.eu/safira/action/viewProduktDetails?eid=22453_22560&fid=40 
# =============================================================================


import xarray as xr
import glob, os
import pandas as pd
import numpy as np
from netCDF4 import Dataset, num2date
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from datetime import datetime
from src.model import BBOX, MIXED_THRESHOLD, OVERCAST_THRESHOLD

def classify_sky_type(cc):
        if cc <= MIXED_THRESHOLD:
            return "clear"
        elif cc >= OVERCAST_THRESHOLD:
            return "overcast"
        else:
            return "mixed"

def inspect_file(filepath, variable_name):
    """Open single netcdf file in filepath, print file structure and display variable. """
    # open the netCDF file
    ds = xr.open_dataset(filepath, decode_times=False)

    print(ds)  # print an overview (variables, dimensions, attributes)

    # list variables
    print("Variables:", list(ds.variables))

    # inspect one variable, e.g. cloud mask
    variable = ds[variable_name]  # if present in file
    print(variable)
    print(f" {variable_name} Values:")
    print(variable.values)



def extract_clara_albedo(data_folder, output_csv, target_lon=5.33):
    """
    Loop over all CLARA SAL NetCDF files in a folder and extract 
    black-, white-, blue-sky mean, median, and all-sky median albedo 
    at given longitude (files contain only one latitude value).
    
    Parameters:
        data_folder (str): folder containing all CLARA NetCDF files
        output_csv (str): path to save resulting CSV
        target_lon (float): longitude to extract data from
        
    Returns: 
        df (Dataframe) : dataframe containing columns for date 
                         and albedo values
    """
    # Find all NetCDF files
    file_list = sorted(glob.glob(os.path.join(data_folder, "*.nc")))
    
    records = []

    for fpath in file_list:
        ds = xr.open_dataset(fpath)

        # Find index of closest longitude
        lon_idx = np.argmin(np.abs(ds.lon.values - target_lon))

        # Extract medians
        blue_med  = ds["blue_sky_albedo_median"].isel(lon=lon_idx).values.flatten()[0]
        black_med = ds["black_sky_albedo_median"].isel(lon=lon_idx).values.flatten()[0]
        white_med = ds["white_sky_albedo_median"].isel(lon=lon_idx).values.flatten()[0]
        blue_all_med  = ds["blue_sky_albedo_all_median"].isel(lon=lon_idx).values.flatten()[0]
        black_all_med = ds["black_sky_albedo_all_median"].isel(lon=lon_idx).values.flatten()[0]
        white_all_med = ds["white_sky_albedo_all_median"].isel(lon=lon_idx).values.flatten()[0]

        # Extract means
        blue_mean  = ds["blue_sky_albedo_mean"].isel(lon=lon_idx).values.flatten()[0]
        black_mean = ds["black_sky_albedo_mean"].isel(lon=lon_idx).values.flatten()[0]
        white_mean = ds["white_sky_albedo_mean"].isel(lon=lon_idx).values.flatten()[0]

        # Extract date
        date = pd.to_datetime(ds.time.values[0])

        # Store record
        records.append({
            "date": date,
            "blue_sky_albedo_median": blue_med,
            "black_sky_albedo_median": black_med,
            "white_sky_albedo_median": white_med,
            "blue_sky_albedo_all_median": blue_all_med,
            "black_sky_albedo_all_median": black_all_med,
            "white_sky_albedo_all_median": white_all_med,
            "blue_sky_albedo_mean": blue_mean,
            "black_sky_albedo_mean": black_mean,
            "white_sky_albedo_mean": white_mean
        })

        ds.close()  # Close file to free memory

    # Build DataFrame
    df = pd.DataFrame(records)
    df.sort_values("date", inplace=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved albedo data to {output_csv}")
    return df


def plot_whole_time_series(df, variable_name, title, outpath):
    """Plot the variable from the dataframe df for the whole available date range."""
    # Plot variable over time
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot, automatically skipping NaNs
    ax.plot(df["date"], df[variable_name], marker='o', linestyle='-', label=variable_name)

    # Format x-axis for year-month
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # tick every 6 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Rotate labels for readability
    plt.xticks(rotation=45)

    # Labels & title
    ax.set_ylabel(variable_name)
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi = 150)
    print(f"Figure saved to {outpath}")
    

def merge_albedo_with_s2_table(s2_path="data/processed/s2_cloud_cover_table_small_and_large.csv",
                                        albedo_path="data/processed/SAL_2015-2025.csv"):
    """Add the timewise closest albedo observation to each cloud cover observations of sentinel 2. """
    # Load S2 observations
    s2_df = pd.read_csv(s2_path, parse_dates=["date"])

    # Load CLARA blue-sky albedo
    sal_df = pd.read_csv(albedo_path, parse_dates=["date"])

    # Sort CLARA dataframe by date for merge_asof
    sal_df = sal_df.sort_values("date")

    # Merge S2 with nearest CLARA date
    s2_df = pd.merge_asof(
        s2_df.sort_values("date"),
        sal_df[["date", "blue_sky_albedo_median"]],
        on="date",
        direction="nearest"  # choose the nearest date
    )

    # The resulting s2_df now has a new column 'blue_sky_albedo_median' corresponding to nearest CLARA date
    print(s2_df.head())
    return s2_df

def plot_claas3_file(aux_path, claas3_file, var="cot"):
    """Plot one variable of sample claas3 file with geo information."""
    # Open the auxiliary file (lat/lon/acq_time live here)
    aux = xr.open_dataset(aux_path, decode_times=False)

    # Open the sample data file (cot lives here)
    sample = xr.open_dataset(claas3_file)

    # Extract variables
    lat = aux["lat"].isel(georef_offset_corrected=0)   # (y, x)
    lon = aux["lon"].isel(georef_offset_corrected=0)   # (y, x)
    values = sample[var].isel(time=0)                   # (y, x)
    
    # Bounding box
    lat_min, lat_max = BBOX["south"], BBOX["north"]
    lon_min, lon_max = BBOX["west"], BBOX["east"]

    # Build mask for bounding box
    mask = ((lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max))

    # Apply mask (set values outside bounding box to NaN)
    values_masked = values.where(mask)

    
    time_val = sample["time"].values[0]  # CF-compliant numeric time
    acq_time = pd.to_datetime(str(time_val))  # convert to pandas datetime
    acq_time_str = acq_time.strftime("%Y%m%dT%H%M%S")
    print(f"acq_time_str: {acq_time_str}")

    # Plot
    plt.figure(figsize=(8,6))
    im = plt.pcolormesh(lon, lat, values_masked, cmap="viridis")
    plt.colorbar(im, label=f"{var}")
    plt.xlabel("Longitude (degrees east)")
    plt.ylabel("Latitude (degrees north)")
    plt.title(f"{var}")
    outpath=f"output/claas3_{var}_{acq_time_str}.png"
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}.")
    
    
def get_claas3_filepath(claas_folderpath, dt, file_prefix, possible_times):
    """
    Construct the CLAAS-3 filepath for a given datetime.
    
    Parameters
    ----------
    claas_folderpath : str or Path
        Root folder of CLAAS-3 dataset (contains year/month/day subfolders).
    dt : datetime-like
        Timestamp (must be timezone-aware UTC).
    file_prefix : str
        Filename prefix (e.g. "CPPin" for COT, "CTXin" for CTH).
    possible_times : list of int
        Candidate acquisition times in minutes after midnight (e.g. [645, 660, 675]).
    
    Returns
    -------
    Path or None
        Path to matching CLAAS-3 file, or None if not found.
    """
    # Compute minutes of day
    t_minutes = dt.hour * 60 + dt.minute
    assert min(possible_times) <= t_minutes <= max(possible_times), f"Minutes of day {t_minutes} are not within the range of possible_times!"
    closest_minutes = min(possible_times, key=lambda x: abs(x - t_minutes))
    hhmm = f"{closest_minutes//60:02d}{closest_minutes%60:02d}"

    # Format search pattern
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    search_pattern = f"{file_prefix}{year}{month}{day}{hhmm}*.nc"

    # Search
    path = Path(claas_folderpath, year, month, day).glob(search_pattern)
    candidates = list(path)

    if candidates:
        return candidates[0]
    else:
        return None    
    
def compute_roi_stats(var, mask=None, variable_name="cot", suffix="_large", base_stats=None):
    """
    Compute statistics (min, max, mean, median, std) for a given ROI (small/large) and variable name (cot/cth/cph).

    Parameters
    ----------
    var : xarray.DataArray
        CLAAS-3 variable data (2D).
    mask : xarray.DataArray or ndarray[bool], optional
        Boolean mask for ROI. If None, uses full array.
    variable_name : str
        Name of the variable (e.g. "cot", "cth").
    suffix : str
        Suffix to append (e.g. "_large" or "_small").
    base_stats : list of str
        List of statistics to compute.

    Returns
    -------
    dict
        Dictionary of {f"{variable_name}_{stat}{suffix}": value}
    """
        
    if base_stats is None:
        base_stats = ["min", "max", "mean", "median", "std"]

    # Apply mask if provided
    if mask is not None:
        vals = var.where(mask).values
    else:
        vals = var.values

    # Clean values: remove NaN/inf, require > 0
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0.0]

    stats_out = {}
    if vals.size > 0:
        stats_out.update({
            f"{variable_name}_min{suffix}": np.nanmin(vals),
            f"{variable_name}_max{suffix}": np.nanmax(vals),
            f"{variable_name}_mean{suffix}": np.nanmean(vals),
            f"{variable_name}_median{suffix}": np.nanmedian(vals),
            f"{variable_name}_std{suffix}": np.nanstd(vals),
        })
    else:
        stats_out.update({f"{variable_name}_{s}{suffix}": np.nan for s in base_stats})

    return stats_out
    

def add_claas3_variable_to_cloud_cover_table(
    claas_folderpath, aux_path, cloud_cover_path,
    variable_name="cot", file_prefix="CPPin"
): 
    """
    Iterate over CLAAS-3 files, read the .nc file, 
    compute statistics for both large ROI and small ROI 
    for a given variable (e.g. 'cot' or 'cth'), 
    and save results back to the cloud cover table.

    Parameters
    ----------
    claas_folderpath : str or Path
        Root folder of CLAAS-3 dataset (contains year/month/day subfolders).
    aux_path : str or Path
        Path to auxiliary file containing lat/lon.
    cloud_cover_path : str or Path
        Path to cloud cover CSV table.
    variable_name : str, optional
        CLAAS-3 variable name to extract (e.g. "cot" or "cth").
    file_prefix : str, optional
        Filename prefix (e.g. "CPPin" for COT, "CTXin" for CTH).
    """

    # Read cloud cover table
    cloud_cover = pd.read_csv(cloud_cover_path)
    
    # Open the auxiliary file (lat/lon live here)
    aux = xr.open_dataset(aux_path, decode_times=False)
    lat = aux["lat"].isel(georef_offset_corrected=0)   # (y, x)
    lon = aux["lon"].isel(georef_offset_corrected=0)   # (y, x)
    
    # Bounding box small roi
    lat_min, lat_max = BBOX["south"], BBOX["north"]
    lon_min, lon_max = BBOX["west"], BBOX["east"]
    
    mask_small = ((lat >= lat_min) & (lat <= lat_max) &
                  (lon >= lon_min) & (lon <= lon_max))

    # Define stats columns
    suffixes = ["_large", "_small"]
    base_stats = ["min", "max", "mean", "median", "std"]
    stats_cols = [f"{variable_name}_{s}{suf}" for suf in suffixes for s in base_stats]

    for col in stats_cols:
        if col not in cloud_cover.columns:
            cloud_cover[col] = np.nan

    # Possible acquisition times in minutes
    possible_times = [10*60+45, 11*60+0, 11*60+15]  # 645, 660, 675 minutes

    # Iterate over rows 
    for idx, row in tqdm(cloud_cover.iterrows(), total=len(cloud_cover), desc=f"Processing CLAAS-3 {variable_name} files"):         
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)

        claas3_file = get_claas3_filepath(
            claas_folderpath, dt, file_prefix, possible_times
        )

        if claas3_file is None:
            print(f"[WARN] No file found for {dt}")
            continue
    
        try:
            sample = xr.open_dataset(claas3_file)
            var = sample[variable_name].isel(time=0)  # (y, x)

            stats_all = {}

            # Large roi
            stats_all.update(compute_roi_stats(var, mask=None, variable_name=variable_name, suffix="_large", base_stats=base_stats))
            # Small roi
            stats_all.update(compute_roi_stats(var, mask=mask_small, variable_name=variable_name, suffix="_small", base_stats=base_stats))
            
            # Assign back to DataFrame
            for k, v in stats_all.items():
                cloud_cover.at[idx, k] = v

        except Exception as e:
            print(f"[ERROR] Failed reading {claas3_file}: {e}")
            continue

    # Save new table
    out_path = cloud_cover_path.replace(".csv", f"_with_{variable_name}.csv")
    cloud_cover.to_csv(out_path, index=False)
    print(f"Saved updated table to {out_path}.")
    
    # Print summary
    print("\n=== Statistics summary ===")
    print(cloud_cover[stats_cols].describe())
    nan_counts = cloud_cover[stats_cols].isna().sum()
    print("\n=== Number of NaNs per column ===")
    print(nan_counts)
    
    all_nan_rows = cloud_cover[stats_cols].isna().all(axis=1)
    problematic_rows = cloud_cover.loc[all_nan_rows, ["date", "cloud_cover_small", "cloud_cover_large"]]
    print("\n=== Rows where all stats are NaN ===")
    print(problematic_rows)

def compare_claas3_small_vs_large_roi(cloud_cover_filepath, selected_vars):
    """For a selected pair of variables (small vs large roi), 
    print the pearsons correlation between the small roi and large roi. """
    df = pd.read_csv(cloud_cover_filepath)

    # Select relevant columns
    df_selected = df[selected_vars].copy()

    # Drop rows where both are NaN (no useful info)
    df_selected = df_selected.dropna(how="all")

    # Print correlation (Pearson and Spearman)
    print("Correlation (Pearson):")
    print(df_selected.corr(method="pearson"))
    print("\nCorrelation (Spearman):")
    print(df_selected.corr(method="spearman"))

    # Scatter plot with regression line
    plt.figure(figsize=(6,6))
    sns.regplot(
        data=df_selected,
        x=selected_vars[0],
        y=selected_vars[1],
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"}
    )
    plt.xlabel(selected_vars[0] + " Small Roi")
    plt.ylabel(selected_vars[1])
    plt.title(f"Scatter plot of {selected_vars[0]} (small vs large ROI)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outpath = f"output/{selected_vars[0]}_small_vs_large_scatter.png"
    plt.savefig(outpath)
    print(f"Saved scatterplot to {outpath}.")
    

def convert_ghi_to_clear_sky_index(irradiance_infile_nc, cloud_cover_table_path, clear_sky_index_outfile_nc=None):
    """
    Convert GHI_total to clear-sky index (GHI_total / total_clear_sky) and
    save incrementally to NetCDF (no full array in memory).

    Parameters
    ----------
    irradiance_infile_nc : str
        Path to NetCDF containing variable 'GHI_total'.
    cloud_cover_table_path : str
        CSV with columns ['date', 'total_clear_sky'].
    clear_sky_index_outfile_nc : str or None
        If None: append 'clear_sky_index' variable to input file.
        If str: create a new NetCDF file (copied metadata + new variable).
    """
    
    # --- Load cloud cover table ---
    df = pd.read_csv(cloud_cover_table_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # --- Open input file (read-only first) ---
    src = Dataset(irradiance_infile_nc, "r")
    time_var = src.variables["time"]
    times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
    ghi_var = src.variables["GHI_total"]
    times_dates = np.array([datetime(t.year, t.month, t.day).date() for t in times])

    # --- Prepare output file ---
    if clear_sky_index_outfile_nc is None:
        # Append to same file
        src.close()
        dst = Dataset(irradiance_infile_nc, "a")
        if "clear_sky_index" not in dst.variables:
            v = dst.createVariable("clear_sky_index", "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            v.long_name = "Clear-sky index (GHI_total / GHI_clear)"
            v.units = "1"
        var_out = dst.variables["clear_sky_index"]
    else:
        # Create new file with same structure
        dst = Dataset(clear_sky_index_outfile_nc, "w")
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        # Copy coordinate variables
        for name in ["time", "lat", "lon"]:
            in_var = src.variables[name]
            out_var = dst.createVariable(name, in_var.datatype, in_var.dimensions)
            out_var.setncatts({k: in_var.getncattr(k) for k in in_var.ncattrs()})
            out_var[:] = in_var[:]
        # Create clear_sky_index variable
        var_out = dst.createVariable("clear_sky_index", "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
        var_out.long_name = "Clear-sky index (GHI_total / GHI_clear)"
        var_out.units = "1"

    # --- Iterate over rows in CSV ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing clear-sky index incrementally"):
        obs_date = row["date"].date()
        cloud_cover = row["cloud_cover_large"]
        total_clear_sky = row["total_clear_sky"]

        # Find matching time index
        time_idx = np.where(times_dates == obs_date)[0]
        if len(time_idx) == 0:
            assert cloud_cover <= MIXED_THRESHOLD or cloud_cover >= OVERCAST_THRESHOLD, f"No matching found for mixed sky type (cloud cover = {cloud_cover}) on {obs_date}!"    
            tqdm.write(f"No matching map found for {obs_date}, skipping...")
            continue

        idx = int(time_idx[0])
        ghi_map = ghi_var[idx, :, :]

        # Compute clear-sky index safely
        clear_sky_index = np.divide(
            ghi_map, total_clear_sky,
            out=np.full_like(ghi_map, np.nan),
            where=np.isfinite(ghi_map)
        )

        # Optional sanity log
        min_val, max_val = np.nanmin(clear_sky_index), np.nanmax(clear_sky_index)
        assert min_val >= 0, f"Negative clear-sky index detected ({min_val})"
        tqdm.write(f"{obs_date}: range {min_val:.3f}–{max_val:.3f}")

        # Write slice directly to disk
        var_out[idx, :, :] = clear_sky_index

    # --- Cleanup ---
    dst.sync()
    dst.close()
    if clear_sky_index_outfile_nc is not None: 
        src.close()
    print("✅ Clear-sky index written successfully.")
    
       
def clear_sky_index_per_sky_type(single_ghi_maps_filepath, cloud_cover_table_path, aggregated_ghi_maps_outpath):
    """
    Preprocess and aggregate GHI maps per sky type (mean clear-sky index).
    Skips computation if the aggregated NetCDF already exists.
    """
    if os.path.exists(aggregated_ghi_maps_outpath):
        print(f"Aggregated file already exists at {aggregated_ghi_maps_outpath}. Skipping aggregation.")
        return

    # Load cloud cover metadata
    df = pd.read_csv(cloud_cover_table_path)
    df["date"] = pd.to_datetime(df["date"])
    if "sky_type" not in df.columns:
        df["sky_type"] = np.where(
            df["cloud_cover_large"] <= MIXED_THRESHOLD,
            "clear",
            np.where(df["cloud_cover_large"] >= OVERCAST_THRESHOLD, "overcast", "mixed")
        )

    # Open single observation NetCDF
    src = Dataset(single_ghi_maps_filepath)
    time_var = src.variables["time"]
    times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
    ghi_data = src.variables["GHI_total"]
    lat = src.variables["lat"][:]
    lon = src.variables["lon"][:]
    shape = (len(lat), len(lon))

    # Running sums and counts
    accum_sum = {sky: np.zeros(shape, dtype=np.float64) for sky in ["clear", "mixed", "overcast"]}
    accum_count = {sky: np.zeros(shape, dtype=np.int32) for sky in ["clear", "mixed", "overcast"]}

    # Precompute times as plain dates
    times_dates = np.array([datetime(t.year, t.month, t.day).date() if not hasattr(t, "date") else t.date() for t in times])

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating to mean clear sky index..."):
        obs_date = row["date"].date()
        sky_type = row["sky_type"]
        total_clear_sky = row["total_clear_sky"]
        florida_ghi_sim_horizontal = row["florida_ghi_sim_horizontal"]

        time_idx = np.where(times_dates == obs_date)[0]
        if len(time_idx) == 0:
            tqdm.write(f"No matching map found for {obs_date}, skipping...")
            continue

        # Debugging info 
        tqdm.write(f"\nDate: {obs_date}, Sky type: {sky_type}, Total clear sky: {total_clear_sky}, Florida ghi: {florida_ghi_sim_horizontal}")


        ghi_map = ghi_data[time_idx[0], :, :]
        clear_sky_index = ghi_map / total_clear_sky
        
        # Sanity check
        min_val, max_val = np.nanmin(clear_sky_index), np.nanmax(clear_sky_index)
        assert min_val >= 0, f"Negative clear-sky index detected ({min_val})"
        tqdm.write(f"Clear-sky index range: {min_val:.3f}–{max_val:.3f}")

        valid_mask = np.isfinite(clear_sky_index)
        accum_sum[sky_type][valid_mask] += clear_sky_index[valid_mask]
        accum_count[sky_type][valid_mask] += 1

    src.close()

    # Compute mean maps
    mean_maps = {}
    for sky in ["clear", "mixed", "overcast"]:
        valid = accum_count[sky] > 0
        mean_map = np.full(shape, np.nan, dtype=np.float32)
        mean_map[valid] = accum_sum[sky][valid] / accum_count[sky][valid]
        mean_maps[sky] = mean_map

    # Save aggregated NetCDF
    os.makedirs(os.path.dirname(aggregated_ghi_maps_outpath), exist_ok=True)
    with xr.Dataset(
        {f"{sky}_mean_clear_sky_index": (["lat", "lon"], data) for sky, data in mean_maps.items()},
        coords={"lat": lat, "lon": lon}
    ) as ds_agg:
        ds_agg.to_netcdf(aggregated_ghi_maps_outpath)
        print(f"Saved aggregated mean maps to {aggregated_ghi_maps_outpath}")


def aggregate_variable_monthly(
    in_nc_file,
    out_nc_file,
    variable_name,
    out_var_name=None,
    out_var_long_name=None,
    out_var_units=None,
):
    """
    Stream through a variable in a NetCDF file, accumulate monthly sums,
    and compute mean monthly maps without loading everything into memory.

    Parameters
    ----------
    in_nc_file : str
        Path to input NetCDF file containing the variable to aggregate.
    out_nc_file : str
        Path to output NetCDF file for monthly aggregated results.
    variable_name : str
        Name of the variable to aggregate (e.g. "shadow_mask", "GHI_total").
    out_var_name : str, optional
        Name for the aggregated variable in the output file (default = variable_name + "_monthly_mean").
    out_var_long_name : str, optional
        Long descriptive name for the output variable.
    out_var_units : str, optional
        Units of the output variable (copied from input if not provided).
    """

    # -------------------------------------------------------------------------
    # Open input NetCDF
    # -------------------------------------------------------------------------
    with Dataset(in_nc_file, "r") as src:
        # Time information
        time_var = src.variables["time"]
        times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        months = np.array([t.month for t in times])

        # Coordinates
        lats = src.variables["lat"][:]
        lons = src.variables["lon"][:]
        nlat, nlon = len(lats), len(lons)

        # Input variable
        if variable_name not in src.variables:
            raise KeyError(f"Variable '{variable_name}' not found in {in_nc_file}.")
        var = src.variables[variable_name]

        # Get variable units if not provided
        if out_var_units is None and hasattr(var, "units"):
            out_var_units = var.units

        # Preallocate accumulators
        sums = np.zeros((12, nlat, nlon), dtype=np.float64)
        counts = np.zeros(12, dtype=np.int32)

        # Loop over all time steps
        for i, m in enumerate(tqdm(months, desc=f"Aggregating {variable_name}", unit="images")):
            data = var[i, :, :].astype(np.float64)
            if np.all(np.isnan(data)):
                continue
            sums[m - 1] += np.nan_to_num(data)
            counts[m - 1] += np.isfinite(data).any()  # count image if any valid data

    # -------------------------------------------------------------------------
    # Compute monthly mean maps
    # -------------------------------------------------------------------------
    monthly_means = np.full_like(sums, np.nan, dtype=np.float32)
    for m in range(12):
        if counts[m] > 0:
            monthly_means[m] = sums[m] / counts[m]

    # -------------------------------------------------------------------------
    # Save output NetCDF
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_nc_file), exist_ok=True)
    with Dataset(out_nc_file, "w") as dst:
        # Dimensions
        dst.createDimension("month", 12)
        dst.createDimension("lat", nlat)
        dst.createDimension("lon", nlon)

        # Month variable
        nc_month = dst.createVariable("month", "i4", ("month",))
        nc_month[:] = np.arange(1, 13)
        nc_month.units = "month number"

        # Coordinates
        nc_lat = dst.createVariable("lat", "f4", ("lat",))
        nc_lat[:] = lats
        nc_lat.units = "degree_north"
        nc_lon = dst.createVariable("lon", "f4", ("lon",))
        nc_lon[:] = lons
        nc_lon.units = "degree_east"

        # Monthly mean
        out_name = out_var_name or f"{variable_name}_monthly_mean"
        nc_mean = dst.createVariable(out_name, "f4", ("month", "lat", "lon"), zlib=True)
        nc_mean[:, :, :] = monthly_means
        nc_mean.units = out_var_units or "unknown"
        nc_mean.long_name = out_var_long_name or f"Mean monthly {variable_name}"

        # Monthly counts (number of valid images)
        nc_count = dst.createVariable(f"{variable_name}_monthly_count", "i4", ("month",))
        nc_count[:] = counts
        nc_count.units = "count"
        nc_count.long_name = f"Number of valid {variable_name} images per month"

    print(f"✅ Monthly aggregation for '{variable_name}' saved to {out_nc_file}")

    
def summarize_monthly_cloud_stats(df, cloud_col="cloud_cover", sky_type_col="sky_type"):
    """
    Calculate monthly cloud cover mean/std and monthly sky type probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least ['month', cloud_col, sky_type_col].
    cloud_col : str, optional
        Column name containing cloud cover percentage values.
    sky_type_col : str, optional
        Column name containing sky type category ('clear', 'mixed', 'overcast').

    Returns
    -------
    monthly_summary : pandas.DataFrame
        DataFrame containing per-month cloud cover statistics and sky type probabilities.
    """

    # Monthly mean and std for cloud cover
    monthly_cloud_stats = (
        df.groupby("month")[cloud_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "cloud_cover_mean", "std": "cloud_cover_std"})
    )

    # Count how many days per sky type per month
    sky_type_counts = (
        df.groupby(["month", sky_type_col])
        .size()
        .unstack(fill_value=0)
    )

    # Combine both summaries
    monthly_summary = monthly_cloud_stats.join(sky_type_counts)
    monthly_summary.reset_index(inplace=True)

    # Print summary
    print("\n📊 Monthly summary:")
    print(monthly_summary)
    
    months = df["month"].unique()
    fig, axs = plt.subplots(4,3, figsize=(8, 12), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i, month in enumerate(months): 
        df_month = df[df["month"] == month]
        df_month = df_month[["cloud_cover_large"]]
        axs[i].hist(df_month, bins=50, density=1)
        axs[i].set_xlabel("cloud cover")
        axs[i].set_ylabel("frequency")
    outpath="output/monthly_cloud_cover_hist.png"
    plt.savefig(outpath)
    print(f"Histogram of cloud cover saved to {outpath}.")

    return monthly_summary


def get_area_mean_clear_sky_index(clear_sky_index_nc_filepath):
    """For the clear-sky index variable in the netcdf file, compute the mean per timestep and 
    return dataframe with columns date and mean_clear_sky_index."""
    dates = []
    means = []

    with Dataset(clear_sky_index_nc_filepath, "r") as src:
        # --- Extract time variable and convert to datetimes ---
        time_var = src.variables["time"]
        time_units = time_var.units
        time_calendar = getattr(time_var, "calendar", "standard")

        times = num2date(time_var[:], units=time_units, calendar=time_calendar)

        # --- Extract variable reference ---
        csi_var = src.variables["clear_sky_index"]

        # --- Loop over time dimension ---
        for i, t in tqdm(enumerate(times), total=len(times), desc="Calculating area-wide mean clear sky index..."):
            # Read slice of shape (lat, lon) for timestep i
            csi_slice = csi_var[i, :, :]

            # Compute mean, ignoring NaNs
            mean_val = np.nanmean(csi_slice)

            dates.append(t)
            means.append(mean_val)

    # --- Construct DataFrame ---
    df = pd.DataFrame({
        "date": dates,
        "mean_clear_sky_index": means
    })

    return df

def merge_area_mean_clear_sky_index_with_sky_type(area_mean_clear_sky_index_path, sim_vs_obs_path): 
    # --- Load and normalize dates ---
    clear_sky_df = pd.read_csv(area_mean_clear_sky_index_path)
    clear_sky_df["date"] = pd.to_datetime(clear_sky_df["date"]).dt.normalize()

    df_sim = pd.read_csv(sim_vs_obs_path)
    df_sim["sky_type"] = df_sim["cloud_cover_large"].apply(classify_sky_type)
    df_sim["date"] = pd.to_datetime(df_sim["date"]).dt.normalize()

    # --- Add month + fix sky_type for existing rows ---
    clear_sky_df["month"] = clear_sky_df["date"].dt.month
    clear_sky_df["sky_type"] = "mixed"   # set all to mixed as required

    # --- Identify missing dates ---
    dates_in_clear = set(clear_sky_df["date"])
    df_sim_missing = df_sim[~df_sim["date"].isin(dates_in_clear)].copy()

    print(f"Missing dates to append: {len(df_sim_missing)}")

    # --- Compute clear_sky_index for missing days ---
    # Required columns must exist in df_sim
    # florida_ghi_sim_horizontal : simulated irradiance
    # total_clear_sky            : clear-sky irradiance
    df_sim_missing["mean_clear_sky_index"] = (
        df_sim_missing["florida_ghi_sim_horizontal"]
        / df_sim_missing["total_clear_sky"]
    )

    # --- Add month column if not present ---
    if "month" not in df_sim_missing:
        df_sim_missing["month"] = df_sim_missing["date"].dt.month

    # --- Extract needed columns ---
    df_sim_missing = df_sim_missing[["date", "month", "sky_type", "mean_clear_sky_index"]]

    # --- Append to clear_sky_df ---
    clear_sky_df_updated = pd.concat([clear_sky_df, df_sim_missing], ignore_index=True)

    # --- Optional: sort by date ---
    clear_sky_df_updated = clear_sky_df_updated.sort_values("date").reset_index(drop=True)

    print(clear_sky_df_updated.head())
    print(clear_sky_df_updated.tail())

    # --- Save ---
    clear_sky_df_updated.to_csv(area_mean_clear_sky_index_outpath, index=False)
    

if __name__ == "__main__":
    sample_file = "data/raw/claas3_201506-2025-08/claas3/cpp/2020/05/04/CPPin20200504110000405SVMSG01MD.nc"
    aux_file = "data/raw/claas3_201506-2025-08/claas3/CM_SAF_CLAAS3_L2_AUX.nc"
    data_folder = "data/raw/CLARA_SAL_2015-2025"
    clara_csv = "data/raw/CWP_2015-2025.csv"
    claas_folder_cpp = "data/raw/claas3_201506-2025-08/claas3/cpp"
    claas_folder_ctx = "data/raw/claas3_201506-2025-08/claas3/ctx"
    s2_csv = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    sim_vs_obs_path = "data/processed/s2_cloud_cover_with_stations_with_pixel_sim.csv"
    single_shadow_maps_nc = "data/processed/cloud_shadow_thresh40.nc"
    monthly_shadow_maps_nc = "data/processed/cloud_shadow_thresh40_monthly.nc"
    single_ghi_maps = "data/processed/simulated_ghi.nc"
    monthly_ghi_maps = "data/processed/simulated_irradiance_monthly.nc"
    monthly_clear_sky_index_maps = "data/processed/simulated_clear_sky_index_monthly_mixed_sky.nc"
    aggregated_sky_type_clear_sky_index_outpath = "data/processed/clear_sky_index_sky_type_all_time_11UTC.nc"
    mixed_sky_ghi ="data/processed/simulated_ghi_without_terrain_only_mixed.nc"
    area_mean_clear_sky_index_outpath = "data/processed/area_mean_clear_sky_index_per_obs.csv"
    monthly_longterm_sim_results = "data/processed/longterm_ghi_spatially_resolved_monthly.nc"

    
    # --------------------- Data exploration --------------------------------
    inspect_file("data/raw/comet2-CFC-2018/CFChm201807121900002UD1000101UD.nc", "CFC")
    plot_claas3_file(aux_file, sample_file, "cth")
       
    # ----------------------- Extract albedo and cloud properties ----------------------
    add_claas3_variable_to_cloud_cover_table(claas_folder_cpp, aux_file, s2_csv, 
                                             variable_name="cgt", file_prefix="CPPin")
    
    df_albedo = extract_clara_albedo(data_folder, clara_csv, target_lon=5.33)
    df_albedo = pd.read_csv(clara_csv, parse_dates=["date"])
    print(df_albedo[["black_sky_albedo_median", "black_sky_albedo_all_median"]].describe())
    outpath = "output/monthly_mean_albedo_comparison_incl_snow.png"
    
    # ------------------- Monthly sky type probabilities ---------------------
    df = pd.read_csv("data/processed/claas3_cloud_cover_sky_type_from_cot.csv")
    claas_monthly_summary = summarize_monthly_cloud_stats(df)
    
    # ----------------- Clear Sky index ------------------
    convert_ghi_to_clear_sky_index(irradiance_infile_nc=mixed_sky_ghi, cloud_cover_table_path=sim_vs_obs_path)
    
    aggregate_variable_monthly(
        in_nc_file=mixed_sky_ghi,
        out_nc_file=monthly_clear_sky_index_maps,
        variable_name="clear_sky_index",
        out_var_name="clear_sky_index",
        out_var_long_name="Mean clear sky index aggregated monthly (UTC=11)",
        out_var_units="0-1",
    ) 
    
    clear_sky_index_per_sky_type(single_ghi_maps_filepath=single_ghi_maps, cloud_cover_table_path=sim_vs_obs_path, 
                                 aggregated_ghi_maps_outpath=aggregated_sky_type_clear_sky_index_outpath)
    
    
    # -------------------- Monthly sky type probabilities ---------------------
    df_sim = pd.read_csv(sim_vs_obs_path)

    # Derive sky_type from cloud_cover_large (assuming in %)
    df_sim["sky_type"] = df_sim["cloud_cover_large"].apply(classify_sky_type)

    # Add month (from date column if available)
    if "date" in df_sim.columns:
        df_sim["month"] = pd.to_datetime(df_sim["date"]).dt.month
    elif {"year", "month", "day"}.issubset(df_sim.columns):
        df_sim["month"] = pd.to_datetime(df_sim[["year", "month", "day"]]).dt.month
    else:
        raise ValueError("No date information found in dataframe to derive 'month'.")

    # Select only needed columns
    df_sim_subset = df_sim[["month", "cloud_cover_large", "sky_type"]].copy()

    # Run summary
    sim_monthly_summary = summarize_monthly_cloud_stats(
        df_sim_subset, 
        cloud_col="cloud_cover_large",
        sky_type_col="sky_type"
    )
    
    monthly_sky_types = sim_monthly_summary[["month", "clear", "mixed", "overcast"]]
    monthly_sky_types.to_csv("data/processed/monthly_sky_type_counts.csv", index=False)
    
    # --------------------- Extract clear-sky area mean (for annual sim) -------------------
    clear_sky_df = get_area_mean_clear_sky_index(mixed_sky_ghi)
    clear_sky_df.to_csv(area_mean_clear_sky_index_outpath, index=False)
    merge_area_mean_clear_sky_index_with_sky_type(area_mean_clear_sky_index_outpath, sim_vs_obs_path)

    
    
    
    
