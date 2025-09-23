import xarray as xr
import glob, time, os
import pandas as pd
import numpy as np
import geopandas as gpd
import math
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from surface_GHI_model import BBOX

def inspect_file(filepath, variable_name):
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

       
def cfc_diurnal_cycle_monthly(filepath):
    """Create a table that stores CFC (cloud cover fraction) values for function f(x,y,h,m) - for each 
    hour h, month m and for each pixel x,y."""
    # Load the sample file
    ds = xr.open_dataset(filepath)

    # Get variable Cloud fraction with dims (time, lat, lon)
    cf = ds["CFC"]
    
    # Add "hour" and "month" coordinates
    cf = cf.assign_coords(
        hour=("time", cf["time"].dt.hour.data),
        month=("time", cf["time"].dt.month.data)
    )

    # Compute mean cloud fraction per (lat, lon, hour, month)
    f_table = cf.groupby(["month", "hour"]).mean("time")

    return f_table


def visualize_peak_hour(filepath):
    # Load dataset
    ds = xr.open_dataset(filepath)
    cf = ds["CFC"]

    # Add "hour" and "month" coordinates explicitly 
    cf = cf.assign_coords(
        hour=("time", cf["time"].dt.hour.data),
        month=("time", cf["time"].dt.month.data)
    )

    # For each month, find the hour of maximum cloud cover
    peak_hour = cf.groupby("month").map(
        lambda x: x.groupby("hour").mean("time").idxmax("hour")
    )

    # Example: July
    month = 7
    peak_map = peak_hour.sel(month=month)

    # Prepare figure + map axis
    fig, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot raster (without auto-colorbar)
    im = peak_map.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="twilight",
        vmin=0, vmax=23,
        add_colorbar=False
    )

    # Align colorbar with map height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Hour of max cloud cover")

    # Add coastline
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, color = "white")

    # Add landmarks
    landmarks = {
        "Bergen Center": (60.39299, 5.32415),
        "Bergen Airport": (60.2934, 5.2181)
    }
    for name, (lat, lon) in landmarks.items():
        ax.plot(lon, lat, "ro", markersize=3, transform=ccrs.PlateCarree())
        ax.text(lon + 0.02, lat + 0.02, name,
                transform=ccrs.PlateCarree(), fontsize=7, color="red")

    ax.set_title(f"Hour of Maximum Cloud Cover - Month {month}")
            
    ax.set_xticks(np.arange(float(peak_map.lon.min()),
                        float(peak_map.lon.max()), 0.5),
              crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(float(peak_map.lat.min()),
                            float(peak_map.lat.max()), 0.5),
                crs=ccrs.PlateCarree())

    # Format tick labels
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # Optional: add gridlines for clarity
    gl = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)

    
    # Save and close
    plt.savefig(f"output/hour_max_cloud_cover_month_{month}.png", bbox_inches="tight")
    plt.close()

def extract_clara_albedo(data_folder, output_csv, target_lon=5.33):
    """
    Loop over all CLARA SAL NetCDF files in a folder and extract 
    black-, white-, blue-sky mean, median, and all-sky median albedo 
    at given longitude.
    
    Parameters:
        data_folder (str): folder containing all CLARA NetCDF files
        output_csv (str): path to save resulting CSV
        target_lon (float): longitude to extract data from
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
    """Plot the variable from the dataframe for the whole date range, set x-axis ticks for years and months only"""
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
    

def plot_monthly_mean_albedos(clara_csv, s5p_csv, outpath):
    """
    Aggregate CLARA albedo medians (blue, black, white) by month across all years.
    Overlay monthly mean S5P surface albedo and number of observations per month.
    """
    # --- Step 1: Load CLARA data ---
    df_clara = pd.read_csv(clara_csv, parse_dates=["date"])
    df_clara["month"] = df_clara["date"].dt.month

    # --- Step 2: Compute monthly means for CLARA ---
    clara_means = df_clara.groupby("month")[
        ["blue_sky_albedo_all_median", "black_sky_albedo_all_median", "white_sky_albedo_all_median"]
    ].mean()
    
    clara_counts = df_clara.groupby("month")["blue_sky_albedo_all_median"].count()

    # --- Step 3: Load S5P data ---
    df_s5p = pd.read_csv(s5p_csv, parse_dates=["month_start"])
    df_s5p["month"] = df_s5p["month_start"].dt.month

    # Compute monthly mean S5P surface albedo
    s5p_monthly_means = df_s5p.groupby("month")["mean_surface_albedo"].mean()

    # --- Step 4: Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # CLARA albedo curves
    ax1.plot(clara_means.index, clara_means["blue_sky_albedo_all_median"], 
             marker="o", linestyle="-", color="blue", label="Blue-sky median")
    ax1.plot(clara_means.index, clara_means["black_sky_albedo_all_median"], 
             marker="s", linestyle="--", color="black", label="Black-sky median")
    ax1.plot(clara_means.index, clara_means["white_sky_albedo_all_median"], 
             marker="^", linestyle=":", color="gray", label="White-sky median")

    # S5P line
    ax1.plot(s5p_monthly_means.index, s5p_monthly_means.values,
             marker="d", linestyle="-.", color="green", label="S5P mean surface albedo")

    # Axis labels
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean surface albedo (fraction)")
    ax1.set_title("Monthly mean surface albedo (CLARA 2015–2025 & S5P 2017–2025) (incl. snow/ice)")

    # Month ticks
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul",
                         "Aug","Sep","Oct","Nov","Dec"])
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Secondary y-axis for number of CLARA observations
    ax2 = ax1.twinx()
    ax2.bar(clara_counts.index, clara_counts.values, 
            alpha=0.3, color="orange", width=0.6, label="# obs CLARA")
    ax2.set_ylabel("Number of observations")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Figure saved to {outpath}.")
    
    return clara_means, clara_counts, s5p_monthly_means

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
    lat_min, lat_max = 59.3802344451226, 60.50219617276416
    lon_min, lon_max = 4.739101855653983, 5.920898144346017

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
    Compute statistics (min, max, mean, median, std) for a given ROI.

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
    

if __name__ == "__main__":
    sample_file = "data/raw/claas3_201506-2025-08/claas3/cpp/2020/05/04/CPPin20200504110000405SVMSG01MD.nc"
    aux_file = "data/raw/claas3_201506-2025-08/claas3/CM_SAF_CLAAS3_L2_AUX.nc"
    data_folder = "data/raw/CLARA_SAL_2015-2025"
    clara_csv = "data/raw/CWP_2015-2025.csv"
    claas_folder_cpp = "data/raw/claas3_201506-2025-08/claas3/cpp"
    claas_folder_ctx = "data/raw/claas3_201506-2025-08/claas3/ctx"
    s2_csv = "data/processed/s2_cloud_cover_table_small_and_large_with_cot_with_cth.csv"
       
    add_claas3_variable_to_cloud_cover_table(claas_folder_cpp, aux_file, s2_csv, 
                                             variable_name="cph", file_prefix="CPPin")
    
    #df_albedo = extract_clara_albedo(data_folder, clara_csv, target_lon=5.33)
    #df_albedo = pd.read_csv(clara_csv, parse_dates=["date"])
    #print(df_albedo[["black_sky_albedo_median", "black_sky_albedo_all_median"]].describe())
    s5p_csv = "data/processed/s5p_monthly_mean_surface_albedo.csv"
    outpath = "output/monthly_mean_albedo_comparison_incl_snow.png"
    
    #inspect_file(sample_file, "cph")
    #plot_claas3_file(aux_file, sample_file, "cth")
    #plot_monthly_mean_albedos(clara_csv=clara_csv, s5p_csv=s5p_csv, outpath=outpath)
    #crop_to_roi("data/raw/claas-3_test/*.nc", "data/raw/claas-3_test_small_roi.nc", aux_filepath = aux_file)
    #cfc_diurnal_cycle_monthly("data/comet2_roi_month.nc")
    #visualize_peak_hour("data/comet2_roi_month.nc")
    #print(roi)
