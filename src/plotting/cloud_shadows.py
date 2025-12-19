# ============================================================================
# Plotting functions using the spatial cloud shadow maps created for all 
# Sentinel-2 overpasses 
# ============================================================================

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import imageio
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta, timezone
from src.model import BBOX, COARSE_RESOLUTIONS
from src.plotting.high_res_maps import plot_single_band, plot_band_with_extent, plot_monthly_results
from src.model.cloud_shadow import get_cloud_shadow_displacement, get_solar_angle, read_shadow_roi

def get_shadow_from_ind(shadow_nc_file, ind): 
    """Read shadow nc file, retrieve shadow map from index and return shadow map and timestamp"""
    ds = xr.open_dataset(shadow_nc_file)
    shadow_mask = ds["shadow_mask"].isel(time=ind).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    timestamp = ds["time"].isel(time=ind).values
    ds.close()

    # Convert timestamp
    timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[s]").astype(int), tz=timezone.utc)
    
    return shadow_mask, timestamp, lat, lon


def analyze_cloud_shadow_displacement(cloud_cover_table_path, cth): 
    """Get cloud shadow displacement in x and y direction for each observation in cloud_cover table.
    Plot the distribution of displacements."""
    cloud_cover = pd.read_csv(cloud_cover_table_path)
    
    displacement_x = []
    displacement_y = []
    
    for idx, row in cloud_cover.iterrows():
        sat_zenith = row["MEAN_ZENITH"]
        sat_azimuth = row["MEAN_AZIMUTH"]
        cth_small = row["cth_median_small"]
        cth_large = row["cth_median_large"]
        print(f"cth: {cth}, cth_small: {cth_small}, cth_large: {cth_large}")
        if not pd.isna(cth_small): 
            cth = cth_small
        elif not pd.isna(cth_large): 
            cth = cth_large
        
        if pd.isna(sat_zenith) or pd.isna(sat_azimuth):
            # They are always both na if one of them is na 
            sat_zenith = 0.0
            sat_azimuth = 0.0
        
        # Get date 
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)
        # ±3 hours
        dt_minus_3h = dt - pd.Timedelta(hours=4)
        dt_plus_3h  = dt + pd.Timedelta(hours=4)

        #print(dt_minus_3h, dt_plus_3h)
        dt = dt_plus_3h
        solar_zenith, solar_azimuth = get_solar_angle(dt)
        
        if solar_zenith < 80:
            dx, dy = get_cloud_shadow_displacement(solar_zenith, solar_azimuth, 0, 
                                        sat_zenith, sat_azimuth, cloud_top_height=cth)
            
            
            #print(f"Displacement for sol_zen {solar_zenith:.1f}, sol_azi {solar_azimuth:.1f}, " \
            #    f"sat_zen {np.round(sat_zenith,1)}, sat_azi {np.round(sat_azimuth,1)} (Time UTC: {dt}) : " \
            #        f"\ndx = {np.round(dx)}, dy = {np.round(dy)}")
            
            displacement_x.append(dx)
            displacement_y.append(dy)
    
    # Plot hist of displacement x and y 
    # Convert to arrays
    displacement_x = np.array(displacement_x)
    displacement_y = np.array(displacement_y)
    
    # Remove NaNs
    displacement_x = displacement_x[~np.isnan(displacement_x)]
    displacement_y = displacement_y[~np.isnan(displacement_y)]
    
    
    # Count how many are between -20000 and +20000 (inclusive)
    count = np.sum((displacement_x >= -20000) & (displacement_x <= 20000))

    print(f"Number of x displacements between -20000 and +20000: {count}/{len(displacement_x)}")
    print(f"Percentage: {count/len(displacement_x)}")
    
    # Compute percentiles safely
    x_percentiles = np.nanpercentile(displacement_x, [25, 50, 75]) if displacement_x.size > 0 else [np.nan]*3
    y_percentiles = np.nanpercentile(displacement_y, [25, 50, 75]) if displacement_y.size > 0 else [np.nan]*3
    
    # Plot histograms
    plt.figure(figsize=(10,6))
    bins = 30  
    
    plt.hist(displacement_x, bins=bins, alpha=0.5, color="tab:blue", label="dx")
    plt.hist(displacement_y, bins=bins, alpha=0.5, color="tab:orange", label="dy")
    
    # Plot vertical lines for percentiles
    for p, val in zip(["25%", "50%", "75%"], x_percentiles):
        plt.axvline(val, color="tab:blue", linestyle="--", alpha=0.7)
    for p, val in zip(["25%", "50%", "75%"], y_percentiles):
        plt.axvline(val, color="tab:orange", linestyle="--", alpha=0.7)
    
    # Legend text with percentiles
    legend_text = [
        f"dx: p25={x_percentiles[0]:.1f}, p50={x_percentiles[1]:.1f}, p75={x_percentiles[2]:.1f}",
        f"dy: p25={y_percentiles[0]:.1f}, p50={y_percentiles[1]:.1f}, p75={y_percentiles[2]:.1f}"
    ]
    
    plt.legend(title="\n".join(legend_text))
    plt.xlabel("Displacement [m]")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Cloud Shadow Displacement (dx, dy)")
    plt.grid(alpha=0.3)
    outpath = f"output/cloud_shadow_displacement_hist_cutoff_SZA_80_afternoon.png"
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}.")
    

def plot_cloud_shadow_for_timestep(shadow_nc_file, ind=0):
    """
    Plot GHI_total for a specific timestep from the NetCDF file.
    """
    shadow_mask, timestamp, _, _ = get_shadow_from_ind(shadow_nc_file, ind)
    ts_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")

    # Define colormap and levels
    colors = ["white", "darkgray"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=len(colors))

    filestem = Path(shadow_nc_file).stem

    plot_single_band(shadow_mask, f"output/{filestem}_{ts_str}.png", 
            f"Cloud shadow for {ts_str}",
            "Cloud shadow", cmap=cmap, norm=norm)
    

def plot_shadow_frequency_all_obs(monthly_shadow_frequency_filepath):
    """
    Compute and plot the overall mean cloud shadow frequency over the whole
    study area. 
    """
    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(monthly_shadow_frequency_filepath)
    
    irr = ds["shadow_frequency"].astype(np.float64)
    count = ds["shadow_mask_monthly_count"].astype(np.float64)

    # -------------------------------------------------------------------------
    # Compute weighted mean across all months
    # -------------------------------------------------------------------------
    total_weight = count.sum(dim="month")
    weighted_sum = (irr * count).sum(dim="month")
    
    mean_alltime = weighted_sum / total_weight
    mean_alltime = mean_alltime.where(np.isfinite(mean_alltime))  # mask NaNs properly

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    value_ranges=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    value_colors = ["yellow", "limegreen", "green", "teal", "dodgerblue", "mediumblue", 
                    "darkblue"]

    cmap = mcolors.ListedColormap(value_colors)
    norm = mcolors.BoundaryNorm(boundaries=value_ranges, ncolors=len(value_colors))
    
    plot_single_band(
        mean_alltime,
        outpath="output/shadow_frequency_total_all_obs_11UTC.png",
        title="Mean Cloud Shadow Frequency (2015–2025, 11:00 UTC)",
        colorbar_label="Frequency of Cloud Shadow (0-1)",
        cmap=cmap, norm=norm
    )

    return mean_alltime


if __name__ == "__main__": 
    s2_cloud_mask_sample = f"data/raw/S2_cloud_mask_large/S2_cloud_mask_large_2017-08-27_10-56-51-2017-08-27_10-56-51.tif"
    cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    cloud_shadow_nc_res10 = "data/processed/cloud_shadow_thresh40.nc"
    monthly_cloud_shadow_nc = "data/processed/cloud_shadow_thresh40_monthly.nc"
    
    analyze_cloud_shadow_displacement(cloud_cover_table_filepath, 2000)
    plot_monthly_results(monthly_cloud_shadow_nc, var_name="shadow_frequency", 
                         outpath="output/monthly_cloud_shadow_freq_thresh_40.png",
                         title="Monthly Cloud Shadow Frequencies (2015-2025)", 
                         colorbar_ylabel="Frequency of Cloud Shadow", 
                         histogram_title="Distribution of Pixel Values for Monthly Cloud Shadow") 
    
    plot_shadow_frequency_all_obs(monthly_shadow_frequency_filepath=monthly_cloud_shadow_nc)
    
    # Plot 10 m resolution sample image
    ind = 148
    shadow_nc_file = f"data/processed/cloud_shadow_thresh40.nc"
    shadow_mask, timestamp, lat, lon = get_shadow_from_ind(shadow_nc_file, ind=ind)
    filestem = Path(shadow_nc_file).stem
    ts_str = timestamp.strftime("%Y-%m-%d")
    plot_band_with_extent(
            shadow_mask,
            BBOX["west"], BBOX["east"], BBOX["south"], BBOX["north"],
            f"output/cloud_shadow_10m_{ts_str}")
    
    # Plot same image with coarser resolutions
    for res in COARSE_RESOLUTIONS: 
        shadow_nc_file = f"data/processed/cloud_shadow_{res}m.nc"
        shadow_mask, timestamp, lat, lon = get_shadow_from_ind(shadow_nc_file, ind=ind)
        print(f"res: {res}, min: {np.nanmin(shadow_mask)}, max: {np.nanmax(shadow_mask)}")

        filestem = Path(shadow_nc_file).stem
        ts_str = timestamp.strftime("%Y-%m-%d")

        plot_band_with_extent(
            shadow_mask,
            BBOX["west"], BBOX["east"], BBOX["south"], BBOX["north"],
            f"output/{filestem}_{ts_str}"
        )

    
    
    