import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import imageio
import xarray as xr
from datetime import datetime, timedelta, timezone
from src.model import BBOX
from src.plotting.high_res_maps import plot_single_band, plot_monthly_results
from src.model.cloud_shadow import get_cloud_shadow_displacement, get_solar_angle, project_cloud_shadow, read_shadow_roi

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
    
def create_shadow_gif(s2_cloud_mask_sample):
    cbh_m = 1000
    sat_zenith = 5
    sat_azimuth = 150.5

    images = []
    timestamp = datetime(2017,8,25,6,0,0)
    endtime = datetime(2017,8,25,18,0,0)

    while timestamp <= endtime:
        print(f"Hour: {timestamp.hour}")
        timestamp_str = timestamp.strftime("%Y%m%dT%H%M%S")
        # Example solar angles (replace with real function)
        solar_zenith, solar_azimuth = get_solar_angle(timestamp)
        dx_m, dy_m = get_cloud_shadow_displacement(solar_zenith, solar_azimuth, cbh_m, sat_zenith, sat_azimuth)
        shadow_new, shadow_bbox = read_shadow_roi(s2_cloud_mask_sample, dx_m, dy_m)
        shadow_old, _ = project_cloud_shadow(s2_cloud_mask_sample, dy_m/10, dx_m/10, BBOX)

        # Plot side-by-side comparison
        frame_path = f"output/gifs/frame_{timestamp_str}.png"
        plot_shadow_comparison(shadow_new, shadow_old, timestamp_str, frame_path)
        images.append(imageio.imread(frame_path))
        timestamp += timedelta(hours=1)

    # Save GIF
    gif_path = "output/shadow_comparison.gif"
    imageio.mimsave(gif_path, images, duration=0.5)
    print("GIF saved at", gif_path)
    

def plot_shadow_comparison(shadow_new, shadow_old, timestamp_str, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(shadow_new, cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title("New method")
    axes[0].axis('off')
    axes[1].imshow(shadow_old, cmap='Greys', vmin=0, vmax=1)
    axes[1].set_title("Old method")
    axes[1].axis('off')
    plt.suptitle(timestamp_str)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cloud_shadow_for_timestep(shadow_nc_file, ind=0, outdir="output"):
    """
    Plot GHI_total for a specific timestep from the NetCDF file.
    """
    # Load dataset
    ds = xr.open_dataset(shadow_nc_file)
    shadow_mask = ds["shadow_mask"].isel(time=ind).values
    timestamp = ds["time"].isel(time=ind).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()

    # Convert timestamp
    timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[s]").astype(int), tz=timezone.utc)
    ts_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")

    # Define colormap and levels
    values = [0,1]
    colors = ["white", "darkgray"]

    plot_single_band(shadow_mask, f"output/shadow_mask_{ts_str}.png", 
            f"Cloud shadow for {ts_str}",
            "Cloud shadow", values, colors)

def plot_shadow_frequency_all_obs(monthly_shadow_frequency_filepath):
    """
    Compute and plot the long-term mean shadow frequency across all months
    from a NetCDF file with monthly means and observation counts.
    
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
    cloud_shadow_nc = "data/processed/cloud_shadow_thresh40.nc"
    monthly_cloud_shadow_nc = "data/processed/cloud_shadow_thresh40_monthly.nc"
    # analyze_cloud_shadow_displacement(cloud_cover_table_filepath, 2000)
    #create_shadow_gif(s2_cloud_mask_sample)
    #plot_cloud_shadow_for_timestep(cloud_shadow_nc, ind=0)
    """ plot_monthly_results(monthly_cloud_shadow_nc_test, var_name="shadow_frequency", 
                         outpath="output/monthly_cloud_shadow_freq_thresh_40.png",
                         title="Monthly Cloud Shadow Frequencies (2015-2025)", 
                         colorbar_ylabel="Frequency of Cloud Shadow", 
                         histogram_title="Distribution of Pixel Values for Monthly Cloud Shadow", 
                         value_ranges=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                         value_colors=["yellow", "limegreen", "green", "teal", "dodgerblue", "mediumblue", 
                                       "darkblue"]) """
    plot_shadow_frequency_all_obs(monthly_shadow_frequency_filepath=monthly_cloud_shadow_nc)
    