import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

from src.model.surface_GHI_model import get_cloud_shadow_displacement, get_solar_angle

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
        solar_zenith, solar_azimuth = get_solar_angle(dt)
        
        if solar_zenith < 80:
            dx_pix, dy_pix = get_cloud_shadow_displacement(solar_zenith, solar_azimuth, 0, 
                                        sat_zenith, sat_azimuth, pixel_size = 10, cloud_top_height=cth)
            
            dx = dx_pix * 10 
            dy = dy_pix * 10 
            
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
    
    print(f"Number of observations: {len(displacement_x)}")
    
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
    outpath = f"output/cloud_shadow_displacement_hist_cutoff_SZA_80.png"
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}.")

if __name__ == "__main__": 
    cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    analyze_cloud_shadow_displacement(cloud_cover_table_filepath, 2000)