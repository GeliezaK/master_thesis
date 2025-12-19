# ===================================================================================
# Various plotting functions to display spatial maps of global horizontal irradiance/
# irradiance/irradiation for different sky types or seasons. 
# ===================================================================================

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timezone
from src.plotting.high_res_maps import plot_single_band, plot_monthly_results, plot_seasonal_results, plot_seasonal_comparison_maps

def plot_ghi_for_timestep(ghi_file, ind=0, outdir="output"):
    """
    Plot GHI_total for a specific timestep from the NetCDF file.
    """
    # Load dataset
    ds = xr.open_dataset(ghi_file)
    ghi = ds["GHI_total"].isel(time=ind).values
    timestamp = ds["time"].isel(time=ind).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()

    # Convert timestamp
    timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[s]").astype(int), tz=timezone.utc)
    ts_str = timestamp.strftime("%Y-%m-%d_%H:%M:%S")

    # Define colormap and levels
    values = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]
    colors = ["black", "darkblue", "mediumblue", "blueviolet", "purple", "mediumvioletred",
              "crimson", "deeppink", "salmon", "orangered", "darkorange", "orange", "yellow"]

    # Create colormap and normalization
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(values, cmap.N, extend="max")

    # Pass cmap and norm to your plotting function
    plot_single_band(
        ghi,
        f"output/ghi_total_{ts_str}.png",
        f"GHI Total for {ts_str}",
        "GHI (W/m²)",
        cmap=cmap,
        norm=norm
    )
    

def plot_sky_type_aggregated(aggregated_ghi_maps_outpath, output_dir="output"):
    """
    Reads the aggregated NetCDF and plots mean clear-sky index for each sky type.
    """
    ds_agg = xr.open_dataset(aggregated_ghi_maps_outpath)
    os.makedirs(output_dir, exist_ok=True)

    for sky_type in ds_agg.data_vars.keys():
        data = ds_agg[sky_type].values
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if sky_type == "clear_mean_clear_sky_index" or sky_type == "mixed_mean_clear_sky_index":
            value_ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0]
        else : 
            value_ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                      
        print(f"Mean for sky type {sky_type}: {np.mean(data)} ")

        viridis = plt.cm.viridis(np.linspace(0.0, 1.0, len(value_ranges)))  # avoid very dark & very light edges
        cmap = mcolors.ListedColormap(viridis)
        norm = mcolors.BoundaryNorm(boundaries=value_ranges, ncolors=cmap.N)
        
        out_png = os.path.join(output_dir, f"{sky_type}_clear_sky_index_all_obs_11UTC.png")
        plot_single_band(
            data,
            outpath=out_png,
            title=f"Mean Clear Sky Index ({sky_type.split('_')[0].capitalize()}, 2015–2025, 11:00 UTC)",
            colorbar_label="Clear Sky Index [-]",
            cmap=cmap,
            norm=norm
        )


def plot_irradiance_all_obs(ghi_monthly_maps_filepath):
    """
    Compute and plot the long-term mean irradiance across all months
    from a NetCDF file with monthly means and observation counts.
    
    Parameters
    ----------
    ghi_monthly_maps_filepath : str
        Path to NetCDF file containing variables:
        - 'irradiance' (monthly mean irradiance, W/m²)
        - 'count' (number of valid observations per month)
    """
    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(ghi_monthly_maps_filepath)
    
    irr = ds["I_total"].astype(np.float64)
    count = ds["GHI_total_monthly_count"].astype(np.float64)

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
    value_ranges = [180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 530]
    value_colors = [
        "black", "dimgrey", "darkblue", "blue", "teal", "green",
        "lime", "greenyellow", "yellow", "gold", "orange", "darkorange", "red"
    ]

    cmap = mcolors.ListedColormap(value_colors)
    norm = mcolors.BoundaryNorm(boundaries=value_ranges, ncolors=len(value_colors))
    
    plot_single_band(
        mean_alltime,
        outpath="output/irradiance_total_all_obs_11UTC.png",
        title="Mean Total Surface Irradiance (2015–2025, 11:00 UTC)",
        colorbar_label="I_total (W/m²)",
        cmap=cmap, norm=norm
    )

    return mean_alltime
   

    
    
if __name__ == "__main__":
    ghi_maps_filepath = "data/processed/simulated_ghi_without_terrain_only_mixed_100m.nc"
    monthly_ghi_maps = "data/processed/simulated_irradiance_monthly.nc"
    monthly_clear_sky_index_maps = "data/processed/simulated_clear_sky_index_monthly_mixed_sky.nc"
    monthly_longterm_sim_results = "data/processed/longterm_ghi_spatially_resolved_monthly.nc"
    sim_vs_obs_path = "data/processed/s2_cloud_cover_with_stations_with_pixel_sim.csv"
    aggregated_sky_type_clear_sky_index_outpath = "data/processed/clear_sky_index_sky_type_all_time_11UTC.nc"

    value_ranges=[0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    value_colors=["indigo", "purple", "darkviolet", "darkslateblue", "blue", "cornflowerblue", "mediumturquoise", "mediumspringgreen", 
                  "springgreen", "lime", "greenyellow", "yellow", "gold", "orange", "darkorange", "red"]

    cmap = ListedColormap(value_colors)
    plot_seasonal_comparison_maps(monthly_longterm_sim_results, ["mixed_sky_ghi", "all_sky_ghi"], 
                                  ["Mixed", "All-Sky"], 
                                  ["month_mixed_count", "month_all_sky_count"], 
                                  outpath= "output/longterm_sim_monthly_maps_mixed_vs_all-sky.png",
                         title="Mean Daily Irradiation [kWh/m²] per Season (Model 3)", 
                         colorbar_label="Mean Daily Irradiation [kWh/m²]", 
                         histogram_title="Distribution of Pixel Values for Mean Daily Irradiation [kWh/m²] (Model 3)", 
                         cmap=cmap) 
    
    plot_ghi_for_timestep(ghi_file=ghi_maps_filepath, ind=3, outdir="output")
    plot_monthly_results(monthly_ghi_maps, "I_total", "output/monthly_irradiance_UTC11.png",
                         "Monthly Surface Irradiance (2015-2025, 11:00 UTC)", 
                         "Mean Total Irradiance [W/m²]", 
                         "Distribution of Pixel Values for Monthly Irradiance (11:00 UTC)", 
                         value_ranges=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
                         value_colors=["black", "dimgrey", "darkblue", "blue", "teal", "green",
                                       "lime", "greenyellow", "yellow", "gold", "orange", "darkorange", "red"]) 
    plot_seasonal_results(monthly_ghi_maps, "I_total", "output/seasonal_irradiance_UTC11.png",
                         "Surface Irradiance by Season (2015-2025, 11:00 UTC)", 
                         "Mean Total Irradiance [W/m²]", 
                         "Distribution of Pixel Values for Monthly Irradiance (11:00 UTC)", 
                         value_colors=["black", "dimgrey", "darkblue", "blue", "teal", "green",
                                       "lime", "greenyellow", "yellow", "gold", "orange", "darkorange", "red"])
    plot_irradiance_all_obs(monthly_ghi_maps)
    plot_sky_type_aggregated(aggregated_ghi_maps_outpath=aggregated_sky_type_clear_sky_index_outpath)
    