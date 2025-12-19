# =====================================================================
# Plotting functions to display and inspect direct shortwave correction
# factor computed in src.model.shortwave_correction_factor. 
# =====================================================================

import xarray as xr
import numpy as np
import rasterio
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from datetime import datetime, timezone, timedelta
from src.model.cloud_shadow import get_solar_angle
from src.preprocessing.merge_station_obs_with_sim import extract_pixel_by_location, convert_cftime_to_datetime

def plot_sw_cor_for_timestep(sw_cor_file, elevation_file, ind=10):
    """Plot elevation and the shortwave correction factor for a specific timestep
    in two subplots. Function adapted from https://github.com/ChristianSteger/HORAYZON 
    in file examples/shadow/gridded_curved_DEM_SRTM.py"""

    # Load sw cor data
    ds = xr.open_dataset(sw_cor_file)
    sw_dir_cor = ds["sw_dir_cor"][ind, :, :].values
    timestamp = ds["time"].isel(time=ind).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    sun_x = ds["sun_x"].values[ind]
    sun_y = ds["sun_y"].values[ind]
    sun_z = ds["sun_z"].values[ind]
    ds.close()
    timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[s]").astype(int), tz=timezone.utc)
    print("timestamp date: ", timestamp)
    sun_alt = np.degrees(np.arcsin(sun_z / np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)))
    sun_az  = np.degrees(np.arctan2(sun_x, sun_y))
    print(f"Sun altitude: {np.round(sun_alt,2)}, sun azimuth: {np.round(sun_az,2)}")
    
    zenith, azimuth = get_solar_angle(timestamp)
    print(f"Solar angles from pvlib: alt: {np.round(90-zenith,2)}, azimuth: {np.round(azimuth,2)} ")
    
    # Load elevation data 
    with rasterio.open(elevation_file) as src:
        dsm = src.read(1)  # first band

    elevation_ortho = np.ascontiguousarray(dsm)
    
    # Plot
    ax_lim = (lon.min(), lon.max(),
            lat.min(), lat.max())
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                        hspace=0.05, wspace=0.05, width_ratios=[1.0, 0.027])
    ax = plt.subplot(gs[0, 0])
    ax.set_facecolor(plt.get_cmap("terrain")(0.15)[:3] + (0.25,))
    
    # mask lov elevation (water)
    masked_elev = np.ma.masked_where(elevation_ortho < 0.5, elevation_ortho)
    levels = np.arange(0.0, 800.0, 50.0)
    cmap_terrain = colors.LinearSegmentedColormap.from_list(
        "terrain", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 100))
    )
    cmap = colors.ListedColormap(['blue'] + list(cmap_terrain(np.linspace(0,1,100))))
    norm = colors.BoundaryNorm(np.concatenate(([0, 0.5], levels[1:])), cmap.N, extend='max')
    #data_plot = np.ma.masked_where(mask_ocean[slice_in], elevation_ortho)
    plt.pcolormesh(lon, lat, elevation_ortho,
                cmap=cmap, norm=norm)
    x_ticks = np.arange(np.nanmin(lon), np.nanmax(lon), 0.05)
    plt.xticks(x_ticks, ["" for i in x_ticks])
    y_ticks = np.arange(np.nanmin(lat), np.nanmax(lat), 0.02)
    plt.yticks(y_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
    plt.axis(ax_lim)
    ax = plt.subplot(gs[0, 1])
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    plt.ylabel("Elevation [m a.s.l.]", labelpad=10.0)
    ax = plt.subplot(gs[1, 0])
    levels = np.arange(0.0, 2.25, 0.25)
    ticks = np.arange(0.0, 2.5, 0.5)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
    plt.pcolormesh(lon, lat, sw_dir_cor,
                cmap=cmap, norm=norm)
    plt.xticks(x_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
    plt.yticks(y_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
    plt.axis(ax_lim)
    txt = timestamp.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    t = plt.text(0.835, 0.935, txt, fontsize=11, fontweight="bold",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    #ts = load.timescale()
    #astrometric = loc_or.at(ts.from_datetime(timestamp).observe(sun)
    #alt, az, d = astrometric.apparent().altaz()
    txt = "Mean solar elevation angle: %.1f" % sun_alt + "$^{\circ}$"
    t = plt.text(0.21, 0.06, txt, fontsize=11, fontweight="bold",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax = plt.subplot(gs[1, 1])
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks,
                            orientation="vertical")
    plt.ylabel("${\downarrow}SW_{dir}$ correction factor [-]", labelpad=10.0)
    ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    outpath = f"data/processed/sw_cor/Elevation_sw_dir_cor_{ts_str}.png"
    fig.savefig(outpath, dpi=300,
                bbox_inches="tight")
    print(f"Saved elevation and sw dir plot to {outpath}.")
    plt.close(fig)
    

def plot_sw_dir_cor_at_locations(lat_list, lon_list, label_list, filepath, outpath, title):
    """
    Plot the direct shortwave correction factor (sw_dir_cor) time series
    for multiple locations (e.g. flesland and florida stations) at hour == 11.

    Parameters
    ----------
    lat_list : list of float
        Latitudes of locations.
    lon_list : list of float
        Longitudes of locations.
    label_list : list of str
        Labels describing each location.
    filepath : str
        Path to the NetCDF file.
    outpath : str
        Path to save the resulting plot.
    title : str
        Plot title.
    """
    assert len(lat_list) == len(lon_list) == len(label_list), \
        "lat_list, lon_list, and label_list must have the same length."

    plt.figure(figsize=(10, 5))

    for lat, lon, label in zip(lat_list, lon_list, label_list):
        # Extract pixel time series
        times, sw_dir_cor = extract_pixel_by_location(filepath, lat, lon, var_name="sw_dir_cor")
        print(f"times before conversion: {times[:5]}")

        # Convert times to pandas datetime (if not already)
        times = [convert_cftime_to_datetime(t) for t in times]
        times = pd.to_datetime(times)
        print(f"times: {times[:5]}")

        # Filter for hour == 11
        mask = times.hour == 11 
        print(f"mask: {mask[:5]}")
        times_11 = times[mask]
        sw_dir_cor_11 = sw_dir_cor[mask]

        # Plot one line per location
        plt.plot(times_11, sw_dir_cor_11, linewidth=1.2, label=label)

    # Add horizontal reference line
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Reference = 1.0")

    # Labels, title, and formatting
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Direct shortwave correction factor")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.gcf().autofmt_xdate()

    # Save and close
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✅ Plot saved to {outpath}")


def check_pixel_location(nc_filepath, var_name, lat_target, lon_target, time_index=0, zoom_radius=200, enlargement=10, title="Pixel Check"):
    """
    Load one time step from a NetCDF file and highlight the pixel closest to the given lat/lon.
    Zoom in around the pixel with a specified radius. Plot the shortwave correction factor 
    for the selected radius around the target location. 
    
    Parameters
    ----------
    nc_filepath : str
        Path to the NetCDF file.
    var_name : str
        Name of the variable to plot.
    lat_target : float
        Latitude of the pixel to check.
    lon_target : float
        Longitude of the pixel to check.
    time_index : int, optional
        Time index to load from the dataset (default 0).
    zoom_radius : int, optional
        Number of pixels to include around the selected pixel in each direction (default 200).
    enlargement : float, optional
        Factor to enlarge the highlighted pixel marker.
    title : str, optional
        Plot title.
    """
    # Load NetCDF
    ds = xr.open_dataset(nc_filepath)
    var = ds[var_name].isel(time=time_index)

    # Get latitude and longitude arrays
    lat = ds["lat"].values
    lon = ds["lon"].values

    # Find closest pixel index
    lat_idx = np.argmin(np.abs(lat - lat_target))
    lon_idx = np.argmin(np.abs(lon - lon_target))
    
    # Determine the zoom window indices
    lat_min_idx = max(lat_idx - zoom_radius, 0)
    lat_max_idx = min(lat_idx + zoom_radius + 1, len(lat))
    lon_min_idx = max(lon_idx - zoom_radius, 0)
    lon_max_idx = min(lon_idx + zoom_radius + 1, len(lon))
    
    # Extract zoomed data
    data_zoom = var.values[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
    lat_zoom = lat[lat_min_idx:lat_max_idx]
    lon_zoom = lon[lon_min_idx:lon_max_idx]
    
    print(f"lat : {lat_zoom[:5]}")
    
    # Flip latitude axis if needed
    if lat_zoom[0] > lat_zoom[-1]:  # descending
        data_zoom = np.flipud(data_zoom)
        lat_zoom = lat_zoom[::-1]

    # Pixel coordinates in zoomed array
    pixel_lon = lon[lon_idx]
    pixel_lat = lat[lat_idx]

    # Print selected time
    times = ds["time"].values
    selected_time = pd.to_datetime(times[time_index])
    print(f"Selected time for time_index={time_index}: {selected_time}")
    print(f"Selected pixel lat/lon: ({pixel_lat}, {pixel_lon})")

    # Plot zoomed 2D variable
    plt.figure(figsize=(6,6))
    plt.imshow(data_zoom, origin='lower',
               extent=[lon_zoom.min(), lon_zoom.max(), lat_zoom.min(), lat_zoom.max()],
               aspect='auto')
    plt.colorbar(label=var_name)
    
    # Plot exactly one red dot at the selected pixel
    plt.scatter([pixel_lon], [pixel_lat], s=enlargement, color='red', marker='o', label='Selected Pixel')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title + f" {selected_time}")

    # Move legend outside
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.show()
    
    ds.close()



def main():
    sw_cor_filepath = "data/processed/sw_cor/sw_cor_bergen.nc"
    DSM_filepath = "data/processed/bergen_dsm_10m_epsg4326_reducer_mean.tif"
    plot_sw_cor_for_timestep(sw_cor_file=sw_cor_filepath, elevation_file=DSM_filepath, ind=100)
    
    Florida_lat_maps, Florida_lon_maps = 60.38375436372568, 5.331906586858453
    airport_lat_maps, airport_lon_maps = 60.28493807989472, 5.222414640437133
    
    plot_sw_dir_cor_at_locations(lat_list=[airport_lat_maps], lon_list=[airport_lon_maps], label_list=["Airport"],
                                 filepath=sw_cor_filepath, 
                                 outpath="output/sw_cor_at_Flesland_Florida_pixels_11UTC_airport.png", 
                                 title="Direct shortwave correction factor at Flesland and Florida pixels (11 UTC)")
    check_pixel_location(sw_cor_filepath, "sw_dir_cor", Florida_lat_maps, Florida_lon_maps, time_index=148, zoom_radius=50, enlargement=1)
    
if __name__ == "__main__":
    main()