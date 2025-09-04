import xarray as xr
import glob, time, os
import pandas as pd
import numpy as np
import geopandas as gpd
import math
from functools import partial
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Convert degree to radian
def deg2rad(deg):
    return deg * math.pi / 180

# Define the center of the bounding box (Bergen, Norway)
CENTER_LAT = 60.39
CENTER_LON = 5.33

# Approximate degree adjustments for 100km x 100km box
DEG_LAT_TO_KM = 111.412  # 1 degree latitude at 60° converted to km (https://en.wikipedia.org/wiki/Latitude)
DEG_LON_TO_KM = 111.317 * math.cos(deg2rad(CENTER_LAT))  # 1 degree longitude converted to km
LAT_OFFSET = 12.5 / DEG_LAT_TO_KM  # ~12.5km north/south
LON_OFFSET = 12.5 / DEG_LON_TO_KM  # ~12.5km east/west (varies with latitude, approximation)

# Define the bounding box
roi = {
    "north": CENTER_LAT + LAT_OFFSET,
    "south": CENTER_LAT - LAT_OFFSET,
    "west": CENTER_LON - LON_OFFSET,
    "east": CENTER_LON + LON_OFFSET
}

def inspect_file(filepath, variable_name):
    # open the netCDF file
    ds = xr.open_dataset(filepath)

    print(ds)  # print an overview (variables, dimensions, attributes)

    # list variables
    print("Variables:", list(ds.variables))

    # inspect one variable, e.g. cloud mask
    variable = ds[variable_name]  # if present in file
    print(variable)

def merge_files_in_folder(folder, outpath):
    """Merge all .nc files in folder into one .nc file. Save to outpath."""
    # List all .nc files in the folder
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")])

    print("Found files:", files)

    # Open and combine them along time dimension
    ds = xr.open_mfdataset(files, combine="by_coords")

    # Save to new file
    ds.to_netcdf(outpath)
    print(f"Merged dataset saved to {outpath}")

    
def preprocess(ds, y_idx, x_idx):
    return ds.isel(y=slice(y_idx.min(), y_idx.max()+1),
                   x=slice(x_idx.min(), x_idx.max()+1))


def crop_to_roi(filepattern, outpath, aux_filepath):
    # Get ROI indices from aux file
    aux = xr.open_dataset(aux_filepath, decode_times=False)
    lat = aux['lat'].sel(georef_offset_corrected=1)
    lon = aux['lon'].sel(georef_offset_corrected=1)
    mask = (lat >= roi["south"]) & (lat <= roi["north"]) & \
           (lon >= roi["west"]) & (lon <= roi["east"])
    y_idx, x_idx = np.where(mask)

    print("Number of ROI pixels:", len(y_idx))
    print("Y indices range:", y_idx.min(), "-", y_idx.max())
    print("X indices range:", x_idx.min(), "-", x_idx.max())

    files = sorted(glob.glob(filepattern))
    print(f"Found {len(files)} files")
    if len(files) == 0:
        raise RuntimeError("No files found matching pattern.")

    t0 = time.time()

    # Open template file lazily and crop to ROI
    template = xr.open_dataset(files[0], engine="h5netcdf")
    template_roi = preprocess(template, y_idx, x_idx)
    template.close()  # free memory

    # Open remaining files lazily and crop using indices
    data_list = [template_roi]
    for f in files[1:]:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds_roi = preprocess(ds, y_idx, x_idx)
        data_list.append(ds_roi)
        ds.close()

    # Concatenate along time axis
    ds_all = xr.concat(data_list, dim="time")

    # Materialize
    ds_all.load()
    t1 = time.time()
    print(f"Cropping and concatenating took {t1 - t0:.1f} seconds")

    # Save cropped dataset
    ds_all.to_netcdf(outpath)
    size_after = os.path.getsize(outpath) / 1e6
    print(f"Size after cropping: {size_after:.2f} MB")

    
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



if __name__ == "__main__":
    filepath = "data/claas-3_2018-2020/CMAin20180802134500405SVMSG01UD.nc"
    filepath_comet2 = "data/comet2_all/CFChm202007011000002231000101MA.nc"
    aux_file = "data/claas-3_2018-2020/CM_SAF_CLAAS3_L2_AUX.nc"
    outpath = "data/claas-3_2018-2020/cma_2018-2020.nc"
    merge_files_in_folder("data/claas-3_2018-2020/monthly_cropped",outpath)

    inspect_file(outpath, "cma")
    #crop_to_roi("data/claas-3_test/*.nc", "data/claas-3_test_small_roi.nc", aux_filepath = aux_file)
    #cfc_diurnal_cycle_monthly("data/comet2_roi_month.nc")
    #visualize_peak_hour("data/comet2_roi_month.nc")
    #print(roi)
