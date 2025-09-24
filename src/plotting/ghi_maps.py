import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timezone
from src.plotting.high_res_maps import plot_single_band

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

    plot_single_band(ghi, f"output/ghi_total_{ts_str}.png", 
            f"GHI Total for {ts_str}",
            "GHI (W/m²)", values, colors)

    
if __name__ == "__main__":
    ghi_maps_filepath = "output/ghi_maps/ghi_total_maps.nc"
    plot_ghi_for_timestep(ghi_file=ghi_maps_filepath, ind=3, outdir="output")
    