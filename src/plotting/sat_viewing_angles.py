# ===========================================================================
# Diagnostic plots to check the distribution of satellite viewing angles
# from Sentinel-2 
# ===========================================================================

import matplotlib.pyplot as plt
import pandas as pd 
import mgrs
import shapely.geometry as geom
import cartopy.crs as ccrs
from src.plotting import STATIONS

cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"

def mean_sat_angles_distribution(cloud_cover_table_filepath, outpath=None):
    """Plot distribution (histogram) of mean azimuth and mean zenith satellite viewing angles."""
    cloud_props = pd.read_csv(cloud_cover_table_filepath)
    cloud_props = cloud_props[["MEAN_AZIMUTH", "MEAN_ZENITH"]]
    # Count NaN values per column
    nan_counts = cloud_props.isna().sum()

    print("NaN counts per column:")
    print(nan_counts)
    
    # Remove NaN values for plotting
    zenith_values = cloud_props["MEAN_ZENITH"].dropna()
    azimuth_values = cloud_props["MEAN_AZIMUTH"].dropna()

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for MEAN_ZENITH
    axes[0].hist(zenith_values, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Distribution of MEAN_ZENITH")
    axes[0].set_xlabel("Zenith Angle [deg]")
    axes[0].set_ylabel("Count")

    # Histogram for MEAN_AZIMUTH
    axes[1].hist(azimuth_values, bins=30, color='salmon', edgecolor='black')
    axes[1].set_title("Distribution of MEAN_AZIMUTH")
    axes[1].set_xlabel("Azimuth Angle [deg]")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if outpath is None: 
        outpath = f"output/mean_sat_viewing_angles_hist.png"
    plt.savefig(outpath)
    print(f"Histogram saved to {outpath}.")
    
    
def mean_sat_angles_per_doy(cloud_cover_filepath, start_date=None, end_date=None, outpath=None):
    """
    Plot mean MEAN_ZENITH and MEAN_AZIMUTH per DOY (day of year) for a sample date range.
    Saves figure to file instead of showing.
    """
    cloud_props = pd.read_csv(cloud_cover_filepath)
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    
    # Filter date range if provided
    if start_date is not None:
        cloud_props = cloud_props[cloud_props["date"] >= start_date]
    if end_date is not None:
        cloud_props = cloud_props[cloud_props["date"] <= end_date]

    # Extract DOY
    cloud_props["doy"] = cloud_props["date"].dt.dayofyear

    # Group by DOY and compute mean angles
    mean_angles = cloud_props.groupby("doy")[["MEAN_ZENITH", "MEAN_AZIMUTH"]].mean().reset_index()

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Mean Zenith
    axes[0].plot(mean_angles["doy"], mean_angles["MEAN_ZENITH"], color="skyblue", marker='o', linestyle='-')
    axes[0].set_ylabel("Mean Zenith Angle [deg]")
    axes[0].set_title("Mean Zenith Angle per DOY")

    # Mean Azimuth
    axes[1].plot(mean_angles["doy"], mean_angles["MEAN_AZIMUTH"], color="salmon", marker='o', linestyle='-')
    axes[1].set_xlabel("Day of Year (DOY)")
    axes[1].set_ylabel("Mean Azimuth Angle [deg]")
    axes[1].set_title("Mean Azimuth Angle per DOY")

    plt.tight_layout()

    if outpath is None:
        outpath = "output/mean_sat_viewing_angles_per_doy.png"
    plt.savefig(outpath)
    print(f"Mean angles per DOY plot saved to {outpath}.")
    

def scatter_viewing_angles(cloud_cover_filepath, outpath=None):
    """Plot the mean zenith against mean azimuth angles for all possible combinations in 
    the Sentinel-2 viewing angles data."""
    cloud_props = pd.read_csv(cloud_cover_filepath)

    unique_angles = cloud_props[["MEAN_ZENITH", "MEAN_AZIMUTH"]].drop_duplicates()

    plt.figure(figsize=(6,6))
    plt.scatter(unique_angles["MEAN_ZENITH"], unique_angles["MEAN_AZIMUTH"], color="blue", s=100)
    plt.xlabel("Zenith Angle [deg]")
    plt.ylabel("Azimuth Angle [deg]")
    plt.title("Zenith vs Azimuth Viewing Geometry")
    plt.grid(True)
    if outpath is None:
        outpath = "output/scatter_sat_zenith_azimuth.png"
    plt.savefig(outpath)
    print(f"Scatter plot saved to {outpath}.")
    

def plot_mgrs_tile(tile="32VKM"):
    """Plot location and coastline of a specific mgrs tile."""
    # Initialize converter
    m = mgrs.MGRS()

    # Get the bounding box of the tile (in lat/lon)
    # Note: MGRS strings usually need full precision (e.g. 32VKM0000000000 for lower-left)
    ll = m.toLatLon(tile + "0000000000")  # lower-left corner
    ur = m.toLatLon(tile + "9999999999")  # upper-right corner

    # Build polygon
    bbox = geom.Polygon([
        (ll[1], ll[0]),   # lower-left (lon, lat)
        (ur[1], ll[0]),   # lower-right
        (ur[1], ur[0]),   # upper-right
        (ll[1], ur[0]),   # upper-left
        (ll[1], ll[0])    # back to start
    ])

    # Plot on map
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([ll[1]-1, ur[1]+1, ll[0]-1, ur[0]+1])

    ax.coastlines(resolution="10m")
    ax.gridlines(draw_labels=True)

    # Plot tile
    x, y = bbox.exterior.xy
    ax.plot(x, y, color="red", linewidth=2, label=f"MGRS tile {tile}")

    # Plot landmarks
    for name, (lat, lon) in STATIONS.items():
        ax.scatter(lon, lat, color="blue", s=50, marker="o", transform=ccrs.PlateCarree(), label=name)
        ax.text(lon + 0.02, lat + 0.02, name, fontsize=10, transform=ccrs.PlateCarree())


    plt.title(f"MGRS tile {tile}")
    plt.show()
 

if __name__ == "__main__": 
    mean_sat_angles_distribution(cloud_cover_table_filepath)
    mean_sat_angles_per_doy(cloud_cover_table_filepath, start_date="2024-01-01", end_date="2025-12-31")
    scatter_viewing_angles(cloud_cover_table_filepath)
    viewing_angles_table_path = "data/processed/S2_viewing_angles_full_table.csv"
    viewing_angles = pd.read_csv(viewing_angles_table_path)
    print(viewing_angles[["MEAN_ZENITH", "MEAN_AZIMUTH", "STD_ZENITH", "STD_AZIMUTH"]].describe())
    plot_mgrs_tile("32VLM")
