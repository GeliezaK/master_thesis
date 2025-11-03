import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
from rasterio.features import dataset_features
from shapely.affinity import scale
from shapely.geometry import shape
import numpy as np
from src.plotting.high_res_maps import plot_single_tif

def plot_map_with_elevation(dsm_filepath, coastline_filepath, roads_filepath, buildings_filepath):
    """Plot DSM elevation with coastline, roads and buildings (clipped to DSM extent)"""

    plot_single_tif(dsm_filepath)


def summarize_dsm(dsm_filepath):
    """Load DSM, exclude water (<1m), and print elevation stats and values at key locations"""

    with rasterio.open(dsm_filepath) as src:
        elevation = src.read(1, masked=True)  # masked array (handles NoData)

        # Mask water (<1m)
        elevation_masked = np.ma.masked_less(elevation, 0)

        # Global stats (excluding water and NoData)
        elev_min = float(np.nanmin(elevation_masked))
        elev_max = float(np.nanmax(elevation_masked))
        elev_median = float(np.nanmedian(elevation_masked))

        print("DSM Elevation Statistics (excluding water <1m):")
        print(f"  Min elevation:     {elev_min:.2f} m")
        print(f"  Max elevation:     {elev_max:.2f} m")
        print(f"  50% percentile:    {elev_median:.2f} m")

        # Locations of interest
        locations = {
            "Bergen Center": (60.39299, 5.32415),
            "Bergen Airport": (60.2934, 5.2181)
        }

        print("\nElevation at key locations:")
        for name, (lat, lon) in locations.items():
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                elev_value = float(elevation[row, col])
                print(f"  {name}: {elev_value:.2f} m")
            else:
                print(f"  {name}: outside raster extent")


if __name__ == "__main__":
    dsm_filepath = "data/processed/bergen_dsm_10m_epsg4326.tif"
    """ plot_map_with_elevation(
        dsm_filepath,
        "data/coastline_bergen.geojson",
        "data/roads_primary.geojson",
        "data/building_footprints_bergen.geojson"
    ) """
    cmap = plt.cm.terrain
    norm = Normalize(vmin=0, vmax=800)  # meters
    landmarks = {
        # Islands / Municipal areas
        "Litlesotra": (60.35858, 5.11),       # from Mapcarta / Wikipedia :contentReference[oaicite:0]{index=0}
        "Askøy": (60.46, 5.16),           # from LatLong.net :contentReference[oaicite:1]{index=1}
        "Sotra": (60.29, 5.105),                # approximate from Sotra location listing :contentReference[oaicite:2]{index=2}
        # Outskirts / suburbs
        "Mjølkeråen": (60.4885, 5.265),          # from Mapcarta :contentReference[oaicite:3]{index=3}
        # Island northeast of Bergen
        "Osterøy": (60.47, 5.49),               # from Wikipedia coordinates of the island :contentReference[oaicite:4]{index=4}
        "Gullfjellet →": (60.34, 5.48)
    }
    buildings_path = "data/raw/building_footprints_bergen.geojson"
    plot_single_tif(dsm_filepath, outpath="output/bergen_roi_map.png", 
                    title="Study Area (Bergen)", 
                    colorbar_label="Elevation (m)", 
                    cmap=cmap,
                    norm=norm, 
                    extra_layers=[buildings_path],
                    extra_landmarks=landmarks)