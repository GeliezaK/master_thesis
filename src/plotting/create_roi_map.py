# ====================================================================
# Create plot for overview of the study area (including elevation, 
# building footprints, coastline, weather station locations and 
# district/islands landmarks)
# ====================================================================

import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from src.plotting.high_res_maps import plot_single_tif


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

    cmap = plt.cm.terrain
    norm = Normalize(vmin=0, vmax=800)  # meters
    landmarks = {                       # locations to plot for orientation
        "Litlesotra": (60.35858, 5.11),       # from mapcarta/google maps
        "Askøy": (60.46, 5.16),           
        "Sotra": (60.29, 5.105),                
        "Mjølkeråen": (60.4885, 5.265),          
        "Osterøy": (60.47, 5.49),               
        "Gullfjellet →": (60.34, 5.48)          
    }
    
    # Geojson file with building footprints, downloaded from Open Street Map
    buildings_path = "data/raw/building_footprints_bergen.geojson"
    plot_single_tif(dsm_filepath, outpath="output/bergen_roi_map.png", 
                    title="Study Area (Bergen)", 
                    colorbar_label="Elevation (m)", 
                    cmap=cmap,
                    norm=norm, 
                    extra_layers=[buildings_path],
                    extra_landmarks=landmarks)