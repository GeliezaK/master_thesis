import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import dataset_features
from shapely.affinity import scale
from shapely.geometry import shape
import numpy as np

def plot_map_with_elevation(dsm_filepath, coastline_filepath, roads_filepath, buildings_filepath):
    """Plot DSM elevation with coastline, roads and buildings (clipped to DSM extent)"""

    # ---------------------------
    # Load elevation raster
    # ---------------------------
    with rasterio.open(dsm_filepath) as src:
        elevation = src.read(1, masked=True)  # masked array handles NoData
        extent = [
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        ]
        transform = src.transform

        # Create mask polygon from valid data
        mask_shapes = dataset_features(src, bidx=1, sampling=1, geographic=True)
        mask_polygons = [shape(feat["geometry"]) for feat in mask_shapes]

    # ---------------------------
    # Load vector data
    # ---------------------------
    coastline = gpd.read_file(coastline_filepath)
    roads = gpd.read_file(roads_filepath)
    buildings = gpd.read_file(buildings_filepath)

    # Clip coastline and roads to DSM valid data
    if mask_polygons:
        mask_union = gpd.GeoSeries(mask_polygons).unary_union
        coastline = gpd.clip(coastline, mask_union)
        roads = gpd.clip(roads, mask_union)
        buildings = gpd.clip(buildings, mask_union)

    # Enlarge building footprints by factor 5
    buildings["geometry"] = buildings.geometry.apply(
        lambda g: scale(g, xfact=5, yfact=5, origin="center")
    )

    # ---------------------------
    # Plot combined map
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12, 12))

    # Raster background
    im = ax.imshow(
        elevation,
        cmap="terrain",
        extent=extent,
        origin="upper"
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Elevation (m)")

    # Vector overlays
    buildings.plot(ax=ax, color="red", alpha=0.5, label="Buildings (enlarged ×5)")
    roads.plot(ax=ax, color="blue", linewidth=1, label="Primary Roads (clipped)")
    coastline.plot(ax=ax, color="black", linewidth=1.2, label="Coastline (clipped)")

    # Annotations
    ax.set_title("Bergen Map with Elevation, Roads, and Buildings", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    plt.tight_layout()
    plt.savefig("output/bergen_map_with_elevation.png", dpi=300)
    plt.show()


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
    summarize_dsm(dsm_filepath)