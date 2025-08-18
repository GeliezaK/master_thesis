#Create a map of the Bergen ROI for visualization.

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

def open_dsm_file(path): 
    """ Open, read and plot dsm tile file"""
    # Open a tile
    with rasterio.open(path) as src:
        elevation = src.read(1)  # Band 1
        plt.imshow(elevation, cmap='terrain')
        plt.colorbar(label="Elevation (m)")
        plt.title("DSM Tile")
        plt.show()
        
def plot_roi(coastline_filepath, roads_filepath, buildings_filepath): 
    """Plot the ROI using coastline, roads and buildings"""
    # Load vector data
    coastline = gpd.read_file(coastline_filepath)
    roads = gpd.read_file(roads_filepath)
    buildings = gpd.read_file(buildings_filepath)

    # Plot everything
    fig, ax = plt.subplots(figsize=(12, 12))

    # TODO: Elevation layer

    # Vector layers
    buildings.plot(ax=ax, color='red', alpha=0.5, label='Buildings')
    roads.plot(ax=ax, color='blue', linewidth=1, label='Primary Roads')
    coastline.plot(ax=ax, color='black', label='Coastline')

    # Annotations
    ax.set_title("Bergen Map with Elevation, Roads, and Buildings", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/bergen_map_with_elevation.png", dpi=300)

if __name__ == "__main__":
    open_dsm_file("data/Bergen_NEM_1m/dom1/data/dom1-33-103-118.tif")
    #plot_roi("data/coastline_bergen.geojson", "data/roads_primary.geojson", "data/building_footprints_bergen.geojson")
