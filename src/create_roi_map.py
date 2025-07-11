import geopandas as gpd
import matplotlib.pyplot as plt

# Load vector data
coastline = gpd.read_file("data/coastline_bergen.geojson")
roads = gpd.read_file("data/roads_primary.geojson")
buildings = gpd.read_file("data/building_footprints_bergen.geojson")


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
