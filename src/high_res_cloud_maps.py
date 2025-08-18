import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import box


# Folder path
folder = "data"

# Month number name map 
months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
              9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# List of season names 
seasons = ["Winter", "Spring", "Summer", "Autumn"]


# Define seasonal order for months
seasonal_months = [12, 1, 2,   # Winter: Dec, Jan, Feb
                   3, 4, 5,    # Spring: Mar, Apr, May
                   6, 7, 8,    # Summer: Jun, Jul, Aug
                   9, 10, 11]  # Autumn: Sep, Oct, Nov

# Path to your exported image
filename = "Cloud_mask_median_alltime"
path = f'{folder}/{filename}.tif'

# Read one file to get metadata (transform, CRS)
sample_path = os.path.join(folder, f"{filename}.tif")
with rasterio.open(sample_path) as src:
    transform = src.transform
    raster_bounds = src.bounds
    crs = src.crs
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Load downloaded roads or coastline GeoJSON or SHP
coast = gpd.read_file("data/coastline_bergen.geojson")     

# Reproject to raster CRS (EPSG:32632)
coast = coast.to_crs("EPSG:32632")

# Create bounding box polygon
raster_bbox = box(*raster_bounds)
raster_gdf = gpd.GeoDataFrame({'geometry': [raster_bbox]}, crs=crs)

# Clip roads and coastlines
coast_clipped = gpd.clip(coast, raster_gdf)

# Define key landmarks (in lat/lon)
landmarks = {
    "Bergen Center": (60.39299, 5.32415),
    "Bergen Airport": (60.2934, 5.2181)
}

# Convert landmarks to GeoDataFrame in UTM32
gdf = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in landmarks.values()],
    index=landmarks.keys(),
    crs="EPSG:4326"
).to_crs(crs.to_string())

def plot_landmarks(ax, coast_clipped, gdf):
    # Add coastlines, borders, roads
    coast_clipped.plot(ax=ax, color='black', linewidth=0.5, label="Coastline")

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False

    # Plot landmarks
    for name, point in gdf.iterrows():
        ax.plot(point.geometry.x, point.geometry.y, 'ro', transform=ccrs.UTM(32))
        ax.text(point.geometry.x + 1000, point.geometry.y + 1000, name, fontsize=8,
                transform=ccrs.UTM(32), color='red')

def read_image_and_append(path, data_list): 
    with rasterio.open(path) as src:
        data = src.read(1).astype(float) # convert to %
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data_list.append(data)
        print(f"pixel range: min: {np.round(np.nanmin(data),3)}, max: {np.round(np.nanmax(data),3)}")
        print(f"max pixel difference: {np.round((np.nanmax(data) - np.nanmin(data)),3)} %")

def plot_alltime_cloud_cover(path, outpath): 
    # Plot alltime aggregates 

    with rasterio.open(path) as src:
        cloud_freq = src.read(1, masked=True) * 100 # remove NaN data, convert to %

    min_value = np.nanmin(cloud_freq)
    max_value = np.nanmax(cloud_freq)

    # Plotting the cloud frequency map
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.UTM(32))
    ax.set_extent(extent, crs=ccrs.UTM(32))
    img = ax.imshow(cloud_freq, cmap='viridis', vmin=np.round(min_value,3), vmax=np.round(max_value,3), transform=ccrs.UTM(32), extent=extent)

    plot_landmarks(ax, coast_clipped, gdf)

    fig.colorbar(img, label='Cloud Frequency (%)')
    plt.title('All-Time Cloud Frequency (10m)')
    plt.savefig(outpath)
    print(f"Mean alltime cloud cover plot saved to {outpath}.")

    # Histogram of pixel values 
    hist_outpath = os.path.splitext(outpath)[0] + "_histogram.png"
    plt.figure(figsize=(8, 4))
    plt.hist(cloud_freq.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Cloud Frequency Pixel Values')
    plt.xlabel('Cloud Frequency (%)')
    plt.ylabel('Pixel Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(hist_outpath)
    print(f"Mean alltime cloud cover plot (pixel value histograms) saved to {hist_outpath}.")

def plot_monthly_cloud_cover(path, outpath): 
    # Pre-allocate list to store data
    monthly_data = []

    # Read all 12 monthly images
    for month in range(1, 13):
        path = os.path.join(folder, f"Cloud_mask_mean_month{month}_mixed.tif")
        print(f"{month}")
        read_image_and_append(path, monthly_data)


    # Determine shared color scale (min/max) across all months
    vmin = np.nanmin([np.nanmin(d) for d in monthly_data])
    vmax = np.nanmax([np.nanmax(d) for d in monthly_data])


    # Plot in 4 rows × 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(14,14), subplot_kw={'projection': ccrs.UTM(32)})
    fig.suptitle("Monthly Cloud Frequency Maps (mean)", fontsize=16)


    for idx, month in enumerate(seasonal_months):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        ax.set_title(months_map[month], fontsize=12)
        ax.set_extent(extent, crs=ccrs.UTM(32))
        
        img = ax.imshow(monthly_data[month-1], origin='upper', transform=ccrs.UTM(32), extent=extent,
                        cmap='viridis', vmin=vmin, vmax=vmax)

        plot_landmarks(ax, coast_clipped, gdf)        
            

    # Add a single colorbar to the right of all plots
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(img, cax=cbar_ax, label="Cloud Frequency (%)")

    plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)

    plt.savefig(outpath)
    print(f"Mean Monthly Cloud Cover plot saved to {outpath}. ")


    # Pixel values histogramm for each month
    fig, axes = plt.subplots(4, 3, figsize=(14,14))
    fig.suptitle("Monthly Cloud Frequency Histograms", fontsize=16)

    # Define colormap and normalization
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for idx, month in enumerate(seasonal_months):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        data = monthly_data[month-1].flatten()
        
        print(f"Calculating histogramm for {months_map[month]}...")

        # Calculate stats
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        mean_val = np.nanmean(data)
        diff = (max_val - min_val)  # in %
        
        # Histogram data
        counts, bins = np.histogram(data, bins=30, range=(vmin, vmax))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bar_colors = [cmap(norm(b)) for b in bin_centers]

        # Plot histogram
        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=bar_colors, edgecolor='black')
        ax.axvline(mean_val, color='red', linewidth=1.5, label=f"Mean: {mean_val}")
        ax.set_title(f"{months_map[month]}", fontsize=12)
        
        # Add stat box in top-right
        stats_text = f"min: {min_val:.1f}\nmax: {max_val:.1f}\nΔ: {diff:.1f}%"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

        ax.set_xlim(vmin, vmax)
        ax.set_xlabel("Cloud Frequency (%)")
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath_without_ext = os.path.splitext(outpath)[0]
    hist_outpath = outpath_without_ext+"_histograms.png"
    plt.savefig(hist_outpath)
    print(f"Mean Monthly Cloud Cover Histogram plot saved to {hist_outpath}. ")

 
def plot_seasonal_cloud_cover(path, outpath): 
    # Seasonal Maps 
    seasonal_data = []

    # Read images for each season
    for season in seasons:
        path = os.path.join(folder, f"Cloud_mask_mean_{season}_mixed.tif")
        print(f"{season}")
        read_image_and_append(path, seasonal_data)


    # Determine shared color scale (min/max) across all seasons
    vmin = np.nanmin([np.nanmin(d) for d in seasonal_data])
    vmax = np.nanmax([np.nanmax(d) for d in seasonal_data])


    # Plot in 2 rows × 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(10,10), subplot_kw={'projection': ccrs.UTM(32)})
    fig.suptitle("Seasonal Cloud Frequency Maps (mean)", fontsize=16)

    # plot for each season
    for idx, season in enumerate(seasons):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        ax.set_title(f"{season}", fontsize=12)
        ax.set_extent(extent, crs=ccrs.UTM(32))
        
        img = ax.imshow(seasonal_data[idx], origin='upper', transform=ccrs.UTM(32), extent=extent,
                        cmap='viridis', vmin=vmin, vmax=vmax)

        plot_landmarks(ax, coast_clipped, gdf)        
        

    # Add a single colorbar to the right of all plots
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(img, cax=cbar_ax, label="Cloud Frequency (%)")

    plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)

    plt.savefig(outpath)
    print(f"Mean Seasonal cloud cover saved to {outpath}.")

    # Pixel values histogramm for each season
    fig, axes = plt.subplots(2, 2, figsize=(14,14))
    fig.suptitle("Seasonal Cloud Frequency Histograms", fontsize=16)

    # Define colormap and normalization
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for idx, season in enumerate(seasons):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        data = seasonal_data[idx].flatten()
        
        print(f"Calculating histogramm for {season}...")

        # Calculate stats
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        mean_val = np.nanmean(data)
        diff = (max_val - min_val)  # in %
        
        # Histogram data
        counts, bins = np.histogram(data, bins=30, range=(vmin, vmax))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bar_colors = [cmap(norm(b)) for b in bin_centers]

        # Plot histogram
        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=bar_colors, edgecolor='black')
        ax.axvline(mean_val, color='red', linewidth=1.5, label=f"Mean: {mean_val}")
        ax.set_title(f"{season}", fontsize=12)
        
        # Add stat box in top-right
        stats_text = f"min: {min_val:.1f}\nmax: {max_val:.1f}\nΔ: {diff:.1f}%"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

        ax.set_xlim(vmin, vmax)
        ax.set_xlabel("Cloud Frequency (%)")
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_outpath = os.path.splitext(outpath)[0] + "_histograms.png"
    plt.savefig(hist_outpath)

def plot_yearly_cloud_cover(path, outpath):
    """Plot high-resolution map with mean cloud cover for each year."""
    yearly_data = []

    # List of years 
    years = range(2016, 2025)

    # Read images for each year
    for year in years:
        path = os.path.join(folder, f"Cloud_mask_mean_{year}_mixed.tif")
        print(f"{year}")
        read_image_and_append(path, yearly_data)

    # Determine shared color scale (min/max) across all years
    vmin = np.nanmin([np.nanmin(d) for d in yearly_data])
    vmax = np.nanmax([np.nanmax(d) for d in yearly_data])


    # Plot in 3 rows × 3 columns (9 years)
    fig, axes = plt.subplots(3, 3, figsize=(14,14), subplot_kw={'projection': ccrs.UTM(32)})
    fig.suptitle("Yearly Cloud Frequency Maps (mean)", fontsize=16)

    # plot for each season
    for idx, year in enumerate(years):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        ax.set_title(f"{year}", fontsize=12)
        ax.set_extent(extent, crs=ccrs.UTM(32))
        
        img = ax.imshow(yearly_data[idx], origin='upper', transform=ccrs.UTM(32), extent=extent,
                        cmap='viridis', vmin=vmin, vmax=vmax)

        plot_landmarks(ax, coast_clipped, gdf)        

            

    # Add a single colorbar to the right of all plots
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(img, cax=cbar_ax, label="Cloud Frequency (%)")

    plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)

    plt.savefig(outpath)
    print(f"Mean yearly cloud cover plot saved to {outpath}.")


    # Pixel values histogramm for each year
    fig, axes = plt.subplots(3, 3, figsize=(14,14))
    fig.suptitle("Yearly Cloud Frequency Histograms", fontsize=16)

    # Define colormap and normalization
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for idx, year in enumerate(years):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        data = yearly_data[idx].flatten()
        
        print(f"Calculating histogramm for {year}...")

        # Calculate stats
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        mean_val = np.nanmean(data)
        diff = (max_val - min_val)  # in %
        
        # Histogram data
        counts, bins = np.histogram(data, bins=30, range=(vmin, vmax))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bar_colors = [cmap(norm(b)) for b in bin_centers]

        # Plot histogram
        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=bar_colors, edgecolor='black')
        ax.axvline(mean_val, color='red', linewidth=1.5, label=f"Mean: {mean_val}")
        ax.set_title(f"{year}", fontsize=12)
        
        # Add stat box in top-right
        stats_text = f"min: {min_val:.1f}\nmax: {max_val:.1f}\nΔ: {diff:.1f}%"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

        ax.set_xlim(vmin, vmax)
        ax.set_xlabel("Cloud Frequency (%)")
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_outpath = os.path.splitext(outpath)[0] + "_histograms.png"
    plt.savefig(hist_outpath)
    print(f"Mean yearly cloud cover plot (pixel value histograms) saved to {hist_outpath}.")

"""
percentile_data = []

# List of filenaes
percentiles = ["Q1", "median", "Q3"]

# Read images for each year
for p in percentiles:
    path = os.path.join(folder, f"Cloud_prob_{p}_alltime.tif")
    print(f"{p}")
    read_image_and_append(path, percentile_data)
    
# Determine shared color scale (min/max) across all years
vmin = np.nanmin([np.nanmin(d) for d in percentile_data])
vmax = np.nanmax([np.nanmax(d) for d in percentile_data])


# Plot in 3 columns (Q1, Median, Q3)
fig, axes = plt.subplots(1, 3, figsize=(18,10), subplot_kw={'projection': ccrs.UTM(32)})
fig.suptitle("Q1, Median and Q3 Percentiles of Cloud Probability", fontsize=16)

# plot for each percentile
for idx, p in enumerate(percentiles):
    ax = axes[idx]

    ax.set_title(f"{p}", fontsize=12)
    ax.set_extent(extent, crs=ccrs.UTM(32))
    
    img = ax.imshow(percentile_data[idx], origin='upper', transform=ccrs.UTM(32), extent=extent,
                    cmap='viridis', vmin=vmin, vmax=vmax)

    plot_landmarks(ax, coast_clipped, gdf)        

# Add a single colorbar to the right of all plots
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
fig.colorbar(img, cax=cbar_ax, label="Cloud Probability (%)")

plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)
plt.savefig("output/cloud_prob_percentiles_landmarks.png")


# Percentiles for each month
monthly_percentile_data = {}

# List of filenaes
percentiles = ["Q1", "median", "Q3"]

vmin = 100
vmax = 0

# Read images for each month
for m in seasonal_months:
    monthly_percentile_data[m] = []
    for p in percentiles:
        path = os.path.join(folder, f"Cloud_prob_{p}_month{m}.tif")
        print(f"{months_map[m]}, {p}")
        read_image_and_append(path, monthly_percentile_data[m])
        
        # Track global min/max
        vmin = min(vmin, np.nanmin([np.nanmin(d) for d in monthly_percentile_data[m]]))
        vmax = max(vmax, np.nanmax([np.nanmax(d) for d in monthly_percentile_data[m]])) 


# Plot in 3 columns (Q1, Median, Q3)
fig, axes = plt.subplots(12, 3, figsize=(15,48), subplot_kw={'projection': ccrs.UTM(32)})
fig.suptitle("Monthly Q1, Median, and Q3 Percentiles of Cloud Probability", fontsize=20)

# plot for each percentile
for row_idx, month in enumerate(seasonal_months):
    for col_idx, p in enumerate(percentiles):
        ax = axes[row_idx, col_idx]
        arr = monthly_percentile_data[month][col_idx]

        ax.set_title(f"{months_map[month]}, {p}", fontsize=12)
        ax.set_extent(extent, crs=ccrs.UTM(32))
        
        img = ax.imshow(monthly_percentile_data[month][col_idx], origin='upper', transform=ccrs.UTM(32), extent=extent,
                        cmap='viridis', vmin=vmin, vmax=vmax)

        plot_landmarks(ax, coast_clipped, gdf)        

# Add a single colorbar to the right of all plots
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
fig.colorbar(img, cax=cbar_ax, label="Cloud Probability (%)")

plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)
plt.savefig("output/cloud_prob_monthly_percentiles_landmarks.png")

 
# Percentiles for each season
seasonal_percentile_data = {}

# List of filenaes
percentiles = ["Q1", "median", "Q3"]

vmin = 100
vmax = 0

# Read images for each month
for s in seasons:
    seasonal_percentile_data[s] = []
    for p in percentiles:
        path = os.path.join(folder, f"Cloud_prob_{p}_{s}.tif")
        print(f"{s}, {p}")
        read_image_and_append(path, seasonal_percentile_data[s])
        
        # Track global min/max
        vmin = min(vmin, np.nanmin([np.nanmin(d) for d in seasonal_percentile_data[s]]))
        vmax = max(vmax, np.nanmax([np.nanmax(d) for d in seasonal_percentile_data[s]])) 


# Plot in 2 rows 3 columns (Q1, Median, Q3)
fig, axes = plt.subplots(4, 3, figsize=(15,18), subplot_kw={'projection': ccrs.UTM(32)})
fig.suptitle("Seasonal Q1, Median, and Q3 Percentiles of Cloud Probability", fontsize=20)

# plot for each percentile
for row_idx, s in enumerate(seasons):
    for col_idx, p in enumerate(percentiles):
        ax = axes[row_idx, col_idx]
        arr = seasonal_percentile_data[s][col_idx]

        ax.set_title(f"{s}, {p}", fontsize=12)
        ax.set_extent(extent, crs=ccrs.UTM(32))
        
        img = ax.imshow(seasonal_percentile_data[s][col_idx], origin='upper', transform=ccrs.UTM(32), extent=extent,
                        cmap='viridis', vmin=vmin, vmax=vmax)

        plot_landmarks(ax, coast_clipped, gdf)        

# Add a single colorbar to the right of all plots
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
fig.colorbar(img, cax=cbar_ax, label="Cloud Probability (%)")

plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)
plt.savefig("output/cloud_prob_seasonal_percentiles_landmarks.png")
"""

def plot_single_tif(path, outpath, title, colorbar_label, histogram_title, hist_ylabel): 
    # Plot alltime aggregates 

    with rasterio.open(path) as src:
        band1 = src.read(1, masked=True) * 100 # remove NaN data, convert to %
        band1 = band1/100 # divide by 100 due to rescaling-error

    min_value = np.nanmin(band1)
    max_value = np.nanmax(band1)

    # Plotting the cloud frequency map
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.UTM(32))
    ax.set_extent(extent, crs=ccrs.UTM(32))
    img = ax.imshow(band1, cmap='viridis', vmin=np.round(min_value,3), vmax=np.round(max_value,3), transform=ccrs.UTM(32), extent=extent)

    plot_landmarks(ax, coast_clipped, gdf)

    fig.colorbar(img, label=colorbar_label)
    plt.title(title)
    plt.savefig(outpath)
    print(f"Tif plot saved as png to {outpath}.")

    # Histogram of pixel values 
    hist_outpath = os.path.splitext(outpath)[0] + "_histogram.png"
    plt.figure(figsize=(8, 4))
    plt.hist(band1.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.title(histogram_title)
    plt.xlabel(colorbar_label)
    plt.ylabel(hist_ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(hist_outpath)
    print(f"Tif pixel value histogram saved to {hist_outpath}.")

if __name__=="__main__": 
    #plot_alltime_cloud_cover("data/Cloud_mask_mean_alltime_mixed.tif", "output/cloud_mask_mean_alltime_mixed.png")
    #plot_monthly_cloud_cover("data/placeholder.tif", "output/cloud_mask_mean_monthly_mixed.png")
    #plot_seasonal_cloud_cover("data/placeholder.tif", "output/cloud_mask_mean_seasonal_mixed.png")
    plot_single_tif("data/surface_ghi_mean_April2020_scale_10.tif", 
                    "output/surface_ghi_mean_April2020_scale_10.png", 
                    "Total GHI at Surface Level (April 2020), 10m", 
                    "GHI (W/m²)", 
                    "Distribution of GHI values (W/m²) (April 2020, 10m)", 
                    "Frequency")
    