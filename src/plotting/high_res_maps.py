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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap



# Folder path
folder = "data/processed"

# Month number name map 
months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
              9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_names = ["January", "February", "March", "April", "May", "June", "July", "August", 
               "September", "October", "November", "December"]

# List of season names 
seasons = ["Winter", "Spring", "Summer", "Autumn"]


# Define seasonal order for months
seasonal_months = [12, 1, 2,   # Winter: Dec, Jan, Feb
                   3, 4, 5,    # Spring: Mar, Apr, May
                   6, 7, 8,    # Summer: Jun, Jul, Aug
                   9, 10, 11]  # Autumn: Sep, Oct, Nov

# Path to your exported image

# Read one file to get metadata (transform, CRS)
sample_path = "data/processed/bergen_dsm_10m_epsg4326.tif"
with rasterio.open(sample_path) as src:
    transform = src.transform
    raster_bounds = src.bounds
    crs = src.crs
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    print(f"Transform: {transform}, crs: {crs}, raster bounds: {raster_bounds}")
    print(f"Extent for plotting: {extent}")

# Load downloaded roads or coastline GeoJSON or SHP
coast = gpd.read_file("data/raw/coastline_bergen.geojson")     

# Reproject to raster CRS (EPSG:32632)
coast = coast.to_crs(crs)

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
    crs=crs
).to_crs(crs.to_string())

def plot_landmarks(ax, coast_clipped, gdf):
    # Add coastlines
    coast_clipped.plot(ax=ax, color='black', linewidth=0.5, label="Coastline",
                       transform=ccrs.PlateCarree())  # EPSG:4326

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False

    # Plot landmarks
    for name, point in gdf.iterrows():
        ax.plot(point.geometry.x, point.geometry.y, 'ro',
                transform=ccrs.PlateCarree())  # use PlateCarree for lon/lat
        ax.text(point.geometry.x + 0.01, point.geometry.y + 0.01, name, fontsize=8,
                transform=ccrs.PlateCarree(), color='red')


def read_image_and_append(path, data_list): 
    with rasterio.open(path) as src:
        data = src.read(1).astype(float) # convert to %
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data_list.append(data)
        print(f"pixel range: min: {np.round(np.nanmin(data),3)}, max: {np.round(np.nanmax(data),3)}")
        print(f"max pixel difference: {np.round((np.nanmax(data) - np.nanmin(data)),3)} %")

def plot_monthly_cloud_cover(monthly_file_dict, outpath, title, colorbar_ylabel, histogram_title, dpi): 
    # Pre-allocate list to store data
    monthly_data = []

    # Read all 12 monthly images
    for month, path in monthly_file_dict.items():
        print(f"{month}")
        read_image_and_append(path, monthly_data)

    # Determine shared color scale (min/max) across all months
    vmin = np.nanmin([np.nanmin(d) for d in monthly_data])
    vmax = np.nanmax([np.nanmax(d) for d in monthly_data])

    # Plot in 4 rows × 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(14, 14), dpi=dpi, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(title, fontsize=16)

    for idx, month in enumerate(seasonal_months):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        ax.set_title(months_map[month], fontsize=12)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        img = ax.imshow(monthly_data[month-1], origin='upper', transform=ccrs.PlateCarree(), extent=extent,
                        cmap='viridis', vmin=vmin, vmax=vmax)

        plot_landmarks(ax, coast_clipped, gdf)        
            

    # Add a single colorbar to the right of all plots
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(img, cax=cbar_ax, label=colorbar_ylabel)

    plt.subplots_adjust(wspace=0.05, hspace=0.2, right=0.9)

    plt.savefig(outpath)
    print(f"Monthly plot saved to {outpath}. ")


    # Pixel values histogramm for each month
    fig, axes = plt.subplots(4, 3, figsize=(14,14))
    fig.suptitle(histogram_title, fontsize=16)

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
        ax.set_xlabel(colorbar_ylabel)
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath_without_ext = os.path.splitext(outpath)[0]
    hist_outpath = outpath_without_ext+"_histograms.png"
    plt.savefig(hist_outpath)
    print(f"Monthly histogram plot saved to {hist_outpath}. ")
 
def plot_seasonal_cloud_cover(seasonal_files_dict, outpath, title, colorbar_label, histogram_title, dpi): 
    # Seasonal Maps 
    seasonal_data = []

    # Read images for each season
    for season, path in seasonal_files_dict.items():
        print(f"{season}")
        read_image_and_append(path, seasonal_data)

    # Plot in 2 rows × 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(10,10), dpi=dpi, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(title, fontsize=16)

    for idx, season in enumerate(seasons):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        ax.set_title(f"{season}", fontsize=12)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Compute season-specific min/max
        vmin = np.nanmin(seasonal_data[idx])
        vmax = np.nanmax(seasonal_data[idx])

        img = ax.imshow(seasonal_data[idx], origin='upper',
                        extent=extent, cmap='viridis', vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree())

        plot_landmarks(ax, coast_clipped, gdf)

        # Add a colorbar specific to this subplot
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Seasonal plot saved to {outpath}.")

    # Pixel values histogramm for each season
    fig, axes = plt.subplots(2, 2, figsize=(14,14))
    fig.suptitle(histogram_title, fontsize=16)

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
        ax.set_xlabel(colorbar_label)
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
    fig, axes = plt.subplots(3, 3, figsize=(14,14), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle("Yearly Cloud Frequency Maps (mean)", fontsize=16)

    # plot for each season
    for idx, year in enumerate(years):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        ax.set_title(f"{year}", fontsize=12)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        img = ax.imshow(yearly_data[idx], origin='upper', transform=ccrs.PlateCarree(), extent=extent,
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


def plot_single_tif(path, outpath, title, colorbar_label, value_ranges=None, value_colors=None): 
    # Plot alltime aggregates 
    with rasterio.open(path) as src:
        band1 = src.read(1, masked=True) # remove NaN data
        band1 = band1 

    plot_single_band(band1, outpath, title, colorbar_label, value_ranges, value_colors)
    

def plot_single_band(band, outpath, title, colorbar_label,
                     value_ranges=None, value_colors=None): 
    """
    Plot a single band with either a continuous colormap (default viridis) 
    or a discrete custom colormap defined by value ranges + colors.
    
    Parameters
    ----------
    band : 2D np.ndarray
        Array of values to plot.
    outpath : str
        Path to save the figure.
    title : str
        Title of the plot.
    colorbar_label : str
        Label for the colorbar.
    value_ranges : list, optional
        List of numeric boundaries for the value intervals.
        Example: [0, 0.5, 1, 1.5]
    value_colors : list, optional
        List of colors corresponding to the intervals in value_ranges.
        Length must be len(value_ranges) + 1.
        Example: ["white", "blue", "green", "yellow", "black"]
    """

    min_value = np.nanmin(band)
    max_value = np.nanmax(band)

    height, width = band.shape
    fig = plt.figure(figsize=(width/300, height/300), dpi=300)    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if value_ranges is not None and value_colors is not None:
        # Ensure last bin catches everything above the last boundary
        if value_ranges[-1] <= max_value:
            boundaries = value_ranges + [max_value + 1e-6]
        else:
            boundaries = value_ranges
        cmap = ListedColormap(value_colors)
        norm = BoundaryNorm(boundaries, cmap.N)
        img = ax.imshow(band, cmap=cmap, norm=norm,
                        extent=extent, origin="upper")
    else:
        # Default continuous viridis
        img = ax.imshow(band, cmap='viridis',
                        vmin=np.round(min_value, 3),
                        vmax=np.round(max_value, 3),
                        extent=extent, origin="upper")

    plot_landmarks(ax, coast_clipped, gdf)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # <-- 2 decimals
    ax.tick_params(labelsize=6)

    plt.title(title, fontsize=8)
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Band plot saved as png to {outpath}.")


def plot_single_tif_histogram(path, outpath, title, xlabel, ylabel):
    with rasterio.open(path) as src:
        band1 = src.read(1, masked=True) # remove NaN data
        band1 = band1 

    plot_single_band_histogram(band1, outpath, title, xlabel, ylabel)
    
    
def plot_single_band_histogram(band, outpath, title, xlabel, ylabel):
    min_value = np.nanmin(band)
    max_value = np.nanmax(band)
    print(f"Min pixel value: {min_value}, max pixel value: {max_value}")
 
    # Histogram of pixel values 
    hist_outpath = os.path.splitext(outpath)[0] + ".png"
    plt.figure(figsize=(8, 4))
    plt.hist(band.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(hist_outpath)
    print(f"Tif pixel value histogram saved to {hist_outpath}.")


if __name__=="__main__": 
    #plot_alltime_cloud_cover("data/processed/Cloud_mask_mean_alltime_mixed.tif", "output/cloud_mask_mean_alltime_mixed.png")
    
    plot_single_tif("data/raw/S2_testfiles/S2_cloud_2018-08-05.tif", 
                    "output/s2_2018-08-05.png", 
                    "Cloud Mask", 
                    "Cloud (binary)") 
    plot_single_tif_histogram("data/raw/S2_testfiles/S2_cloud_2018-07-28.tif", 
                              "output/s2_2018-07-28_hist.png", 
                              "Histogramm of pixel values", 
                              "Pixel value",
                              "Frequency")
    