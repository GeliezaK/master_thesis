import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Point
from netCDF4 import Dataset, date2num, num2date
import geopandas as gpd
from shapely.geometry import box
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
from src.plotting import set_paper_style

set_paper_style()

# Folder path
folder = "data/processed"

# Month number name map 
months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
              9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_names = ["January", "February", "March", "April", "May", "June", "July", "August", 
               "September", "October", "November", "December"]

# List of season names 
seasons = ["Winter", "Spring", "Summer", "Autumn"]

season_months = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }

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
    #"Bergen Center": (60.39299, 5.32415),
    #"Bergen Airport": (60.2934, 5.2181),
    "Florida": (60.3833, 5.3333),
    "Flesland": (60.292792, 5.222689)
}

# Convert landmarks to GeoDataFrame in UTM32
gdf = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in landmarks.values()],
    index=landmarks.keys(),
    crs=crs
).to_crs(crs.to_string())

def plot_landmarks(ax, coast_clipped, gdf):
    # Add coastlines
    coast_clipped.plot(ax=ax, color='black', linewidth=0.75, label="Coastline",
                       transform=ccrs.PlateCarree())  # EPSG:4326

    # Plot landmarks
    for name, point in gdf.iterrows():
        ax.scatter(
            point.geometry.x,
            point.geometry.y,
            marker='^',          # triangle
            s=100,               # size of marker
            facecolor='yellow',  # fill color
            edgecolor='black',   # outline color
            linewidth=1.5,
            zorder=6,
            transform=ccrs.PlateCarree()
        ) 
        
        txt = ax.text(
            point.geometry.x + 0.005, point.geometry.y + 0.005,
            name,
            fontsize=14,
            color='yellow',
            weight='bold',
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        
        # Add black outline for readability
        txt.set_path_effects([
                PathEffects.Stroke(linewidth=3, foreground='black'),
                PathEffects.Normal()
        ]) 


def read_image_and_append(path, data_list): 
    with rasterio.open(path) as src:
        data = src.read(1).astype(float) # convert to %
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data_list.append(data)
        print(f"pixel range: min: {np.round(np.nanmin(data),3)}, max: {np.round(np.nanmax(data),3)}")
        print(f"max pixel difference: {np.round((np.nanmax(data) - np.nanmin(data)),3)} %")
 
 
def plot_band_with_extent(data, left, right, bottom, top, outpath):
    print(f"Extent (left, right, bottom, top): ({left}, {right}, {bottom}, {top}) ")
    print(f"Data shape: {data.shape}")
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        extent=(left, right, bottom, top),
        cmap='gray',  # or 'viridis', 'Blues', etc.
        origin='upper'
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("")  # no title as requested

    # Save to outpath
    plt.savefig(f"{outpath}.png", dpi=150, bbox_inches='tight')
    print(f"Saved figure to {outpath}.")
           
        
def plot_tif_with_latlon(tif_path, outpath):
    # Open raster file
    with rasterio.open(tif_path) as src:
        data = src.read(1)

        # Compute coordinate bounds for axes
        left, bottom, right, top = src.bounds
    
    plot_band_with_extent(data, left, right, bottom, top, outpath)
    


def plot_monthly_results(source, var_name, outpath, title, colorbar_ylabel, histogram_title, dpi=150, 
                         value_ranges=None, value_colors=None):
    """
    Plot monthly cloud cover or ghi maps (12 subplots) + histograms.
    
    Parameters
    ----------
    source : dict or str
        - dict: {month_number: filepath, ...}
        - str: path to .nc file containing variables "month" and "shadow_frequency"
    outpath : str
        Output filepath for map plot (histograms will be saved separately)
    title : str
        Title of the map plot
    colorbar_ylabel : str
        Label for the colorbar and histogram x-axis
    histogram_title : str
        Title of the histogram plot
    dpi : int
        Resolution of output figure
    extent : list [lon_min, lon_max, lat_min, lat_max], optional
        Map extent
    months_map : dict
        Mapping from month index (1–12) to month name
    coast_clipped, gdf : optional
        GeoDataFrames for plotting landmarks/borders
    """
    
    monthly_data = []

    # -------------------------------------------------------------------------
    # Case 1: dictionary of file paths
    # -------------------------------------------------------------------------
    if isinstance(source, dict):
        for month, path in source.items():
            read_image_and_append(path, monthly_data)

    # -------------------------------------------------------------------------
    # Case 2: NetCDF file
    # -------------------------------------------------------------------------
    elif isinstance(source, str) and source.endswith(".nc"):
        with Dataset(source, "r") as nc:
            data = nc.variables[var_name][:]  # shape (12, lat, lon)
        for i in range(data.shape[0]):
            monthly_data.append(data[i, :, :])
    else:
        raise ValueError("source must be a dict {month: filepath} or a .nc file path")

    # -------------------------------------------------------------------------
    # Shared color scale
    # -------------------------------------------------------------------------
    vmin = np.nanmin([np.nanmin(d) for d in monthly_data])
    vmax = np.nanmax([np.nanmax(d) for d in monthly_data])
    print(f"vmin: {vmin}")
    print(f"vmax: {vmax}")
    
    
    if value_ranges is not None and value_colors is not None:
        # Ensure last bin catches everything above the last boundary
        if value_ranges[-1] <= vmax:
            boundaries = value_ranges + [vmax + 1e-6]
        else:
            boundaries = value_ranges
        cmap = ListedColormap(value_colors)
        norm = BoundaryNorm(boundaries, cmap.N)
    else:
        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # -------------------------------------------------------------------------
    # Map plots
    # -------------------------------------------------------------------------
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 18), dpi=dpi,
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for idx, month in enumerate(seasonal_months):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        label = months_map[month] if months_map is not None else f"Month {month}"
        ax.set_title(label)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        id_data = idx - 1 if idx > 0 else 11
        img = ax.imshow(monthly_data[id_data], origin="upper", transform=ccrs.PlateCarree(),
                        extent=extent, cmap=cmap, norm=norm)

        plot_landmarks(ax, coast_clipped, gdf)
        
        # ---------------------------------------------------------------------
        # Control which labels appear
        # ---------------------------------------------------------------------
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col == 0)           # first column only
        gl.bottom_labels = (row == nrows-1)   # bottom row only

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    fig.colorbar(img, cax=cbar_ax, label=colorbar_ylabel)
    plt.subplots_adjust(wspace=0.0, hspace=0.1)
    fig.suptitle(title, y=0.95)

    plt.savefig(outpath)
    print(f"Monthly map plot saved to {outpath}")

    # -------------------------------------------------------------------------
    # Histogram plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    fig.suptitle(histogram_title, fontsize=16)


    for idx, month in enumerate(seasonal_months):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        id_data = idx - 1 if idx > 0 else 11
        data = monthly_data[id_data].flatten()
        label = months_map[month] if months_map is not None else f"Month {month}"

        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        mean_val = np.nanmean(data)
        diff = max_val - min_val

        counts, bins = np.histogram(data, bins=30, range=(vmin, vmax))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bar_colors = [cmap(norm(b)) for b in bin_centers]

        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=bar_colors, edgecolor="black")
        ax.axvline(mean_val, color="red", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.set_title(label, fontsize=12)

        stats_text = f"min: {min_val:.2f}\nmax: {max_val:.2f}\nmean: {mean_val:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

        ax.set_xlim(vmin, vmax)
        ax.set_xlabel(colorbar_ylabel)
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Hide ticks/labels except left column (y) and bottom row (x)
        if col > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        if row < len(axes) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_outpath = os.path.splitext(outpath)[0] + "_histograms.png"
    plt.savefig(hist_outpath)
    print(f"Monthly histogram plot saved to {hist_outpath}")
    

def plot_seasonal_results(source, var_name, outpath, title, colorbar_label, histogram_title, dpi=150, value_colors=None):
    """
    Compute and plot seasonal mean maps + histograms from monthly data.

    Supports:
    - A dict of monthly image paths, or
    - A NetCDF file with monthly data (shape: 12 x lat x lon)

    The function aggregates months into seasons using _monthly_count
    (number of valid images) as weights.
    
    Seasons:
        - Winter: Dec, Jan, Feb
        - Spring: Mar, Apr, May
        - Summer: Jun, Jul, Aug
        - Autumn: Sep, Oct, Nov
    """

    # -------------------------------------------------------------------------
    # Define month-to-season mapping
    # -------------------------------------------------------------------------
    
    seasons = list(season_months.keys())
    seasonal_data = []
    
    # -------------------------------------------------------------------------
    # Case 1: NetCDF file input
    # -------------------------------------------------------------------------
    if isinstance(source, str) and source.endswith(".nc"):
        with Dataset(source, "r") as nc:
            data = nc.variables[var_name][:]  # shape (12, lat, lon)
            months = nc.variables["month"][:]
            count_var = f"GHI_total_monthly_count"
            if count_var in nc.variables:
                monthly_counts = nc.variables[count_var][:]  # shape (12,)
            else:
                raise KeyError(f"{count_var} not found in .nc file!")

        # Aggregate monthly data → seasonal means (weighted by counts)
        for season in seasons:
            months_idx = [m - 1 for m in season_months[season]]
            valid_counts = np.array([monthly_counts[i] for i in months_idx])
            valid_counts = np.where(valid_counts == 0, np.nan, valid_counts)
            weighted_sum = np.nansum([data[i] * valid_counts[j] for j, i in enumerate(months_idx)], axis=0)
            total_weights = np.nansum(valid_counts)
            seasonal_mean = weighted_sum / total_weights if total_weights > 0 else np.nan
            seasonal_data.append(seasonal_mean)

    # -------------------------------------------------------------------------
    # Case 2: Dictionary of image paths
    # -------------------------------------------------------------------------
    elif isinstance(source, dict):
        # Expecting manually pre-aggregated seasonal files
        for season, path in source.items():
            read_image_and_append(path, seasonal_data)
        seasons = list(source.keys())
    else:
        raise ValueError("source must be a .nc file path or a dict {season: filepath}")

    # -------------------------------------------------------------------------
    # Plot Seasonal Maps
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10), dpi=dpi,
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    fig.suptitle(title, fontsize=20)
    
    for idx, season in enumerate(seasons):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        ax.set_title(f"{season}", fontsize=18)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        vmin = np.nanmin(seasonal_data[idx])
        vmax = np.nanpercentile(seasonal_data[idx], 99)

        if value_colors is None:
            cmap = cm.get_cmap("viridis")
        else:
            cmap = mcolors.ListedColormap(value_colors)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


        img = ax.imshow(seasonal_data[idx], origin='upper',
                        extent=extent, cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree())

        if coast_clipped is not None or gdf is not None:
            plot_landmarks(ax, coast_clipped, gdf)
            
        # ---------------------------------------------------------------------
        # Control which labels appear
        # ---------------------------------------------------------------------
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col == 0)           # first column only
        gl.bottom_labels = (row == 1)   # bottom row only

        cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    plt.savefig(outpath, bbox_inches="tight")
    print(f"✅ Seasonal map plot saved to {outpath}")

    # -------------------------------------------------------------------------
    # Plot Histograms
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(histogram_title, fontsize=20)

    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=np.nanmin(seasonal_data), vmax=np.nanmax(seasonal_data))

    for idx, season in enumerate(seasons):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        data = seasonal_data[idx].flatten()

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        mean_val = np.nanmean(data)
        diff = vmax - vmin

        counts, bins = np.histogram(data, bins=30, range=(vmin, vmax))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bar_colors = [cmap(norm(b)) for b in bin_centers]

        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), color=bar_colors, edgecolor='black')
        ax.axvline(mean_val, color='red', linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.set_title(f"{season}", fontsize=12)
        
        stats_text = f"min: {vmin:.1f}\nmax: {vmax:.1f}\nΔ: {diff:.1f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

        ax.set_xlabel(colorbar_label)
        ax.set_ylabel("Pixel Count")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_outpath = os.path.splitext(outpath)[0] + "_histograms.png"
    plt.savefig(hist_outpath, bbox_inches="tight")
    print(f"✅ Seasonal histogram plot saved to {hist_outpath}")

    
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


def plot_single_tif(
    path,
    outpath,
    title,
    colorbar_label,
    cmap=None,
    norm=None,
    extra_landmarks=None,
    extra_layers=None):
    """
    Plot a single-band GeoTIFF with optional overlays.

    Parameters
    ----------
    path : str
        Path to the raster file (.tif)
    outpath : str
        Output path for the PNG
    title : str
        Figure title
    colorbar_label : str
        Label for the colorbar
    cmap, norm : optional
        Colormap and normalization for imshow
    extra_landmarks : dict, optional
        Dictionary of landmark names and (lat, lon) tuples to plot
        Example: {"Askøy": (60.4, 5.15), "Arna": (60.42, 5.45)}
    extra_layers : list, optional
        List of file paths to GeoJSON/SHP layers to overlay (roads, buildings, etc.)
    """
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)

    plot_single_band(
        band,
        outpath,
        title,
        colorbar_label,
        cmap=cmap,
        norm=norm,
        extra_landmarks=extra_landmarks,
        extra_layers=extra_layers
    ) 


def plot_single_band(
    band,
    outpath,
    title,
    colorbar_label,
    cmap=None,
    norm=None,
    extra_landmarks=None,
    extra_layers=None):    
    """
    Plot a single 2D band (e.g. surface irradiance, elevation) using a given colormap and normalization.

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
    cmap : matplotlib Colormap, optional
        Colormap to use (e.g. plt.cm.terrain, plt.cm.viridis).
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap (e.g. BoundaryNorm, Normalize).
    """

    min_value = np.nanmin(band)
    max_value = np.nanmax(band)

    # Default colormap and normalization
    if cmap is None:
        cmap = plt.cm.viridis
    if norm is None:
        norm = plt.Normalize(vmin=np.round(min_value, 3), vmax=np.round(max_value, 3))

    height, width = band.shape
    fig = plt.figure(figsize=(width/250, height/250), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot the image
    img = ax.imshow(band, cmap=cmap, norm=norm, extent=extent, origin="upper")

    plot_landmarks(ax, coast_clipped, gdf)
    
    # --- EXTRA LAYERS (roads, buildings, etc.) ---
    if extra_layers:
        for layer_path in extra_layers:
            try:
                gdf_layer = gpd.read_file(layer_path).to_crs(crs)
                            # Check if this is a buildings layer
                            
                if "building" in layer_path.lower():
                    # Reproject to projected CRS for buffering (meters)
                    gdf_layer_proj = gdf_layer.to_crs(32632)  # UTM zone for Bergen
                    # Expand polygons by 10 meters
                    gdf_layer_proj["geometry"] = gdf_layer_proj.geometry.buffer(10)
                    # Transform back to raster CRS
                    gdf_layer = gdf_layer_proj.to_crs(crs)
                    
                gdf_layer.plot(ax=ax, linewidth=0.5, color='orange',
                               alpha=0.7, transform=ccrs.PlateCarree())
                print(f"Plotted extra layer: {layer_path}")
            except Exception as e:
                print(f"⚠️ Could not load layer {layer_path}: {e}")

    # --- EXTRA LANDMARKS (points) ---
    if extra_landmarks:
        for name, (lat, lon) in extra_landmarks.items():
            txt = ax.text(
                lon, lat, name,
                fontsize=12, color='white', weight='bold',
                transform=ccrs.PlateCarree(), zorder=6,
            )
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=3, foreground='black'),
                PathEffects.Normal()
            ])

    # Tick intervals
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_xticks(np.linspace(lon_min, lon_max, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(lat_min, lat_max, 5), crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(number_format=".2f")
    lat_formatter = LatitudeFormatter(number_format=".2f")
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=14)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # Title
    plt.title(title, fontsize=16, weight="bold")

    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Band plot saved as PNG to {outpath}.")


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
    