# ========================================================================
# Various plotting functions for monthly/seasonal plots of some spatial
# variable (cloud cover, ghi, irradiation,...) including coastline,
# lat/lon grid, and ground station location. 
# ========================================================================

import rasterio
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Point
from netCDF4 import Dataset
import geopandas as gpd
from shapely.geometry import box
from src.plotting import set_paper_style

# ------------------------------------------
# Set-up 
# ------------------------------------------

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
    """Read .tif image from path and append to data_list."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(float) # convert to %
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data_list.append(data)
        print(f"pixel range: min: {np.round(np.nanmin(data),3)}, max: {np.round(np.nanmax(data),3)}")
        print(f"max pixel difference: {np.round((np.nanmax(data) - np.nanmin(data)),3)} %")
 
 
def plot_band_with_extent(data, left, right, bottom, top, outpath):
    """Plot single band data with extent information (left, right, bottom, top).
    Save plot to outpath. """

    print(f"Extent (left, right, bottom, top): ({left}, {right}, {bottom}, {top}) ")
    print(f"Data shape: {data.shape}")
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        extent=(left, right, bottom, top),
        cmap='gray',  
        origin='upper', 
        vmin=0, 
        vmax=1
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("")  

    # Save to outpath
    plt.savefig(f"{outpath}.png", dpi=150, bbox_inches='tight')
    print(f"Saved figure to {outpath}.")
           
        
def plot_tif_with_latlon(tif_path, outpath):
    """Read .tif data from tif_path, plot with Latitude (x-axis) and 
    longitude (y-axis) information and save to outpath.
    """
    # Open raster file
    with rasterio.open(tif_path) as src:
        data = src.read(1)

        # Compute coordinate bounds for axes
        left, bottom, right, top = src.bounds
    
    plot_band_with_extent(data, left, right, bottom, top, outpath)
    


def plot_monthly_results(source, var_name, outpath, title, colorbar_ylabel, histogram_title, dpi=150, 
                         cmap = None, norm=None):
    """
    Plot monthly spatial maps and corresponding histograms for a given variable.

    This function generates 12 monthly subplots of spatial data (e.g., cloud cover or GHI)
    and saves the figure. It also creates histograms for each month showing the distribution
    of the variable values.

    Parameters
    ----------
    source : dict or str
        - dict: mapping month number (1–12) to file paths containing monthly data
        - str: path to a NetCDF file containing the variable `var_name` with dimension "month".
    var_name : str
        Name of the variable to plot from the NetCDF file (ignored if `source` is a dict).
    outpath : str
        File path to save the monthly map figure.
    title : str
        Title for the map figure.
    colorbar_ylabel : str
        Label for the colorbar in the map figure and x-axis in histograms.
    histogram_title : str
        Title for the histogram figure.
    dpi : int, optional
        Resolution of the output figures. Default is 150.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the maps and histograms. Default is "viridis".
    norm : matplotlib.colors.Normalize, optional
        Normalization for the color scale. If None, automatically set from data.

    Outputs
    -------
    Saves two figures to disk:
    - Monthly spatial maps with shared color scale (saved to `outpath`)
    - Histograms of monthly values (saved to `*_histograms.png`)
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
        ds = xr.open_dataset(source)
        data = ds[var_name].values # shape: (12, lat, lon)
        ds.close()
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
    
    
    if cmap is None:
        cmap = cm.get_cmap("viridis")
    if norm is None: 
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
    

def plot_seasonal_results(source, var_name, count_var, outpath, title, colorbar_label, histogram_title, dpi=150, cmap=None):
    """
    Compute and plot seasonal mean maps and histograms from monthly data.

    This function aggregates monthly spatial data into four meteorological seasons 
    (Winter: Dec–Feb, Spring: Mar–May, Summer: Jun–Aug, Autumn: Sep–Nov) using 
    monthly counts as weights. It then generates seasonal mean maps and histograms.

    Parameters
    ----------
    source : str or dict
        - str: Path to a NetCDF file containing `var_name` (monthly data) and `count_var` (monthly observation counts)
        - dict: {season_name: filepath} with pre-aggregated seasonal images
    var_name : str
        Variable name in the NetCDF file to process (ignored if `source` is a dict)
    count_var : str
        Name of the monthly counts variable in the NetCDF file used for weighting seasonal means
    outpath : str
        File path to save seasonal map figure
    title : str
        Title for the seasonal map figure
    colorbar_label : str
        Label for the colorbar in map plots and x-axis in histograms
    histogram_title : str
        Title for the histogram figure
    dpi : int, optional
        Resolution of output figures (default: 150)
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the maps and histograms (default: "viridis")

    Outputs
    -------
    Saves two figures to disk:
    - Seasonal mean maps (saved to `outpath`)
    - Seasonal histograms of pixel values (saved to `*_histograms.png`)
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
        print(f"season {season}, vmin: {vmin}, vmax: {vmax}")

        if cmap is None:
            cmap = cm.get_cmap("viridis")
        
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


def plot_seasonal_comparison_maps(
        source_nc,
        variable_names,
        variable_names_plotting,
        count_vars,
        outpath,
        title,
        colorbar_label,
        histogram_title,
        cmap=None,
        dpi=150,
    ):
    """
    Creates two figures
    (A) Seasonal comparison maps: 4 seasons × N variables  
        - each row uses shared colorbar
        - weighted seasonal means

    (B) Seasonal comparison histograms: 4 seasons × N variables  
        - histogram colors use same cmap + norm as maps
        - per-season color scale identical to figure (A)

    Parameters
    ----------
    source_nc : str
        Path to .nc file
    variable_names : list[str]
        List of data variable names to compare
    count_vars : list[str]
        Monthly count variable names corresponding to each data variable
    outpath : str
        Output .png filename
    title : str
        Global title
    colorbar_label : str
    cmap : Colormap, optional
    dpi : int
    """

    if cmap is None:
        cmap = cm.get_cmap("viridis")

    # -------------------------------
    # Load data
    # -------------------------------
    with Dataset(source_nc, "r") as nc:
        monthly_data = {v: nc.variables[v][:] for v in variable_names}
        monthly_counts = {c: nc.variables[c][:] for c in count_vars}

    n_vars = len(variable_names)

    # Prepare storage: seasonal_data[season][varname]
    seasonal_data = {s: {} for s in seasons}

    # -------------------------------
    # Compute seasonal weighted means
    # -------------------------------
    for season in seasons:
        idxs = [month - 1 for month in season_months[season]]
        for var, cnt in zip(variable_names, count_vars):
            data_stack = monthly_data[var][idxs]          # shape (3, lat, lon)
            count_stack = monthly_counts[cnt][idxs]       # shape (3,)

            # Replace 0 with NaN so they don't contribute
            weights = np.where(count_stack == 0, np.nan, count_stack)

            weighted_sum = np.nansum([data_stack[i] * weights[i] for i in range(len(idxs))], axis=0)
            total_weight = np.nansum(weights)

            seasonal_mean = weighted_sum / total_weight if total_weight > 0 else np.nan
            seasonal_data[season][var] = seasonal_mean

    # -------------------------------
    # Create figure: 4 rows × n_vars columns
    # -------------------------------
    fig, axes = plt.subplots(
        len(seasons),
        n_vars,
        figsize=(5 * n_vars, 18),
        dpi=dpi,
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    fig.suptitle(title, fontsize=22)

    # -------------------------------
    # Plot rows
    # -------------------------------
    season_norms = {}
    season_min = {}
    season_max = {}
    for r, season in enumerate(seasons):

        # Determine colorbar scale for all variables in this season
        vmin = min(np.nanmin(seasonal_data[season][v]) for v in variable_names)
        vmax = max(np.nanmax(seasonal_data[season][v]) for v in variable_names)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        season_norms[season] = norm
        season_min[season] = vmin
        season_max[season] = vmax

        for c, var in enumerate(variable_names):
            ax = axes[r, c] if n_vars > 1 else axes[r]  # handle N=1 case
            season_map = seasonal_data[season][var]

            # --- Map ---
            ax.set_title(f"{season} - {variable_names_plotting[c]}", fontsize=14)
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            img = ax.imshow(
                season_map, origin="upper",
                extent=extent, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree()
            )

            # Add coastlines
            if coast_clipped is not None or gdf is not None:
                plot_landmarks(ax, coast_clipped, gdf)

            # Gridline rules
            gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (c == 0)
            gl.bottom_labels = (r == len(seasons) - 1)

            # --- Add row-wise colorbar (once per season) ---
            if c == n_vars - 1:  # last column in the row
                cbar = fig.colorbar(
                    img, ax=axes[r, :],
                    orientation="vertical",
                    fraction=0.015, pad=0.02
                )
                cbar.set_label(colorbar_label)

    # -------------------------------
    # Save figure
    # -------------------------------
    plt.savefig(outpath, bbox_inches="tight")
    print(f"✅ Seasonal comparison map saved to {outpath}")
    
    # -----------------------------------------
    # Histograms
    # -----------------------------------------
    
    hist_outpath = outpath.replace(".png", "_histograms.png")

    fig, axes = plt.subplots(
        len(seasons),
        n_vars,
        figsize=(5 * n_vars, 18),
        dpi=dpi,
        constrained_layout=True
    )
    fig.suptitle(histogram_title, fontsize=22)

    for r, season in enumerate(seasons):
        norm = season_norms[season]

        # Used for colorizing histogram bars
        for c, var in enumerate(variable_names):
            ax = axes[r, c] if n_vars > 1 else axes[r]

            data = seasonal_data[season][var].flatten()
            data = data[np.isfinite(data)]

            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center")
                continue

            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            mean_val = np.nanmean(data)
            diff = vmax - vmin

            # Histogram
            counts, bins = np.histogram(data, bins=30, range=(season_min[season], season_max[season]))
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bar_colors = [cmap(norm(b)) for b in bin_centers]

            ax.bar(
                bin_centers, counts,
                width=(bins[1] - bins[0]),
                color=bar_colors, edgecolor="black"
            )

            ax.axvline(mean_val, color='red', linewidth=1.5)

            ax.set_title(f"{season} – {variable_names_plotting[c]}", fontsize=12)
            ax.set_xlabel(colorbar_label)
            ax.set_ylabel("Pixel Count")
            ax.grid(True, linestyle='--', alpha=0.5)

            stats_text = f"min: {vmin:.2f}\nmax: {vmax:.2f}\nΔ: {diff:.2f}"
            ax.text(
                0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8)
            )

    plt.savefig(hist_outpath, bbox_inches="tight")
    print(f"✅ Seasonal comparison histograms saved to {hist_outpath}")


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
    extra_landmarks : dict, optional
        Dictionary of landmark names and (lat, lon) tuples to plot
        Example: {"Askøy": (60.4, 5.15), "Arna": (60.42, 5.45)}
    extra_layers : list, optional
        List of file paths to GeoJSON/SHP layers to overlay (roads, buildings, etc.)
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
