# Model surface GHI 
# Inputs:   - filepath to DSM/DEM file 
#           - folderpath to LUT 
#           - folderpath to S2 images

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm
from netCDF4 import Dataset, date2num
import xarray as xr
import geopandas as gpd
from rasterio.windows import from_bounds, Window
from shapely.geometry import box
import glob
import horayzon as hray
import os
import numpy as np
from scipy.ndimage import shift
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import math
from time import time
import pvlib  # for solar position
from skyfield.api import load, wgs84
from datetime import datetime, timezone, timedelta
from rasterio.plot import show
from high_res_cloud_maps import plot_landmarks, plot_single_band, plot_single_band_histogram, plot_single_tif

# Define the center of the bounding box (Bergen, Norway)
CENTER_LAT = 60.39
CENTER_LON = 5.33

# Approximate degree adjustments for 100km x 100km box
DEG_LAT_TO_KM = 111.412  # 1 degree latitude at 60° converted to km (https://en.wikipedia.org/wiki/Latitude)
DEG_LON_TO_KM = 111.317 * math.cos(np.deg2rad(CENTER_LAT))  # 1 degree longitude converted to km
LAT_OFFSET = 12.5 / DEG_LAT_TO_KM  # ~10km north/south
LON_OFFSET = 12.5 / DEG_LON_TO_KM  # ~10km east/west (varies with latitude, approximation)

# Define the bounding box
BBOX = {
    "north": CENTER_LAT + LAT_OFFSET,
    "south": CENTER_LAT - LAT_OFFSET,
    "west": CENTER_LON - LON_OFFSET,
    "east": CENTER_LON + LON_OFFSET
}

LUT_folderpath = "output/LUT"
DSM_filepath = "data/bergen_dem.tif"
S2_folderpath = "data/S2_testfiles"

    
def load_image(filepath): 
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(float)
        profile = src.profile
    return dem, profile


# Get solar zenith and azimuth
def get_solar_angle(datetime, lat=CENTER_LAT, lon=CENTER_LON): 
    solpos = pvlib.solarposition.get_solarposition(datetime, lat, lon)
    zenith = float(solpos['zenith'].values[0])
    azimuth = float(solpos['azimuth'].values[0])
    return zenith, azimuth

def calculate_sw_dir_cor(
    dsm_filepath,
    times, 
    out_dir="output/sw_cor"
):
    """
    Compute correction factor for direct shortwave radiation using Horayzon.

    Parameters
    ----------
    
    Returns
    -------
    out_file : str
        Path to written GeoTIFF.
    """
    # Read Dsm file    
    with rasterio.open(dsm_filepath) as src:
        dsm = src.read(1)  # first band
        transform = src.transform
        crs = src.crs
        print(f"crs: {crs}")
        origin = (transform.c, transform.f)  # (west, north) of top-left corner
        print(f"origin: {origin}")
        # Column and row indices
        nrows, ncols = dsm.shape
        cols = np.arange(ncols)
        rows = np.arange(nrows)

        # Convert column indices (x direction) to longitude
        lon, _ = rasterio.transform.xy(transform, np.zeros_like(cols), cols)
        lon = np.array(lon, dtype=np.float64)  # shape = (ncols,)

        # Convert row indices (y direction) to latitude
        _, lat = rasterio.transform.xy(transform, rows, np.zeros_like(rows))
        lat = np.array(lat, dtype=np.float64)  # shape = (nrows,)

    #lat = lat[::-1]
    #dsm = dsm[::-1, :]
    print("lon:", lon.shape, lon[:5])
    print("lat:", lat.shape, lat[:5])
    print("DSM type:", type(dsm), "dtype:", dsm.dtype, "shape:", dsm.shape)

    # Quick stats, ignoring NaNs
    print("DSM summary (ignoring NaN):")
    print("  min:", np.nanmin(dsm))
    print("  max:", np.nanmax(dsm))
    print("  mean:", np.nanmean(dsm))
    print("  std:", np.nanstd(dsm))
    
    elevation_ortho = np.ascontiguousarray(dsm)
    # orthometric height (-> height above mean sea level)
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"sw_cor_bergen.nc"
    )
    
    ellps = "WGS84"
    
    # Compute ECEF coordinates
    print("DSM first values before correction:", dsm[2:4, 2:4])
    dsm += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]
    print("DSM first values after correction:", dsm[2:4,2:4])
    
    # Quick stats, ignoring NaNs
    print("DSM summary (after correction):")
    print("  min:", np.nanmin(dsm))
    print("  max:", np.nanmax(dsm))
    print("  mean:", np.nanmean(dsm))
    print("  std:", np.nanstd(dsm))
    
    x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                    dsm, ellps=ellps)
    print(f"ECEF: X: {x_ecef[2:4, 2:4]}, \nY: {y_ecef[2:4, 2:4]}, \nZ: {z_ecef[2:4, 2:4]}")
    
    trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[int(len(lon) / 2)], lat_or=lat[int(len(lat) / 2)], ellps=ellps)
    x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

    print(f"ENU: x_enu: {x_enu.shape}, \ny_enu: {y_enu.shape}, \nz_enu: {z_enu.shape}")
    
    # Compute unit vectors (up and north) in ENU coordinates for whole domain
    vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon,lat))
    vec_north_ecef = hray.direction.north_dir(x_ecef, y_ecef,
                                            z_ecef, vec_norm_ecef,
                                            ellps=ellps)
    print(f"Vec_norm_ecef: {vec_norm_ecef.shape}, {vec_norm_ecef[2:4, 2:4, 1]}")
    print(f"Vec_north_ecef: {vec_north_ecef.shape}, {vec_north_ecef[2:4, 2:4, 1]}")
    vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
    vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)

    print(f"Vec_norm_enu:  {vec_norm_enu.shape}, {vec_norm_enu[2:4, 2:4, 1]}")
    print(f"Vec_north_enu: {vec_north_enu.shape}, {vec_north_enu[2:4, 2:4, 1]}")
  
    # Merge vertex coordinates and pad geometry buffer
    vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

    # Compute rotation matrix (global ENU -> local ENU)
    rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)
    print(f"Rotation Matrix shape: ", rot_mat_glob2loc.shape)
    
    # Compute slope (full grid, pad by 1 on each side)
    x_enu_pad = np.pad(x_enu, 1, mode="edge")
    y_enu_pad = np.pad(y_enu, 1, mode="edge")
    z_enu_pad = np.pad(z_enu, 1, mode="edge")
    
    print(f"Shapes of padded: x_enu_pad {x_enu_pad.shape}, y_enu_pad {y_enu_pad.shape}, z_enu_pad {z_enu_pad.shape}")

    vec_tilt_enu = np.ascontiguousarray(
        hray.topo_param.slope_plane_meth(
            x_enu_pad, y_enu_pad, z_enu_pad,
            rot_mat=rot_mat_glob2loc, output_rot=False
        )[1:-1, 1:-1]
    )

    print(f"vec_tilt_enu shape: ", vec_tilt_enu.shape, vec_tilt_enu[2:4, 2:4, 1])
    
    # Compute surface enlargement factor
    surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
    # surf_enl_fac[:] = 1.0
    print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
        + ", %.3f" % surf_enl_fac.max())
    print("surf_enl_fac dtype:", surf_enl_fac.dtype)
    print("surf_enl_fac shape:", surf_enl_fac.shape)
    print("Number of NaNs:", np.isnan(surf_enl_fac).sum())
    print("Number of infs:", np.isinf(surf_enl_fac).sum())
    print("Valid value range (ignoring NaNs):",
        np.nanmin(surf_enl_fac), "to", np.nanmax(surf_enl_fac))
    
    mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
    
    terrain = hray.shadow.Terrain()
    dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
    terrain.initialise(vert_grid, nrows, ncols,
                    0, 0, vec_tilt_enu, vec_norm_enu,
                    surf_enl_fac, mask=mask, elevation=elevation_ortho,
                    refrac_cor=True)
    
    # Load Skyfield data
    load.directory = "data"
    planets = load("de421.bsp")
    sun = planets["sun"]
    earth = planets["earth"]
    loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)
    print("sun: ", sun)
    print("earth: ", earth)
    print("loc_or: ", loc_or)
    # -> position lies on the surface of the ellipsoid by default

    # -----------------------------------------------------------------------------
    # Compute shortwave correction factor
    # -----------------------------------------------------------------------------

    # Loop through time steps and save data to NetCDF file
    ncfile = Dataset(filename=out_file, mode="w")
    ncfile.createDimension(dimname="time", size=None)
    ncfile.createDimension(dimname="lat", size=dim_in_0)
    ncfile.createDimension(dimname="lon", size=dim_in_1)
    nc_time = ncfile.createVariable(varname="time", datatype="f",
                                    dimensions="time")
    nc_time.units = "hours since 2015-01-01 00:00:00"
    nc_time.calendar = "gregorian"
    # Create sun position variables
    nc_sun_x = ncfile.createVariable(varname="sun_x", datatype="f4", dimensions=("time",))
    nc_sun_y = ncfile.createVariable(varname="sun_y", datatype="f4", dimensions=("time",))
    nc_sun_z = ncfile.createVariable(varname="sun_z", datatype="f4", dimensions=("time",))
    nc_sun_x.units = nc_sun_y.units = nc_sun_z.units = "m"
    nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                                dimensions="lat")
    nc_lat[:] = lat
    nc_lat.units = "degree"
    nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                                dimensions="lon")
    nc_lon[:] = lon
    nc_lon.units = "degree"
    nc_data = ncfile.createVariable(varname="sw_dir_cor", datatype="f",
                                    dimensions=("time", "lat", "lon"))
    nc_data.long_name = "correction factor for direct downward shortwave radiation"
    nc_data.units = "-"
    ncfile.close()
    comp_time_sw_dir_cor = []
    sw_dir_cor_buffer = np.zeros(vec_tilt_enu.shape[:2], dtype=np.float32)
    
    print("Number of timestamps:", len(times))
    
    for i, timestamp in enumerate(tqdm(times, desc="Computing SW dir cor", unit="step")):
        t_beg = time()

        ts = load.timescale()
        t = ts.from_datetime(timestamp)
        
        astrometric = loc_or.at(t).observe(sun)
        alt, az, d = astrometric.apparent().altaz()
        
        x = d.m * np.cos(alt.radians) * np.sin(az.radians)
        y = d.m * np.cos(alt.radians) * np.cos(az.radians)
        z = d.m * np.sin(alt.radians)
        sun_position = np.array([x, y, z], dtype=np.float32)
        
        if i % 20 == 0:
            tqdm.write(f"\n{i}/{len(times)}  {t.utc_strftime('%Y-%m-%d %H:%M:%S')}")
            tqdm.write(f"Solar alt: {alt.degrees:.2f}, az: {az.degrees:.2f}")
            tqdm.write(f"sun position: {np.round(sun_position, 2)}")

        terrain.sw_dir_cor(sun_position, sw_dir_cor_buffer)

        comp_time_sw_dir_cor.append((time() - t_beg))

        ncfile = Dataset(filename=out_file, mode="a")
        nc_time = ncfile.variables["time"]
        nc_time[i] = date2num(timestamp, units=nc_time.units,
                            calendar=nc_time.calendar)
         # Store sun position
        ncfile.variables["sun_x"][i] = x
        ncfile.variables["sun_y"][i] = y
        ncfile.variables["sun_z"][i] = z
        nc_data = ncfile.variables["sw_dir_cor"]
        nc_data[i, :, :] = sw_dir_cor_buffer
        ncfile.close()
        
    # -----------------------------------------------------------------------------
    # Analyse performance of correction factor
    # -----------------------------------------------------------------------------   
    # Performance plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(times, comp_time_sw_dir_cor, lw=1.5, color="red",
            label="SW_dir_cor (mean: %.2f"
                % np.array(comp_time_sw_dir_cor).mean() + ")")
    plt.ylabel("Computing time [seconds]")
    plt.legend(loc="upper center", frameon=False, fontsize=11)
    plt.title("Terrain size (" + str(dim_in_0) + " x " + str(dim_in_1) + ")",
            fontweight="bold", fontsize=12)
    performance_outpath = out_dir + "/Performance.png"
    fig.savefig(performance_outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Performance plot saved to {performance_outpath}.")

    return out_file
    
    
def plot_sw_cor_for_timestep(sw_cor_file, elevation_file, ind=10):
    """Plot elevation the shortwave correction factor for a specific timestep"""

    # Load sw cor data
    ds = xr.open_dataset(sw_cor_file)
    sw_dir_cor = ds["sw_dir_cor"][ind, :, :].values
    timestamp = ds["time"].isel(time=ind).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    sun_x = ds["sun_x"].values[ind]
    sun_y = ds["sun_y"].values[ind]
    sun_z = ds["sun_z"].values[ind]
    ds.close()
    timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[s]").astype(int), tz=timezone.utc)
    print("timestamp date: ", timestamp)
    sun_alt = np.degrees(np.arcsin(sun_z / np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)))
    sun_az  = np.degrees(np.arctan2(sun_x, sun_y))
    print(f"Sun altitude: {np.round(sun_alt,2)}, sun azimuth: {np.round(sun_az,2)}")
    
    zenith, azimuth = get_solar_angle(timestamp)
    print(f"Solar angles from pvlib: alt: {np.round(90-zenith,2)}, azimuth: {np.round(azimuth,2)} ")
    
    # Load elevation data 
    with rasterio.open(elevation_file) as src:
        dsm = src.read(1)  # first band

    elevation_ortho = np.ascontiguousarray(dsm)
    
    # Plot
    ax_lim = (lon.min(), lon.max(),
            lat.min(), lat.max())
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                        hspace=0.05, wspace=0.05, width_ratios=[1.0, 0.027])
    ax = plt.subplot(gs[0, 0])
    ax.set_facecolor(plt.get_cmap("terrain")(0.15)[:3] + (0.25,))
    
    # mask lov elevation (water)
    masked_elev = np.ma.masked_where(elevation_ortho < 0.5, elevation_ortho)
    levels = np.arange(0.0, 800.0, 50.0)
    cmap_terrain = colors.LinearSegmentedColormap.from_list(
        "terrain", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 100))
    )
    cmap = colors.ListedColormap(['blue'] + list(cmap_terrain(np.linspace(0,1,100))))
    norm = colors.BoundaryNorm(np.concatenate(([0, 0.5], levels[1:])), cmap.N, extend='max')
    #data_plot = np.ma.masked_where(mask_ocean[slice_in], elevation_ortho)
    plt.pcolormesh(lon, lat, elevation_ortho,
                cmap=cmap, norm=norm)
    x_ticks = np.arange(np.nanmin(lon), np.nanmax(lon), 0.05)
    plt.xticks(x_ticks, ["" for i in x_ticks])
    y_ticks = np.arange(np.nanmin(lat), np.nanmax(lat), 0.02)
    plt.yticks(y_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
    plt.axis(ax_lim)
    ax = plt.subplot(gs[0, 1])
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    plt.ylabel("Elevation [m a.s.l.]", labelpad=10.0)
    ax = plt.subplot(gs[1, 0])
    vmax = np.nanmax(sw_dir_cor)
    levels = np.arange(0.0, 2.25, 0.25)
    ticks = np.arange(0.0, 2.5, 0.5)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
    plt.pcolormesh(lon, lat, sw_dir_cor,
                cmap=cmap, norm=norm)
    plt.xticks(x_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
    plt.yticks(y_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
    plt.axis(ax_lim)
    txt = timestamp.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    t = plt.text(0.835, 0.935, txt, fontsize=11, fontweight="bold",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    #ts = load.timescale()
    #astrometric = loc_or.at(ts.from_datetime(timestamp).observe(sun)
    #alt, az, d = astrometric.apparent().altaz()
    txt = "Mean solar elevation angle: %.1f" % sun_alt + "$^{\circ}$"
    t = plt.text(0.21, 0.06, txt, fontsize=11, fontweight="bold",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax = plt.subplot(gs[1, 1])
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks,
                            orientation="vertical")
    plt.ylabel("${\downarrow}SW_{dir}$ correction factor [-]", labelpad=10.0)
    ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    outpath = f"output/sw_cor/Elevation_sw_dir_cor_{ts_str}.png"
    fig.savefig(outpath, dpi=300,
                bbox_inches="tight")
    print(f"Saved elevation and sw dir plot to {outpath}.")
    plt.close(fig)


def get_cloud_shadow_displacement(solar_zenith, solar_azimuth,
                         cbh_m, sat_zenith=0.0, sat_azimuth=0.0,
                         pixel_size=10, cloud_top_height=None):
    """
    Calculate cloud shadow displacement onto the surface given cloud,
    solar and satellite geometry.
    
    Parameters
    ----------
    solar_zenith : solar zenith angle in degrees
    solar_azimuth : solar azimuth angle in degrees (0=north, clockwise)
    cbh_m : cloud base height in meters
    sat_zenith : satellite viewing zenith angle in degrees
    sat_azimuth : satellite viewing azimuth angle in degrees (0=north, clockwise)
    pixel_size : resolution of the raster in meters
    cloud_top_height : optional, cloud top height in meters.
                       If not provided, cbh_m is used.
    
    Returns
    -------
    dx_pix : Total displacement in x-direction in pixels
    dy_pix : Total displacement in y_direction in pixels
    """
    assert 0 <= solar_zenith <= 85, (
        f"Invalid solar zenith angle {solar_zenith:.2f}°. "
        "Sun is below the horizon and cannot cast shadows."
    )
    
    # Use cloud top height if available, else fall back to base height
    cth_m = cloud_top_height if cloud_top_height is not None else cbh_m

    # --- Step 1: Correct cloud position for parallax due to satellite view ---
    sat_zenith_rad = np.deg2rad(sat_zenith)
    sat_azimuth_rad = np.deg2rad(sat_azimuth)

    parallax_disp = cth_m * np.tan(sat_zenith_rad)
    dx_sat = parallax_disp * np.sin(sat_azimuth_rad)
    dy_sat = -parallax_disp * np.cos(sat_azimuth_rad)
    print(f"parallax disp: {parallax_disp:.2f}, x: {dx_sat:.2f}, y: {dy_sat:.2f}")

    # --- Step 2: Project shadow using solar geometry from corrected mask ---
    solar_zenith_rad = np.deg2rad(solar_zenith)
    solar_azimuth_rad = np.deg2rad(solar_azimuth)

    horiz_disp = cth_m * np.tan(solar_zenith_rad)
    dx_sun = horiz_disp * np.sin(solar_azimuth_rad)
    dy_sun = -horiz_disp * np.cos(solar_azimuth_rad)
    print(f"horizontal_disp: {horiz_disp:.2f}, x: {dx_sun:.2f}, y: {dy_sun:.2f}")

    dx_total = dx_sat + dx_sun
    dy_total = dy_sat + dy_sun

    # Convert to pixels
    dx_pix = dx_total / pixel_size
    dy_pix = dy_total / pixel_size

    print(f"Total displacement in pixels: x={dx_pix:.2f}, y={dy_pix:.2f}")

    return dx_pix, dy_pix


def project_cloud_shadow(filepath, dy_pix, dx_pix, bbox):
    """
    Shift cloud mask from a GeoTIFF and extract ROI defined by bounding box.
    """
    # Load image + profile
    with rasterio.open(filepath) as src:
        cloud_mask = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    # Shift cloud mask
    shadow_mask = shift(cloud_mask,
                        shift=(-dy_pix, dx_pix),
                        mode="constant", order=0, cval=0)
    shadow_mask = (shadow_mask > 0).astype(np.uint8)
    print(f"Shape of unbounded shadow mask: {shadow_mask.shape}")

    # Convert bbox to raster window
    window = from_bounds(bbox["west"], bbox["south"],
                         bbox["east"], bbox["north"],
                         transform=transform)
    print(f"Window: {window}")

    # Round values
    row_off = math.floor(window.row_off)
    col_off = math.floor(window.col_off)
    height = math.ceil(window.height)
    width = math.ceil(window.width)

    int_window = Window(col_off, row_off, width, height)
    print(f"int window: {int_window}")

    roi_mask = shadow_mask[
        row_off:row_off + height,
        col_off:col_off + width
    ]
    print(f"Shape of roi mask: {roi_mask.shape}")

    return roi_mask


def simulate_annual_ghi():
    """Simulate for a theoretical year in Bergen the annual GHI."""
    pass 


def get_closest_lut_entry(lut, unique_values, doy, hour, albedo, altitude_km,
                          cloud_top_km=None, cot=None, cloud_phase=None):
    """
    Return LUT entry closest to the given parameters.
    
    Parameters
    ----------
    lut : pd.DataFrame
        LUT table with all parameters and irradiance outputs.
    unique_values : dict
        Dictionary of unique values per LUT variable.
    doy, hour, albedo, altitude_km : float
        Mandatory input parameters.
    cloud_top_km, cot, cloud_phase : float, optional
        Cloud parameters; if None, returns clear-sky values only.
    
    Returns
    -------
    dict
        {'Direct_clear': ..., 'Diffuse_clear': ..., 'Direct_cloudy': ..., 'Diffuse_cloudy': ...}
        If cloud parameters are not provided, cloudy values are set to None.
    """
    
    # Step 1: Find closest values in LUT bins
    def closest(value, array):
        array = np.array(array)

        # If value is a string → look for exact match
        if isinstance(value, str):
            matches = array[array == value]
            if len(matches) == 0:
                raise ValueError(f"No exact match found for string '{value}' in array")
            return matches[0]

        # Otherwise → assume numeric → closest value
        else:
            idx = np.argmin(np.abs(array - value))
            return array[idx]
    
    doy_bin = closest(doy, unique_values['DOY'])
    hour_bin = closest(hour, unique_values['Hour'])
    albedo_bin = closest(albedo, unique_values['Albedo'])
    alt_bin = closest(altitude_km, unique_values['Altitude_km'])
    
    # Optional cloud parameters
    cloud_top_bin = closest(cloud_top_km, unique_values['CloudTop_km']) if cloud_top_km is not None else None
    tau_bin = closest(cot, unique_values['Tau550']) if cot is not None else None
    cloud_type_bin = closest(cloud_phase, unique_values['CloudType']) if cloud_phase is not None else None

    print(f"Closest bin found: {doy_bin}, {hour_bin}, {albedo_bin}, {alt_bin}, cth: {cloud_top_bin}, cot: {tau_bin}, cph: {cloud_type_bin}")
    
    # Step 2: Filter LUT by closest bins
    df_filtered = lut[
        (lut['DOY'] == doy_bin) &
        (lut['Hour'] == hour_bin) &
        (lut['Albedo'] == albedo_bin) &
        (lut['Altitude_km'] == alt_bin)
    ]
    
    # If cloud parameters are provided, further filter
    if cloud_top_bin is not None:
        df_filtered = df_filtered[
            (df_filtered['CloudTop_km'] == cloud_top_bin) &
            (df_filtered['Tau550'] == tau_bin) &
            (df_filtered['CloudType'] == cloud_type_bin)
        ]
        
    # Step 3: Return the irradiance values
    if len(df_filtered) == 0:
        # No match found, return None
        return {'Direct_clear': None,
                'Diffuse_clear': None,
                'Direct_cloudy': None,
                'Diffuse_cloudy': None}
    
    # Take the first row (or you could take mean if multiple matches)
    row = df_filtered.iloc[0]
    
    Direct_clear = row['Direct_clear']
    Diffuse_clear = row['Diffuse_clear']
    
    if cloud_top_bin is not None:
        Direct_cloudy = row['Direct_cloudy']
        Diffuse_cloudy = row['Diffuse_cloudy']
    else:
        Direct_cloudy = None
        Diffuse_cloudy = None
    
    res_dict = {'direct_clear': Direct_clear,
            'diffuse_clear': Diffuse_clear,
            'direct_cloudy': Direct_cloudy,
            'diffuse_cloudy': Diffuse_cloudy}
    
    return res_dict

def get_sw_cor_idx(sw_cor_times, timestamp): 
    """Given timestamp, read the observation with the closest timestamp from sw_cor_path nc file."""
    
    # Ensure timestamp is numpy datetime64[ns]
    # Conver to timestamp without timezone, numpy does not like timezone
    if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    timestamp = np.datetime64(timestamp, "ns")
    
    # Find closest time index
    idx = np.argmin(np.abs(sw_cor_times - timestamp))
    closest_time = sw_cor_times[idx]
    print(f"Closest time to {timestamp} is {closest_time}.")    
    
    return idx


def total_ghi_from_sat_imgs(sat_cloud_mask_filepath, cloud_cover_filepath, LUT_filepath,
                            sw_cor_path, 
                            outpath,
                            mixed_threshold, overcast_threshold):
    """For each observation in cloud_cover_filepath, classify into overcast/mixed/clear weather and get 
    the correct LUT entry for the whole map (overcast/clear) or for each pixel (mixed). Output is the total ghi
    over all satellite files per pixel."""
    
    # Read cloud_cover table
    cloud_props = pd.read_csv(cloud_cover_filepath)
    lut = pd.read_csv(LUT_filepath)
    variables = ["DOY", "Hour", "Albedo", "Altitude_km", "CloudTop_km", "Tau550", "CloudType"]

    unique_values = {}
    
    for var in variables:
        if var in lut.columns:
            unique_values[var] = lut[var].unique()
            print(f"{var}: {unique_values[var]}")
    
    # Get times of shortwave direct correction factors      
    ds = xr.open_dataset(sw_cor_path)
    sw_cor_times = ds["time"].values  # dtype=datetime64[ns]
    sw_cor_shape = ds["sw_dir_cor"][0, :, :].values.shape #(2501, 5061)
    print(f"sw_cor_shape : {sw_cor_shape}")

    count_clear = 0
    count_overcast = 0
    count_mixed = 0
    # Iterate over df rows
    for idx, row in cloud_props.iterrows():
        cloud_cover_large = row["cloud_cover_large"]
        # Get SAL, DOY, hour
        surface_albedo = cloud_props["blue_sky_albedo_median"].mean()
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)
        doy = dt.timetuple().tm_yday
        hour = dt.hour + round(dt.minute / 60)
        altitude = 0.08 # in km 

        # Classify into clear, mixed, overcast weather 
        if cloud_cover_large <= mixed_threshold:
            #count_clear += 1
            #### Case 1: Clear Sky Type ####            
            res_dict = get_closest_lut_entry(lut, unique_values, 
                                             doy, hour, surface_albedo, altitude)
            direct_clear = res_dict["direct_clear"]
            diffuse_clear = res_dict["diffuse_clear"]
            total_ghi = direct_clear + diffuse_clear
            #print(f"Total GHI: {total_ghi} ({direct_clear} direct + {diffuse_clear} diffuse).")
            
            # Get shortwave correction factor
            idx = get_sw_cor_idx(sw_cor_times, dt)
            sw_dir_cor = ds["sw_dir_cor"].isel(time=idx).values
            
            # Calculate GHI_total 
            GHI_total = sw_dir_cor * direct_clear + diffuse_clear
            
            if count_clear +1 % 7 == 0: 
                timestamp_str = dt.strftime("%Y%m%dT%H%M%S")

                values = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]

                colors = ["black", "darkblue", "mediumblue", "blueviolet", "purple", "mediumvioletred",
                          "crimson", "deeppink", "salmon", "orangered", "darkorange", "orange", "yellow"]

                plot_single_band(GHI_total, f"output/ghi_total_{timestamp_str}.png", 
                        f"GHI Total for {timestamp_str}",
                        "GHI (W/m²)", values, colors)

            # Save to outpath
    
        elif cloud_cover_large >= overcast_threshold:
            count_overcast += 1
            #### Case 2 : Overcast Sky Type ####
            # Get COT, CTH, CPH
            COT = 3
            CTH = 2.0 # in km
            CPH = "water"
            res_dict = get_closest_lut_entry(lut, unique_values, 
                                             doy, hour, surface_albedo, altitude,
                                             CTH, COT, CPH)
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            total_ghi = direct_cloudy + diffuse_cloudy
            print(f"Total GHI: {total_ghi} ({np.round(direct_cloudy,2)} direct + {np.round(diffuse_cloudy,2)} diffuse).")
            
            # Get shortwave correction factor
            idx = get_sw_cor_idx(sw_cor_times, dt)
            sw_dir_cor = ds["sw_dir_cor"].isel(time=idx).values
 
            # Calculate GHI_total 
            GHI_total = sw_dir_cor * direct_cloudy + diffuse_cloudy
            
            if count_overcast % 12 == 0: 
                timestamp_str = dt.strftime("%Y%m%dT%H%M%S")

                values = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]

                colors = ["black", "darkblue", "mediumblue", "blueviolet", "purple", "mediumvioletred",
                          "crimson", "deeppink", "salmon", "orangered", "darkorange", "orange", "yellow"]

                plot_single_band(GHI_total, f"output/ghi_total_{timestamp_str}.png", 
                        f"GHI Total for {timestamp_str}",
                        "GHI (W/m²)", values, colors)


        elif mixed_threshold < cloud_cover_large < overcast_threshold:
            count_mixed += 1
            #### Case 3: Mixed #### 
            
            # Perform shadow calculation 
            
            # Get closest LUT 
            
            # Get shortwave correction factor 
            
            # Get GHI and save 
            
            # Return outpath
        else : 
            print(f"Model not implemented for cloud_cover_large {cloud_cover_large}.")
    ds.close()

    pass 


def cloud_mask_to_ghi(sat_cloud_mask_img, GHI_direct_clear, GHI_direct_cloudy, GHI_diffuse_clear, GHI_diffuse_cloudy):
    """Given a satellite cloud mask image, get the GHI at each pixel."""
    pass

def analyze_cloud_shadow_displacement(cloud_cover_table_path, cth): 
    """Get cloud shadow displacement in x and y direction for each observation in cloud_cover table.
    Plot the distribution of displacements."""
    cloud_cover = pd.read_csv(cloud_cover_table_path)
    
    displacement_x = []
    displacement_y = []
    
    for idx, row in cloud_cover.iterrows():
        sat_zenith = row["MEAN_ZENITH"]
        sat_azimuth = row["MEAN_AZIMUTH"]
        cth_small = row["cth_median_small"]
        cth_large = row["cth_median_large"]
        if cth_small is not None: 
            cth = cth_small
        elif cth_large is not None: 
            cth = cth_large
        
        # Get date 
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)
        solar_zenith, solar_azimuth = get_solar_angle(dt)
        
        dx_pix, dy_pix = get_cloud_shadow_displacement(solar_zenith, solar_azimuth, 0, 
                                       sat_zenith, sat_azimuth, pixel_size = 10, cloud_top_height=cth)
        
        dx = dx_pix * 10 
        dy = dy_pix * 10 
        
        #print(f"Displacement for sol_zen {solar_zenith:.1f}, sol_azi {solar_azimuth:.1f}, " \
        #    f"sat_zen {np.round(sat_zenith,1)}, sat_azi {np.round(sat_azimuth,1)} (Time UTC: {dt}) : " \
        #        f"\ndx = {np.round(dx)}, dy = {np.round(dy)}")
        
        displacement_x.append(dx)
        displacement_y.append(dy)
    
    # Plot hist of displacement x and y 
    # Convert to arrays
    displacement_x = np.array(displacement_x)
    displacement_y = np.array(displacement_y)
    
    # Remove NaNs
    displacement_x = displacement_x[~np.isnan(displacement_x)]
    displacement_y = displacement_y[~np.isnan(displacement_y)]
    
    # Compute percentiles safely
    x_percentiles = np.nanpercentile(displacement_x, [25, 50, 75]) if displacement_x.size > 0 else [np.nan]*3
    y_percentiles = np.nanpercentile(displacement_y, [25, 50, 75]) if displacement_y.size > 0 else [np.nan]*3
    
    # Plot histograms
    plt.figure(figsize=(10,6))
    bins = 30  
    
    plt.hist(displacement_x, bins=bins, alpha=0.5, color="tab:blue", label="dx")
    plt.hist(displacement_y, bins=bins, alpha=0.5, color="tab:orange", label="dy")
    
    # Plot vertical lines for percentiles
    for p, val in zip(["25%", "50%", "75%"], x_percentiles):
        plt.axvline(val, color="tab:blue", linestyle="--", alpha=0.7)
    for p, val in zip(["25%", "50%", "75%"], y_percentiles):
        plt.axvline(val, color="tab:orange", linestyle="--", alpha=0.7)
    
    # Legend text with percentiles
    legend_text = [
        f"dx: p25={x_percentiles[0]:.1f}, p50={x_percentiles[1]:.1f}, p75={x_percentiles[2]:.1f}",
        f"dy: p25={y_percentiles[0]:.1f}, p50={y_percentiles[1]:.1f}, p75={y_percentiles[2]:.1f}"
    ]
    
    plt.legend(title="\n".join(legend_text))
    plt.xlabel("Displacement [m]")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Cloud Shadow Displacement (dx, dy)")
    plt.grid(alpha=0.3)
    outpath = f"output/cloud_shadow_displacement_hist.png"
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}.")



if __name__ == "__main__": 
    # Paths
    cloud_cover_table_filepath = "data/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    DSM_filepath = "data/bergen_dsm_10m_epsg4326.tif"
    sw_cor_filepath = "output/sw_cor/sw_cor_bergen.nc"
    s2_cloud_mask_folderpath = "data/S2_cloud_mask_large"
    lut_filepath = "output/LUT/LUT.csv"
    
    analyze_cloud_shadow_displacement(cloud_cover_table_filepath, 2000)
    
    """lut = pd.read_csv(lut_filepath)

    total_ghi_from_sat_imgs(s2_cloud_mask_folderpath, cloud_cover_table_filepath, lut_filepath,
                            sw_cor_path=sw_cor_filepath,
                            outpath="output/ghi_maps/", mixed_threshold=1, overcast_threshold=99)

    # Cloud shadow computation
    # Cloud base height (e.g., 2000 m)
    s2_cloud_mask_sample = "data/S2_cloud_mask_large/S2_cloud_mask_large_2017-08-25_11-06-49-2017-08-25_11-06-49.tif"
    #s2_img, profile = load_image(s2_cloud_mask_sample)
    timestamp = datetime(2017,8,25,11, 6, 49)
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%S")
    solar_zenith, solar_azimuth = get_solar_angle(timestamp)
    print(f"Solar zenith angle: {solar_zenith} and azimuth: {solar_azimuth}")
    cbh_m = 1000 
    sat_zenith = 5
    sat_azimuth = 150.5

    start = time()
    dx_pix, dy_pix = get_cloud_shadow_displacement(solar_zenith, solar_azimuth, cbh_m, 
                                       sat_zenith, sat_azimuth, pixel_size = 10, cloud_top_height=cbh_m+1000)
    shadow_mask = project_cloud_shadow(s2_cloud_mask_sample, dy_pix, dx_pix, BBOX)
    print(f"Shadow computation takes {np.round(time()-start,3)} secs.")
    plot_single_band(shadow_mask, f"output/shadow_mask_{timestamp_str}_sza_{solar_zenith:.2f}_azi_{solar_azimuth:.2f}.png", 
                     f"Shadow Mask for {timestamp_str} (SZA {solar_zenith:.2f}, AZI {solar_azimuth:.2f})",
                     "Shadow", [0, 1], ["white", "darkgrey"])"""
