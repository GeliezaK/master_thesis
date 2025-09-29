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
from shapely.geometry import box
import glob
import horayzon as hray
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from time import time
from skyfield.api import load, wgs84
from rasterio.plot import show
from src.plotting.high_res_maps import plot_landmarks, plot_single_band, plot_single_band_histogram, plot_single_tif

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
    
def load_image(filepath): 
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(float)
        profile = src.profile
    return dem, profile


def calculate_sw_dir_cor(
    dsm_filepath,
    times, 
    out_dir="data/processed/sw_cor"
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
    

    # Ensure correct dtypes for Horayzon
    if lon.dtype != np.float32:
        lon = lon.astype(np.float32)
    if lat.dtype != np.float32:
        lat = lat.astype(np.float32)
    if dsm.dtype != np.float32:
        dsm = dsm.astype(np.float32)

    print(f"Datatypes after conversion: lon {lon.dtype}, lat {lat.dtype}, dsm {dsm.dtype}")

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
    

def simulate_annual_ghi():
    """Simulate for a theoretical year in Bergen the annual GHI."""
    pass 


def get_closest_lut_entry(lut, unique_values, doy, hour, albedo, altitude_km,
                          cloud_top_km=None, cot=None, cloud_phase=None, verbose=False):
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
        {'direct_clear': ..., 'diffuse_clear': ..., 'direct_cloudy': ..., 'diffuse_cloudy': ...}
        If cloud parameters are not provided, cloudy values are set to None.
    """
    # Assert input params are correct
    if cloud_top_km is not None:
        assert cloud_top_km <= 50, f"Cloud Top Height is larger than 50 km ({cloud_top_km})! Did you convert m to km?"
        
    if cloud_phase is not None:
        assert isinstance(cloud_phase, str), f"Cloud phase should be a string!"
        assert (cloud_phase == "water") or (cloud_phase == "ice"), f"Cloud phase should be either 'water', 'ice' or None! Current value: {cloud_phase}."
    
    if cot is not None: 
        assert 0.0 <= cot <= 300, f"COT is expected to be within range 0-300 (unitless). Current value: {cot}."
        
    assert 1 <= doy <= 366, f"Doy {doy} is out of range! Expected values between 1-366."
    assert 0 <= hour <= 24, f"Hour of day {hour} is out of range! Expected values between 0 and 24. Remember hour is UCT. "
    assert 0.0 <= albedo <= 1.0, f"Albedo {albedo} is out of range! Values should be between 0 and 1."
    assert 0.0 <= altitude_km <= 5.0, f"Altitude {altitude_km} is out of range! Expected values between 0 and 5.0. Did you convert m to km? "
    
    # Find closest values in LUT bins
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
    
    if verbose: 
        print(f"Closest LUT bin found: {doy_bin}, {hour_bin}, {albedo_bin}, {alt_bin}, cth: {cloud_top_bin:.3f}, cot: {tau_bin}, cph: {cloud_type_bin}")

    # Filter LUT by closest bins
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
        
    # Return the irradiance values
    if len(df_filtered) == 0:
        # No match found, return None
        return {'direct_clear': None,
                'diffuse_clear': None,
                'direct_cloudy': None,
                'diffuse_cloudy': None}
    
    # Take the first row (or you could take mean if multiple matches)
    row = df_filtered.iloc[0]
    
    direct_clear = row['Direct_clear']
    diffuse_clear = row['Diffuse_clear']
    
    if cloud_top_bin is not None:
        direct_cloudy = row['Direct_cloudy']
        diffuse_cloudy = row['Diffuse_cloudy']
    else:
        direct_cloudy = None
        diffuse_cloudy = None
    
    res_dict = {'direct_clear': direct_clear,
            'diffuse_clear': diffuse_clear,
            'direct_cloudy': direct_cloudy,
            'diffuse_cloudy': diffuse_cloudy}
    
    return res_dict


def get_sw_cor_idx(sw_cor_times, timestamp, verbose = False): 
    """Given timestamp, read the observation with the closest timestamp from sw_cor_path nc file."""
    
    # Ensure timestamp is numpy datetime64[ns]
    # Conver to timestamp without timezone, numpy does not like timezone
    if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    timestamp = np.datetime64(timestamp, "ns")
    
    # Find closest time index
    idx = np.argmin(np.abs(sw_cor_times - timestamp))
    closest_time = sw_cor_times[idx]
    if verbose:
        print(f"Closest time to {timestamp} is {closest_time}.")    
    
    return idx


def get_cloud_properties(row, monthly_medians, month, verbose=False):
    """Return COT, CTH (in km), and CPH (water/ice) for a given row in pandas dataframe."""
    
    # --- COT ---
    if not pd.isna(row["cot_median_small"]):
        COT = row["cot_median_small"]
    elif not pd.isna(row["cot_median_large"]):
        COT = row["cot_median_large"]
    else:
        COT = monthly_medians.loc[monthly_medians['month'] == month, "cot_median_small"].values[0]
        
    # --- CTH ---
    if not pd.isna(row["cth_median_small"]):
        CTH = row["cth_median_small"]
    elif not pd.isna(row["cth_median_large"]):
        CTH = row["cth_median_large"]
    else:
        CTH = monthly_medians.loc[monthly_medians['month'] == month, "cth_median_small"].values[0]
    
    CTH = CTH / 1000.0  # convert to km

    # --- CPH ---
    if not pd.isna(row["cph_median_small"]):
        CPH = "water" if row["cph_median_small"] <= 1.5 else "ice"
    elif not pd.isna(row["cph_median_large"]):
        CPH = "water" if row["cph_median_large"] <= 1.5 else "ice"
    else:
        monthly_cph = monthly_medians.loc[monthly_medians['month'] == month, "cph_median_small"].values[0]
        CPH = "water" if monthly_cph <= 1.5 else "ice"

    if verbose:
        print(f"cloud props: COT {COT:.2f}, CTH {CTH:.2f}, CPH {CPH}")
    return COT, CTH, CPH

def nc_to_raster_meta(ds):
    lat = ds["lat"].values
    lon = ds["lon"].values
    var = ds["sw_dir_cor"].isel(time=0).values
    
    nrows, ncols = var.shape[-2], var.shape[-1]

    # Assuming regular lat/lon grid
    res_lon = (lon[-1] - lon[0]) / (ncols - 1)
    res_lat = (lat[-1] - lat[0]) / (nrows - 1)

    # Top-left origin transform
    transform = from_origin(lon.min(), lat.max(), res_lon, abs(res_lat))

    return transform, (nrows, ncols), (lon.min(), lat.max(), lon.max(), lat.min())


def reproject_shadow_to_nc(shadow_mask, shadow_transform, nc_transform, nc_shape):
    dst = np.zeros(nc_shape, dtype=np.uint8)

    reproject(
        source=shadow_mask,
        destination=dst,
        src_transform=shadow_transform,
        src_crs="EPSG:4326",       # or the CRS of your GeoTIFF
        dst_transform=nc_transform,
        dst_crs="EPSG:4326",       # or CRS of your NetCDF
        resampling=Resampling.nearest
    )
    return dst 


def total_ghi_from_sat_imgs(sat_cloud_mask_filepath, cloud_cover_filepath, LUT_filepath,
                               sw_cor_path, out_nc_file,
                               mixed_threshold, overcast_threshold, verbose=False):
    """Compute total GHI from satellite images and save directly to NetCDF per iteration."""

    # -----------------------------------------------------------------------------
    # Initialize
    # -----------------------------------------------------------------------------
    cloud_props = pd.read_csv(cloud_cover_filepath)
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    cloud_props["month"] = cloud_props["date"].dt.month

    # Default monthly medians
    monthly_medians = (
        cloud_props.groupby("month")[["cot_median_small","cth_median_small","cph_median_small"]]
        .median().reset_index()
    )

    # Surface albedo
    surface_albedo = cloud_props["blue_sky_albedo_median"].mean()
    median_sat_zen = cloud_props["MEAN_ZENITH"].median()
    median_sat_azi = cloud_props["MEAN_AZIMUTH"].median()

    # Read LUT and unique values
    lut = pd.read_csv(LUT_filepath)
    variables = ["DOY", "Hour", "Albedo", "Altitude_km", "CloudTop_km", "Tau550", "CloudType"]
    unique_values = {var: lut[var].unique() for var in variables if var in lut.columns}

    # SW correction factors
    ds_sw = xr.open_dataset(sw_cor_path)
    sw_cor_times = ds_sw["time"].values
    lat = ds_sw["lat"].values
    lon = ds_sw["lon"].values
    shape_lat = len(lat)
    shape_lon = len(lon)

    # -----------------------------------------------------------------------------
    # Create NetCDF file if it does not exist
    # -----------------------------------------------------------------------------
    if not os.path.exists(out_nc_file):
        os.makedirs(os.path.dirname(out_nc_file), exist_ok=True)
        ncfile = Dataset(out_nc_file, mode="w")
        ncfile.createDimension("time", size=None)
        ncfile.createDimension("lat", size=shape_lat)
        ncfile.createDimension("lon", size=shape_lon)

        # Time variable
        nc_time = ncfile.createVariable("time", "f8", ("time",))
        nc_time.units = "hours since 2015-01-01 00:00:00"
        nc_time.calendar = "gregorian"

        # Lat/Lon
        nc_lat = ncfile.createVariable("lat", "f4", ("lat",))
        nc_lat[:] = lat
        nc_lat.units = "degree"
        nc_lon = ncfile.createVariable("lon", "f4", ("lon",))
        nc_lon[:] = lon
        nc_lon.units = "degree"

        # GHI variable
        nc_ghi = ncfile.createVariable("GHI_total", "f4", ("time", "lat", "lon"), zlib=True)
        nc_ghi.units = "W/m2"
        nc_ghi.long_name = "Total surface GHI including cloud effects"

        ncfile.close()

    # -----------------------------------------------------------------------------
    # Open NetCDF in append mode and get existing times
    # -----------------------------------------------------------------------------
    ncfile = Dataset(out_nc_file, mode="a")
    existing_times = ncfile.variables["time"][:]
    # Convert existing numeric times to datetime (tz-naive)
    existing_datetimes = pd.to_datetime(existing_times, origin=pd.Timestamp("2015-01-01 00:00:00"), unit='h')
    existing_dates = existing_datetimes.date  # numpy array of datetime.date
    ncfile.close()
    
    
    print(f"Number of observations saved in {out_nc_file}: {len(existing_times)}")
    
    # -----------------------------------------------------------------------------
    # Iterate and write per timestep
    # -----------------------------------------------------------------------------
    for idx, row in tqdm(cloud_props.iterrows(), total=len(cloud_props), desc="Processing rows"):
        dt = pd.to_datetime(row['system:time_start_large'], unit='ms', utc=True)
        
        # Skip if date already in NetCDF
        # Make dt tz-naive for comparison with existing_datetimes
        dt_naive = dt.tz_convert(None) if hasattr(dt, 'tz') else dt
        obs_date = dt_naive.date()  

        # Check if the date already exists
        if obs_date in existing_dates:
            if verbose:
                tqdm.write(f"Skipping {dt}, date {obs_date} already exists in NetCDF.")
            continue
        
        month = row["month"]
        doy = dt.timetuple().tm_yday
        hour = dt.hour + round(dt.minute/60)
        altitude = 0.08  # km

        cloud_cover_large = row["cloud_cover_large"]

        # Initialize GHI_total
        GHI_total = None

        # ---------------- Clear Sky ----------------
        if cloud_cover_large <= mixed_threshold:
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude)
            direct_clear = res_dict["direct_clear"]
            diffuse_clear = res_dict["diffuse_clear"]
            if direct_clear is None or diffuse_clear is None:
                tqdm.write(f"Skip {dt}: no data returned. Direct clear: {direct_clear}, Diffuse clear: {diffuse_clear}.")
                tqdm.write(f"Inputs to LUT: DOY {doy}, hour {hour}.")
                continue
            idx_sw = get_sw_cor_idx(sw_cor_times, dt)
            sw_dir_cor = ds_sw["sw_dir_cor"].isel(time=idx_sw).values
            GHI_total = sw_dir_cor * direct_clear + diffuse_clear

        # ---------------- Overcast ----------------
        elif cloud_cover_large >= overcast_threshold:
            COT, CTH, CPH = get_cloud_properties(row, monthly_medians,month, verbose=verbose)
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude,
                                             CTH, COT, CPH, verbose=verbose)
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            if direct_cloudy is None or diffuse_cloudy is None:
                tqdm.write(f"Skip {dt}: no data returned. Direct cloudy: {direct_cloudy}, Diffuse cloudy: {diffuse_cloudy}.")
                tqdm.write(f"Inputs to LUT: DOY {doy}, hour {hour}, CTH {CTH}, COT {COT}, CPH {CPH}.")
                continue
            idx_sw = get_sw_cor_idx(sw_cor_times, dt)
            sw_dir_cor = ds_sw["sw_dir_cor"].isel(time=idx_sw).values
            
            GHI_total = sw_dir_cor * direct_cloudy + diffuse_cloudy

        # ---------------- Mixed ----------------
        else:
            # Mixed: compute per-pixel
            date = dt.strftime("%Y-%m-%d")
            pattern = os.path.join(sat_cloud_mask_filepath, f"S2_cloud_mask_large_{date}_*.tif")
            files = glob.glob(pattern)
            if len(files) != 1:
                tqdm.write(f"Skip {dt} because no files were found for this pattern: {pattern}.")
                continue
            cloud_mask_path = files[0]

            COT, CTH, CPH = get_cloud_properties(row, monthly_medians, month, verbose=verbose)
            solar_zenith, solar_azimuth = get_solar_angle(dt)
            if solar_zenith > 85:
                tqdm.write(f"Skip {dt} because SZA > 85 (SZA={solar_zenith:.2f}).")
                continue
            
            sat_zenith, sat_azimuth = row["MEAN_ZENITH"], row["MEAN_AZIMUTH"]
            # Replace missing satellite angles with alltime median
            if pd.isna(sat_zenith) or pd.isna(sat_azimuth):
                sat_zenith, sat_azimuth = median_sat_zen, median_sat_azi
            # Compute cloud shadow
            cth_m = CTH * 1000.0
            cbh_m = cth_m - 1000.0
            dx_pix, dy_pix = get_cloud_shadow_displacement(
                solar_zenith, solar_azimuth, cbh_m,
                sat_zenith, sat_azimuth, pixel_size=10, cloud_top_height=cth_m
            )
            shadow_mask, _ = project_cloud_shadow(cloud_mask_path, dy_pix, dx_pix, BBOX)
            # TODO: this is just a quick fix - make sure the pixels align location-wise
            # Pad cloud shadow to match sw_dir_cor shape
            idx_sw = get_sw_cor_idx(sw_cor_times, dt)
            sw_dir_cor = ds_sw["sw_dir_cor"].isel(time=idx_sw).values
            nrows, ncols = shadow_mask.shape
            target_rows, target_cols = sw_dir_cor.shape
            pad_rows = target_rows - nrows
            pad_cols = target_cols - ncols
            shadow_mask = np.pad(shadow_mask, ((0,pad_rows),(0,pad_cols)),
                                 mode="constant", constant_values=False)
            # get LUT entries
            res_dict = get_closest_lut_entry(lut, unique_values, doy, hour,
                                             surface_albedo, altitude,
                                             CTH, COT, CPH, verbose=verbose)
            
            direct_clear = res_dict["direct_clear"]
            diffuse_clear = res_dict["diffuse_clear"]
            direct_cloudy = res_dict["direct_cloudy"]
            diffuse_cloudy = res_dict["diffuse_cloudy"]
            
            if (direct_clear is None or diffuse_clear is None or
                direct_cloudy is None or diffuse_cloudy is None):
                tqdm.write(f"Skip {dt}: no data returned. Direct clear: {direct_clear}," \
                           f"Diffuse clear: {diffuse_clear}, Direct cloudy: {direct_cloudy}, " \
                           f"Diffuse cloudy: {diffuse_cloudy}.")
                tqdm.write(f"Inputs to LUT: DOY {doy}, hour {hour}, CTH {CTH}, COT {COT}, CPH {CPH}.")
                continue
            
            # Compute Ghi total
            GHI_total = np.where(shadow_mask, sw_dir_cor*direct_cloudy + diffuse_cloudy,
                                 sw_dir_cor*direct_clear + diffuse_clear)
        
        if GHI_total is None:
            print(f"Skipped at {dt} because GHI_total is None. res_dict: {res_dict}")

        # Write to NetCDF and close after each iteration
        if GHI_total is not None and GHI_total.shape == (shape_lat, shape_lon):
            ncfile = Dataset(out_nc_file, mode="a")
            t_var = ncfile.variables["time"]
            i = len(t_var)  # next index
            t_var[i] = date2num(dt.to_pydatetime(), units=t_var.units, calendar=t_var.calendar)
            ncfile.variables["GHI_total"][i,:,:] = GHI_total
            ncfile.close()

    ds_sw.close()
    print(f"GHI maps saved incrementally to {out_nc_file}")
   

if __name__ == "__main__": 
    # Paths
    cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    DSM_filepath = "data/processed/bergen_dsm_10m_epsg4326.tif"
    sw_cor_filepath = "data/processed/sw_cor/sw_cor_bergen.nc"
    s2_cloud_mask_folderpath = "data/raw/S2_cloud_mask_large"
    lut_filepath = "data/processed/LUT/LUT.csv"
    cloud_props = pd.read_csv(cloud_cover_table_filepath) 

        
    total_ghi_from_sat_imgs(s2_cloud_mask_folderpath, cloud_cover_table_filepath, lut_filepath,
                            sw_cor_path=sw_cor_filepath, out_nc_file="data/processed/simulated_ghi.nc",
                            mixed_threshold=1, overcast_threshold=99, verbose=False)

    