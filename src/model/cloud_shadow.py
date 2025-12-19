import pvlib # for solar position
import math
import numpy as np 
import os 
import glob 
import rasterio 
from rasterio.windows import from_bounds
from tqdm import tqdm 
from netCDF4 import Dataset, date2num
import pandas as pd
from src.model.instantaneous_GHI_model import get_cloud_properties
from src.model import CENTER_LAT, CENTER_LON, BBOX, MIXED_THRESHOLD, OVERCAST_THRESHOLD, COARSE_RESOLUTIONS

# Radius of Earth
R_EARTH = 6371000


def get_solar_angle(datetime, lat=CENTER_LAT, lon=CENTER_LON): 
    """Get solar azimuth and zenith angles from timestamp datetime and location lat/lon."""
    solpos = pvlib.solarposition.get_solarposition(datetime, lat, lon)
    zenith = float(solpos['zenith'].values[0])
    azimuth = float(solpos['azimuth'].values[0])
    return zenith, azimuth


def get_cloud_shadow_displacement(solar_zenith, solar_azimuth,
                         cbh_m, sat_zenith=0.0, sat_azimuth=0.0,
                         cloud_top_height=None, verbose=False):
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
    cloud_top_height : optional, cloud top height in meters.
                       If not provided, cbh_m is used.
    verbose : print logging messages if True. Default: False. 
    
    Returns
    -------
    dx_total : Total displacement in x-direction in meters
    dy_total : Total displacement in y_direction in meters
    """
    assert 0 <= solar_zenith <= 85, (
        f"Invalid solar zenith angle {solar_zenith:.2f}°. "
        "Sun is below the horizon and cannot cast shadows."
    )
    assert 0 <= solar_azimuth <= 360, f"Invalid solar azimuth angle. Must be between 0 and 360 degrees. Current value {solar_azimuth}."
    assert 0 <= sat_zenith <= 90, f"Invalid satellite zenith viewing angle. Must be between 0 and 90 degrees. Current value {sat_zenith}."
    assert 0 <= sat_azimuth <= 360, f"Invalid satellite azimuth viewing angle. Must be between 0 and 360 degrees. Current value {sat_azimuth}."
    
    if cloud_top_height is not None: 
        assert cbh_m < cloud_top_height, f"Cloud top height should be greater than or equal cloud base height! (CTH {cloud_top_height} < CBH {cbh_m})"
        assert cloud_top_height > 10.0, f"Cloud top height should be greater than 10m. Did you convert km to m? Current value: {cth_m}."
    else :
        assert cbh_m > 0, f"If Cloud top height is None, Cloud base height must be given and > 0. Current value: {cbh_m}."
        if cbh_m <= 10.0: 
            print(f"Warning: Cloud base height must be given in meters. Current value is {cbh_m}. Did you convert km to m?")
    
    # Use cloud top height if available, else fall back to base height
    cth_m = cloud_top_height if cloud_top_height is not None else cbh_m

    # --- Step 1: Correct cloud position for parallax due to satellite view ---
    sat_zenith_rad = np.deg2rad(sat_zenith)
    sat_azimuth_rad = np.deg2rad(sat_azimuth)

    parallax_disp = cth_m * np.tan(sat_zenith_rad)
    dx_sat = parallax_disp * np.sin(sat_azimuth_rad)
    dy_sat = -parallax_disp * np.cos(sat_azimuth_rad)

    # --- Step 2: Project shadow using solar geometry from corrected mask ---
    solar_zenith_rad = np.deg2rad(solar_zenith)
    solar_azimuth_rad = np.deg2rad(solar_azimuth)

    horiz_disp = cth_m * np.tan(solar_zenith_rad)
    dx_sun = horiz_disp * np.sin(solar_azimuth_rad)
    dy_sun = -horiz_disp * np.cos(solar_azimuth_rad)

    dx_total = dx_sat + dx_sun
    dy_total = dy_sat + dy_sun

    if verbose:
        print(f"Total displacement in meters: x={dx_total:.2f}, y={dy_total:.2f}")

    return dx_total, dy_total


def meters_to_latlon_offset(dx_m, dy_m, lat_center):
    """
    Convert displacement in meters to lat/lon offset in degrees.
    """
    dlat = dy_m / R_EARTH * (180/np.pi)
    dlon = dx_m / (R_EARTH * math.cos(lat_center)) * (180/np.pi)
    return dlat, dlon


def read_shadow_roi(cloud_mask_filepath, dx_m, dy_m):
    """
    Cut out the cloud shadow from the large cloud mask. 
    Resulting cloud shadow mask has the size of the study area (region of interest (ROI)), 
    but is shifted in x and y direction according to the shadow projection.
    
    Parameters
    ----------
    cloud_mask_filepath : path to cloud mask raster
    dx_m, dy_m : shadow displacement in meters
    """
    
    # Compute center latitude of ROI for accurate lon scaling
    dlat, dlon = meters_to_latlon_offset(dx_m, dy_m, CENTER_LAT)

    # New bounding box of shadowed area
    shadow_bbox = {
        "north": BBOX["north"] - dlat,
        "south": BBOX["south"] - dlat,
        "east":  BBOX["east"] - dlon,
        "west":  BBOX["west"] - dlon
    }

    with rasterio.open(cloud_mask_filepath) as src:
        window = from_bounds(
            shadow_bbox["west"], shadow_bbox["south"],
            shadow_bbox["east"], shadow_bbox["north"],
            transform=src.transform
        )
        shadow_mask = src.read(1, window=window).astype(np.uint8)

    return shadow_mask, shadow_bbox

    
def save_shadow_masks_to_nc(cloud_props_filepath, sat_cloud_mask_dir, out_nc_file, verbose=False):
    """
    Cloud shadow computation workflow:
    Loop over selected rows in cloud_props_filepath, compute cloud shadow displacement,
    extract shadow masks using read_shadow_roi, and save to NetCDF incrementally.

    Filtering: only rows with
        mixed_threshold < cloud_cover_large < overcast_threshold

    """

    # -------------------------------------------------------------------------
    # Load cloud properties and filter
    # -------------------------------------------------------------------------
    cloud_props = pd.read_csv(cloud_props_filepath)
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    cloud_props["month"] = cloud_props["date"].dt.month
    

    cloud_props = cloud_props[
        (cloud_props["cloud_cover_large"] > MIXED_THRESHOLD) &
        (cloud_props["cloud_cover_large"] < OVERCAST_THRESHOLD)
    ].reset_index(drop=True)
    
    median_sat_zen = cloud_props["MEAN_ZENITH"].median()
    median_sat_azi = cloud_props["MEAN_AZIMUTH"].median()
    
    monthly_medians = (
        cloud_props.groupby("month")[["cot_median_small","cth_median_small","cph_median_small"]]
        .median().reset_index()
    )

    if verbose:
        print(f"Filtered rows: {len(cloud_props)} "
              f"(thresholds {MIXED_THRESHOLD}–{OVERCAST_THRESHOLD})")

    # -------------------------------------------------------------------------
    # Open NetCDF in append mode and get existing times
    # -------------------------------------------------------------------------
    if os.path.exists(out_nc_file):
        ncfile = Dataset(out_nc_file, mode="a")
        existing_times = ncfile.variables["time"][:]
        existing_datetimes = pd.to_datetime(existing_times,
                                            origin=pd.Timestamp("2015-01-01 00:00:00"),
                                            unit="h")
        existing_dates = existing_datetimes.date
        ncfile.close()
    else :
        existing_dates = []

    if verbose:
        print(f"Number of shadow masks already in {out_nc_file}: {len(existing_dates)}")

    # -------------------------------------------------------------------------
    # Iterate rows and compute shadow masks
    # -------------------------------------------------------------------------
    for idx, row in tqdm(cloud_props.iterrows(), total=len(cloud_props),
                         desc="Processing cloud_props"):

        dt = pd.to_datetime(row["system:time_start_large"], unit="ms", utc=True)
        obs_date = dt.date()
        month = row["month"]

        # Skip if already in file
        if obs_date in existing_dates:
            if verbose:
                tqdm.write(f"Skipping {dt} (already in NetCDF).")
            continue

        # Find matching S2 cloud mask file
        date = dt.strftime("%Y-%m-%d")
        pattern = os.path.join(sat_cloud_mask_dir, f"S2_cloud_mask_large_{date}*.tif")
        files = glob.glob(pattern)
        if len(files) != 1:
            tqdm.write(f"Skip {dt} because no files were found for pattern: {pattern}.")
            continue
        cloud_mask_path = files[0]

        # Compute geometry
        solar_zenith, solar_azimuth = get_solar_angle(dt)
        sat_zenith, sat_azimuth = row["MEAN_ZENITH"], row["MEAN_AZIMUTH"]
        # Replace missing satellite angles with alltime median
        if pd.isna(sat_zenith) or pd.isna(sat_azimuth):
            sat_zenith, sat_azimuth = median_sat_zen, median_sat_azi


        # Assume CBH = CTH - 1000m - technically not needed because CTH is sufficient
        COT, CTH, CPH = get_cloud_properties(row, monthly_medians, month)
        cth_m = CTH * 1000
        cbh_m = cth_m - 1000.0
        
        if verbose:
            tqdm.write(f"Cloud base height: {cbh_m}, top height: {cth_m}")

        dx_m, dy_m = get_cloud_shadow_displacement(
            solar_zenith, solar_azimuth, cbh_m,
            sat_zenith, sat_azimuth,
            cloud_top_height=cth_m, verbose=verbose
        )

        # Extract shifted shadow mask
        shadow_mask, shadow_bbox = read_shadow_roi(cloud_mask_path, dx_m, dy_m)
        
        if verbose: 
            tqdm.write(f"Shadow mask shape: {shadow_mask.shape}")
            
        # -------------------------------------------------------------------------
        # Create NetCDF file if it does not exist - only first iteration
        # -------------------------------------------------------------------------
        if not os.path.exists(out_nc_file):
            os.makedirs(os.path.dirname(out_nc_file), exist_ok=True)
            nlat, nlon = shadow_mask.shape
            tqdm.write(f"nlat, nlon: {(nlat, nlon)}")
            lon = np.linspace(BBOX["west"], BBOX["east"], nlon)
            lat = np.linspace(BBOX["south"], BBOX["north"], nlat)
            
            ncfile = Dataset(out_nc_file, mode="w")
            ncfile.createDimension("time", size=None)
            ncfile.createDimension("lat", size=nlat)
            ncfile.createDimension("lon", size=nlon)

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

            # Shadow mask variable
            nc_shadow = ncfile.createVariable("shadow_mask", "i1",
                                            ("time", "lat", "lon"),
                                            zlib=True)
            nc_shadow.units = "1"
            nc_shadow.long_name = "Cloud shadow mask (0=clear, 1=shadow)"

            ncfile.close()

        # Save to NetCDF
        ncfile = Dataset(out_nc_file, mode="a")
        t_var = ncfile.variables["time"]
        i = len(t_var)
        t_var[i] = date2num(dt.to_pydatetime(),
                            units=t_var.units,
                            calendar=t_var.calendar)
        ncfile.variables["shadow_mask"][i, :, :] = shadow_mask
        ncfile.close()

    print(f"Shadow masks saved incrementally to {out_nc_file}")



if __name__=="__main__":
    # Cloud shadow computation
    s2_cloud_mask_folderpath = "data/raw/S2_cloud_mask_large_thresh_40"
    cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_stations_data.csv"
    single_shadow_maps_nc = "data/processed/cloud_shadow_thresh40.nc"
    monthly_shadow_maps_nc = "data/processed/cloud_shadow_thresh40_monthly.nc"
    
    # Compute shadow projection and save shadow masks: 
    save_shadow_masks_to_nc(cloud_cover_table_filepath, s2_cloud_mask_folderpath,
                           single_shadow_maps_nc)
    
    # Now for upscaled cloud mask images
    for res in COARSE_RESOLUTIONS: 
        print(f"Resolution: {res}m")
        save_shadow_masks_to_nc(cloud_cover_table_filepath, f"data/processed/S2_cloud_mask_{res}m",
                            f"data/processed/cloud_shadow_{res}m.nc", verbose=False)
    