import pvlib # for solar position
import math
import numpy as np 
from datetime import datetime
import rasterio 
from rasterio.windows import from_bounds
from scipy.ndimage import shift
from time import time
from src.plotting.high_res_maps import plot_single_band
from src.model.surface_GHI_model import CENTER_LAT, CENTER_LON, BBOX

R_EARTH = 6371000


# Get solar zenith and azimuth
def get_solar_angle(datetime, lat=CENTER_LAT, lon=CENTER_LON): 
    solpos = pvlib.solarposition.get_solarposition(datetime, lat, lon)
    zenith = float(solpos['zenith'].values[0])
    azimuth = float(solpos['azimuth'].values[0])
    return zenith, azimuth


def get_cloud_shadow_displacement(solar_zenith, solar_azimuth,
                         cbh_m, sat_zenith=0.0, sat_azimuth=0.0,
                         pixel_size=10, cloud_top_height=None, verbose=False):
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

    # Convert to pixels
    dx_pix = dx_total / pixel_size
    dy_pix = dy_total / pixel_size

    if verbose:
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

    # Shift cloud mask
    shadow_mask = shift(cloud_mask,
                        shift=(-dy_pix, dx_pix),
                        mode="constant", order=0, cval=0)
    shadow_mask = (shadow_mask > 0).astype(np.uint8)

    # Convert bbox to raster window
    window = from_bounds(bbox["west"], bbox["south"],
                         bbox["east"], bbox["north"],
                         transform=transform)

    # Round values
    row_off = math.floor(window.row_off)
    col_off = math.floor(window.col_off)
    height = math.ceil(window.height)
    width = math.ceil(window.width)

    roi_mask = shadow_mask[
        row_off:row_off + height,
        col_off:col_off + width
    ]

    return roi_mask, transform


if __name__=="__main__":
    # Cloud shadow computation
    # Cloud base height (e.g., 2000 m)
    s2_cloud_mask_sample = "data/raw/S2_cloud_mask_large/S2_cloud_mask_large_2017-08-25_11-06-49-2017-08-25_11-06-49.tif"
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
    shadow_mask, shadow_transform = project_cloud_shadow(s2_cloud_mask_sample, dy_pix, dx_pix, BBOX)
    print(f"Shadow computation takes {np.round(time()-start,3)} secs.")
    plot_single_band(shadow_mask, f"output/shadow_mask_{timestamp_str}_sza_{solar_zenith:.2f}_azi_{solar_azimuth:.2f}.png", 
                     f"Shadow Mask for {timestamp_str} (SZA {solar_zenith:.2f}, AZI {solar_azimuth:.2f})",
                     "Shadow", [0, 1], ["white", "darkgrey"])
