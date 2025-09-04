# Model surface GHI 
# Inputs:   - filepath to DSM/DEM file 
#           - folderpath to LUT 
#           - folderpath to S2 images

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box
import glob
import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import math
import pvlib  # for solar position
from datetime import datetime
from rasterio.plot import show
from high_res_cloud_maps import plot_landmarks, plot_single_band, plot_single_band_histogram

# Convert degree to radian
def deg2rad(deg):
    return deg * math.pi / 180


# Define the center of the bounding box (Bergen, Norway)
CENTER_LAT = 60.39
CENTER_LON = 5.33

# Approximate degree adjustments for 100km x 100km box
DEG_LAT_TO_KM = 111.412  # 1 degree latitude at 60° converted to km (https://en.wikipedia.org/wiki/Latitude)
DEG_LON_TO_KM = 111.317 * math.cos(deg2rad(CENTER_LAT))  # 1 degree longitude converted to km
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
def get_solar_angle(datetime_str, lat=CENTER_LAT, lon=CENTER_LON): 
    dt = datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")
    solpos = pvlib.solarposition.get_solarposition(dt, lat, lon)
    zenith = float(solpos['zenith'].values[0])
    azimuth = float(solpos['azimuth'].values[0])
    return zenith, azimuth

# Compute hillshade from DEM
def get_hillshade_band(dem, zenith, azimuth, profile): 
    # resolution (in map units)
    xres = profile['transform'][0]
    yres = -profile['transform'][4]
    print(xres)
    print(yres)

    # slope and aspect
    x, y = np.gradient(dem, xres, yres)
    slope = np.pi/2 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    # convert angles to rad
    zenith_rad = np.deg2rad(zenith)
    azimuth_rad = np.deg2rad(azimuth)

    # hillshade (0–1)
    shaded = (np.sin(zenith_rad) * np.sin(slope) +
              np.cos(zenith_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect))

    hillshade = np.clip(shaded, 0, 1)

    return hillshade


def project_cloud_shadow(cloud_mask, solar_zenith, solar_azimuth,
                         cbh_m, sat_zenith=0.0, sat_azimuth=0.0,
                         pixel_size=10, cloud_top_height=None):
    """
    Project cloud shadows onto the surface given cloud mask,
    solar and satellite geometry.
    
    Parameters
    ----------
    cloud_mask : 2D numpy array (0 clear, 1 cloud)
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
    shadow_mask : 2D numpy array (0 no shadow, 1 shadow)
    """
    assert 0 <= solar_zenith < 90, (
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

    horiz_disp = cbh_m * np.tan(solar_zenith_rad)
    dx_sun = horiz_disp * np.sin(solar_azimuth_rad)
    dy_sun = -horiz_disp * np.cos(solar_azimuth_rad)
    print(f"horizontal_disp: {horiz_disp:.2f}, x: {dx_sun:.2f}, y: {dy_sun:.2f}")

    dx_total = dx_sat + dx_sun
    dy_total = dy_sat + dy_sun

    # Convert to pixels
    dx_pix = dx_total / pixel_size
    dy_pix = dy_total / pixel_size

    print(f"Total displacement in pixels: x={dx_pix:.2f}, y={dy_pix:.2f}")

    # Apply single shift
    shadow_mask = shift(cloud_mask,
                        shift=(-dy_pix, dx_pix),
                        mode='constant', order=0, cval=0)

    shadow_mask = (shadow_mask > 0).astype(np.uint8)
    
    return shadow_mask


if __name__ == "__main__": 
    s2_img, profile = load_image("data/S2_testfiles/S2_cloud_2018-07-28.tif")
    #print(profile)
    
    # example datetime
    datetime_str = "2017-03-21_13-03-58"
    solar_zenith, solar_azimuth = get_solar_angle(datetime_str) # Note: times in UTC!
    print(f"Solar angles for datetime {datetime_str}: SZA {np.round(solar_zenith,2)}, and AZI {np.round(solar_azimuth,2)}.")
    
    # Cloud base height (e.g., 2000 m)
    cbh_m = 1000 
    sat_zenith = 7.5
    sat_azimuth = 108.5

    shadow_mask = project_cloud_shadow(s2_img, solar_zenith, solar_azimuth, cbh_m, 
                                       sat_zenith, sat_azimuth, pixel_size = 10, cloud_top_height=cbh_m)


    plot_single_band(
        band=shadow_mask,
        outpath=f"output/shadow_mask_{datetime_str}_satzenith_{np.round(sat_zenith,1)}_satazi_{np.round(sat_azimuth,1)}.png",
        title=f"Shadow mask {datetime_str}",
        colorbar_label="Shadow",
        value_ranges=[0,0.5],
        value_colors=["white", "#4C4E52"]
    )
    
    """plot_single_band_histogram(hillshade_band, 
                               f"output/hillshade_SZA{np.round(zenith,2)}_AZI{np.round(azimuth,2)}_histogram.png",
                               f"Histogram of pixel values in hillshade image",
                               "Relative illumination value",
                               "Frequency")
    plot_single_band_histogram(dsm10m, 
                               f"output/dsm_bergen_10m_histogram.png",
                               "Histogram of elevation values (Bergen, 10m pixels)",
                               "Elevation (m)",
                               "Frequency")
    
    """