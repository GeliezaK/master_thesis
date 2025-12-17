# ==================================================================================
# Calculate the direct shortwave correction factor given the DSM and timestamps
# ==================================================================================

import horayzon as hray
import numpy as np
import rasterio
import os
from netCDF4 import Dataset, date2num
from tqdm import tqdm
import pandas as pd
from time import time
from skyfield.api import load, wgs84
from src.model.generate_LUT import DOY, HOD_DICT


def calculate_sw_dir_cor(
    dsm_filepath,
    times, 
    out_dir="data/processed/sw_cor"
):
    """
    Compute correction factor for direct shortwave radiation using Horayzon.

    Parameters
    ----------
    dsm_filepath : str
        Path to the digital surface model. 
    times : list of pd.Timestamp
        Timestamps for which to calculate the direct shortwave correction factor.
    out_dir : str
        Path to folder where to store the maps of correction factors for each timestamp.
    
    Returns
    -------
    out_file : str
        Path to .nc file where outputs are stored.
    """
    
    # -------------------------------------------------
    # Setup 
    # -------------------------------------------------
    
    # Read Dsm file    
    with rasterio.open(dsm_filepath) as src:
        dsm = src.read(1)  # first band
        transform = src.transform
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
    
    elevation_ortho = np.ascontiguousarray(dsm)
    # orthometric height (-> height above mean sea level)
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"sw_cor_bergen.nc"
    )
    
    ellps = "WGS84"
    
    # Compute ECEF coordinates
    dsm += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]
    
    x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                    dsm, ellps=ellps)
    
    # Transform to ENU coordinates
    trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[int(len(lon) / 2)], lat_or=lat[int(len(lat) / 2)], ellps=ellps)
    x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

    
    # Compute unit vectors (up and north) in ENU coordinates for whole domain
    vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon,lat))
    vec_north_ecef = hray.direction.north_dir(x_ecef, y_ecef,
                                            z_ecef, vec_norm_ecef,
                                            ellps=ellps)
    vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
    vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
 
    # Merge vertex coordinates and pad geometry buffer
    vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

    # Compute rotation matrix (global ENU -> local ENU)
    rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)
    
    # Compute slope (full grid, pad by 1 on each side)
    x_enu_pad = np.pad(x_enu, 1, mode="edge")
    y_enu_pad = np.pad(y_enu, 1, mode="edge")
    z_enu_pad = np.pad(z_enu, 1, mode="edge")
    
    vec_tilt_enu = np.ascontiguousarray(
        hray.topo_param.slope_plane_meth(
            x_enu_pad, y_enu_pad, z_enu_pad,
            rot_mat=rot_mat_glob2loc, output_rot=False
        )[1:-1, 1:-1]
    )
    
    # Compute surface enlargement factor
    surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
    
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
    # -> position lies on the surface of the ellipsoid by default

    # -----------------------------------------------------------------------------
    # Compute shortwave correction factor
    # -----------------------------------------------------------------------------

    # Loop through time steps and save data to NetCDF file incrementally
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
        
    return out_file


def main(): 
    sw_cor_outpath = "data/processed/sw_cor/sw_cor_bergen_binned.nc"
    DSM_filepath = "data/processed/bergen_dsm_10m_epsg4326_reducer_mean.tif"
    year=2023
    
    # Generate datetime objects for binned times to reduce number of runs
    times = []
    for doy in DOY:
        month_day = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy-1)
        sunshine_hours = HOD_DICT[doy]  # list of hours (UTC)
        for hour in sunshine_hours:
            dt = pd.Timestamp(year=month_day.year, month=month_day.month, day=month_day.day,
                            hour=hour, tz="UTC")
            times.append(dt)

    print(f"Generated {len(times)} timestamps for direct SW correction.")
    # Example: times is your list of datetime objects
    time_strings = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in times]

    # Or print all
    for t_str in time_strings:
        print(t_str)
        
    calculate_sw_dir_cor(dsm_filepath=DSM_filepath, 
                         times=times,
                         out_dir=sw_cor_outpath)
    


if __name__ == "__main__": 
    main()