# ==========================================================================================
# This script contains function to upscale the cloud mask images to coarser resolutions 
# and to count the number of misclassifications of 10m pixels at coarser resolutions. 
# ==========================================================================================

import os
import rasterio
import numpy as np
from netCDF4 import Dataset, num2date
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from rasterio.transform import from_bounds
from glob import glob
from src.model import COARSE_RESOLUTIONS


def upscale_to_res(folderpath, res, outpath):
    """
    Upscale 10m binary Sentinel-2 cloud masks to a coarser resolution by computing
    the median value (majority cloud/clear) per block.

    Parameters
    ----------
    folderpath : str
        Path to folder containing input 10m resolution GeoTIFFs (binary masks).
    res : int 
        Target resolution in meters (e.g., 100 for 100m).
    outpath : str
        Output folder path for upscaled GeoTIFFs.
    """
    
    os.makedirs(outpath, exist_ok=True)

    files = sorted(glob(os.path.join(folderpath, "*.tif")))

    # Loop over files, compute upscaled image and save
    for i,f in tqdm(enumerate(files), total=len(files), desc=f"Upscaling to res {res} m"):
        with rasterio.open(f) as src:
            data = src.read(1)
            orig_bounds = src.bounds
            deg_per_meter = 1 / 111320  # rough conversion
            scale = int((res * deg_per_meter) / src.res[0])
            
            if i == 0 :
                tqdm.write(f"Res {res}, scale: {scale}")
            
            if scale <= 1:
                tqdm.write(f"Skipping {os.path.basename(f)} — target res ≤ source res")
                continue

            # Compute new shape
            new_height = data.shape[0] // scale
            new_width = data.shape[1] // scale

            # Reshape array and compute median block-wise
            data_cropped = data[:new_height * scale, :new_width * scale]
            reshaped = data_cropped.reshape(new_height, scale, new_width, scale)
            median_block = np.median(reshaped, axis=(1, 3)).astype(np.int8)

            # Create new transform so that geographic bounds stay identical
            new_transform = from_bounds(
                orig_bounds.left, orig_bounds.bottom,
                orig_bounds.right, orig_bounds.top,
                new_width, new_height
            )

            # Write output GeoTIFF
            profile = src.profile
            profile.update({
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "dtype": "int8",
                "compress": "lzw"
            })

            outname = os.path.join(outpath, os.path.basename(f).replace(".tif", f"_{res}m.tif"))
            with rasterio.open(outname, "w", **profile) as dst:
                dst.write(median_block, 1)

            #tqdm.write(f"Upscaled {os.path.basename(f)} → {res}m")
            
            

def count_misclassifications(cloud_mask_10m_path, cloud_mask_coarse_path, resolution):
    """
    Compare a fine-resolution shadow mask NetCDF file (e.g. 10m)
    with a coarse-resolution shadow mask (e.g. 100m, 500m, 1000m),
    and count per-timestamp how many fine pixels disagree. 
    Compare fine- vs coarse-resolution cloud/shadow masks even when the grids
    do not align or divide evenly. Uses nearest-neighbor mapping in lat/lon space.

    Parameters
    ----------
    cloud_mask_10m_path : str
        Path to the .nc file with cloud shadow masks at 10m resolution. 
    cloud_mask_coarse_path : str
        Path to the .nc file with cloud shadow masks at coarse resolution.
    resolution : int 
        Resolution of the coarse resolution in m. 

    Returns
    -------
    pd.DataFrame with columns:
        date, resolution, misclassified_count, misclassified_percentage
    """

    # -----------------------
    # Open NetCDF files
    # -----------------------
    nc_fine = Dataset(cloud_mask_10m_path)
    nc_coarse = Dataset(cloud_mask_coarse_path)

    t_fine = nc_fine["time"][:]
    t_coarse = nc_coarse["time"][:]

    # Convert times
    dates_fine = num2date(t_fine, units=nc_fine["time"].units)
    dates_coarse = num2date(t_coarse, units=nc_coarse["time"].units)
    
    # Find common timestamps
    dt_fine_strings = np.array([d.isoformat() for d in dates_fine])
    dt_coarse_strings = np.array([d.isoformat() for d in dates_coarse])

    common_dates = np.intersect1d(dt_fine_strings, dt_coarse_strings)

    if len(common_dates) == 0:
        print("No matching timestamps between fine and coarse datasets.")
        return pd.DataFrame()

    # -----------------------
    # Read lat/lon to detect mapping
    # -----------------------
    fine_lat = nc_fine["lat"][:]
    fine_lon = nc_fine["lon"][:]
    coarse_lat = nc_coarse["lat"][:]
    coarse_lon = nc_coarse["lon"][:]

    # Create coordinate grids
    fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)
    coarse_lon_grid, coarse_lat_grid = np.meshgrid(coarse_lon, coarse_lat)

    # Flatten grids for KD-tree
    fine_pts = np.column_stack((fine_lat_grid.ravel(), fine_lon_grid.ravel()))
    coarse_pts = np.column_stack((coarse_lat_grid.ravel(), coarse_lon_grid.ravel()))

    # Build KD-tree on coarse pixel centers
    tree = cKDTree(coarse_pts)

    # Query nearest coarse pixel for each fine pixel
    _, nn_idx = tree.query(fine_pts)     # nearest neighbor index per fine pixel
    coarse_row = nn_idx // len(coarse_lon)
    coarse_col = nn_idx % len(coarse_lon)

    # Store mapping (1D index of coarse pixel per fine pixel)
    # This is reused for all timestamps → very fast
    coarse_row = coarse_row.reshape(fine_lat_grid.shape)
    coarse_col = coarse_col.reshape(fine_lat_grid.shape)

    # -----------------------
    # Compare per timestamp
    # -----------------------
    df_rows = []

    for date_str in tqdm(common_dates, total=len(common_dates), desc=f"Counting misclassifications for res {resolution} m"):
        i_f = np.where(dt_fine_strings == date_str)[0][0]
        i_c = np.where(dt_coarse_strings == date_str)[0][0]

        fine_mask = nc_fine["shadow_mask"][i_f, :, :]
        coarse_mask = nc_coarse["shadow_mask"][i_c, :, :]

        # Upscale coarse → fine grid
        coarse_mapped = coarse_mask[coarse_row, coarse_col]
        
        # Misclassification = differing mask if non-nan
        diff = fine_mask != coarse_mapped
        misclassified_count = diff.sum()
        total_pixels = diff.size
        percentage = misclassified_count / total_pixels * 100
        
        # -----------------------
        # Cloud cover counts
        # -----------------------
        # Cloudy = shadow_mask == 1 
        fine_cloudy = (fine_mask == 1)
        coarse_cloudy = (coarse_mask == 1)

        cloud_cover_10m = fine_cloudy.sum() / fine_cloudy.size * 100
        cloud_cover_coarse = coarse_cloudy.sum() / coarse_cloudy.size * 100

        df_rows.append({
            "date": date_str,
            "resolution": resolution,
            "misclassified_count": misclassified_count,
            "misclassified_percentage": percentage,
            "cloud_cover_10m": cloud_cover_10m,
            "cloud_cover_coarse": cloud_cover_coarse
        })

    nc_fine.close()
    nc_coarse.close()

    return pd.DataFrame(df_rows)
        

if __name__ == "__main__":
    s2_cloud_mask_folderpath = "data/raw/S2_cloud_mask_large_thresh_40"
    cloud_shadow_10m_filepath = "data/processed/cloud_shadow_thresh40.nc"
    misclassification_count_out_csv = "data/processed/misclassification_counts_upscaled_cloud_mask.csv"
    date="2016-02-04"
    sample_file = f"data/raw/S2_cloud_mask_large_thresh_40/S2_cloud_mask_large_{date}.tif"
    
    # Extract the max and min coordinates from sample 10 m resolution file 
    with rasterio.open(sample_file) as src:
            bounds = src.bounds  # (left, bottom, right, top)
            shape = src.shape    # (height, width)
            lon_min_orig = bounds.left
            lon_max_orig = bounds.right
            lat_min_orig = bounds.bottom
            lat_max_orig = bounds.top

            print(f"\nResolution: 10 m")
            print(f"  Shape (rows, cols): {shape}")
            print(f"  Lon (min, max): ({lon_min_orig:.6f}, {lon_max_orig:.6f})")
            print(f"  Lat (min, max): ({lat_min_orig:.6f}, {lat_max_orig:.6f})")
    
    for res in COARSE_RESOLUTIONS:
        # Upscale all 10 m resolution images to target resolution and save as .tif files in new folder
        upscale_to_res(s2_cloud_mask_folderpath, res, f"data/processed/S2_cloud_mask_{res}m")
        
        # Test for a sample image that upscaling preserves coordinate boundaries
        tif_path = f"data/processed/S2_cloud_mask_{res}m/S2_cloud_mask_large_{date}_{res}m.tif"
        with rasterio.open(tif_path) as src:
            bounds = src.bounds  # (left, bottom, right, top)
            shape = src.shape    # (height, width)

            # Assert min and max coordinates stay the same for upscaled images 
            print(f"\nResolution: {res} m")
            print(f"  Shape (rows, cols): {shape}")
            delta = 10e-8
            assert abs(bounds.left - lon_min_orig) < delta , f"Lon (min) different from orig: ({bounds.left:.6f}, {lon_min_orig:.6f})"
            assert abs(bounds.right - lon_max_orig) < delta , f"Lon (max) different from orig: ({bounds.right:.6f}, {lon_max_orig:.6f})"
            assert abs(bounds.bottom - lat_min_orig) < delta , f"Lat (min) different from orig: ({bounds.bottom:.6f}, {lat_min_orig:.6f})"
            assert abs(bounds.top - lat_max_orig) < delta , f"Lat (max) different from orig: ({bounds.top:.6f}, {lat_max_orig:.6f})"
    
    all_results = []
    
    for res in COARSE_RESOLUTIONS: 
        print(f"---------------- Resolution {res} m ---------------")
        coarse_filepath = f"data/processed/cloud_shadow_{res}m.nc"
        df = count_misclassifications(cloud_shadow_10m_filepath, coarse_filepath, res)
        print(df.head())
        all_results.append(df)
        
    # Concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(misclassification_count_out_csv, index=False)

    print("\nSaved combined misclassification dataframe:")
    print(misclassification_count_out_csv)
        