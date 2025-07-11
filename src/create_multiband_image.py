# Create multiband image from several one-band .tif files for k-means clustering in GEE
import geemap
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np
import os

def write_to_multiband(tif_files, outpath):
    """Write multiple tif images to one multiband image"""
    # Use metadata from first file
    with rasterio.open(tif_files[0]) as src0:
        meta = src0.meta.copy()
        meta.update(count=len(tif_files))  # set number of bands

    # Write each band directly without stacking to RAM
    with rasterio.open(outpath, 'w', **meta) as dst:
        for idx, file in enumerate(tif_files):
            print(f"Reading file {file}...")
            with rasterio.open(file) as src:
                band = src.read(1)
                dst.write(band, idx + 1)

    print(f"Written multi-band file: {outpath}")

# List your files
tif_files = []

months = range(1,13)
years = range(2016,2025)
seasons = ["Winter", "Spring", "Summer", "Autumn"]
percentiles = ["Q1", "median"] # without q3 because it's not discriminative

#tif_files.append(f"data/Cloud_mask_mean_alltime_EPSG32632.tif")

#for year in years: # Add yearly aggregates
#    tif_files.append(f"data/Cloud_mask_mean_{year}.tif")

""" for season in seasons: 
    tif_files.append(f"data/Cloud_mask_mean_{season}.tif")

write_to_multiband(tif_files, "output/Cloud_mask_mean_seasons.tif")
tif_files.clear()

for month in months: 
    tif_files.append(f"data/Cloud_mask_mean_month{month}_EPSG32632.tif")

write_to_multiband(tif_files, "output/Cloud_mask_mean_months.tif")
tif_files.clear() """

for p in percentiles: 
    tif_files.append(f"data/Cloud_prob_{p}_alltime.tif")
    for month in months: 
        tif_files.append(f"data/Cloud_prob_{p}_month{month}.tif")
    for season in seasons: 
        tif_files.append(f"data/Cloud_prob_{p}_{season}.tif")
    
    write_to_multiband(tif_files, f"output/Cloud_prob_{p}_months_seasons.tif")
    tif_files.clear()
