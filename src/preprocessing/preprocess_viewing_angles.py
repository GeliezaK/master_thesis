# ===============================================================================
# Script with functions for preprocessing steps for Sentinel-2 viewing angles 
# Sentinel-2 viewing angles are exported from GEE Sentinel-2 Harmonized dataset 
# with the export script in src/io/export_sentinel2_images.ipynb
# ================================================================================

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import glob 
import os

def merge_all_in_path(inpath, outpath): 
    """Merge all .csv files in inpath and write new merged dataframe as .csv to outpath."""
    # Read and merge all CSV files
    all_files = glob.glob(os.path.join(inpath, "*.csv"))
    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Create "date" column from "system:time_start" (ms → datetime)
    merged_df["date"] = pd.to_datetime(merged_df["system:time_start"], unit="ms").dt.strftime("%Y-%m-%d")

    # Drop bands B3, B6, B7 because they are not used in cloud detection algorithm
    cols_to_drop = [col for col in merged_df.columns 
                    if any(band in col for band in ["B3", "B6", "B7"])]
    cols_to_drop.append(".geo")

    merged_df = merged_df.drop(columns=cols_to_drop)
    
    # Create columns with mean zenith/azimuth and columns with std zenith/azimuth across bands 
    
    # Select the azimuth and zenith columns across B1–B12
    azimuth_cols = [col for col in merged_df.columns if "MEAN_INCIDENCE_AZIMUTH_ANGLE_B" in col]
    zenith_cols  = [col for col in merged_df.columns if "MEAN_INCIDENCE_ZENITH_ANGLE_B" in col]

    # Compute mean and std across bands
    merged_df["MEAN_AZIMUTH"] = merged_df[azimuth_cols].mean(axis=1, skipna=True)
    merged_df["STD_AZIMUTH"]  = merged_df[azimuth_cols].std(axis=1, ddof=0, skipna=True)

    merged_df["MEAN_ZENITH"]  = merged_df[zenith_cols].mean(axis=1, skipna=True)
    merged_df["STD_ZENITH"]   = merged_df[zenith_cols].std(axis=1, ddof=0, skipna=True)
    
    # Save merged DataFrame
    merged_df.to_csv(outpath, index=False)
    
def plot_std_hist_of_daily_groups(df, variable_name, outpath):
    """Group df by date and compute the standard dev of given variable. Then plot histogram and save to 
    outpath"""
    # Group by date and compute std across granules
    std_per_date = df.groupby("date").agg(
        STD_PER_DATE=(variable_name, "std"),
    ).reset_index()

    # Plot histogram of STD_MEAN_ZENITH
    plt.figure(figsize=(8,5))
    plt.hist(std_per_date["STD_PER_DATE"].dropna(), bins=30, edgecolor="black")
    plt.xlabel(f"Std of {variable_name} across granules")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Std of {variable_name} across granules")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved histogram to {outpath}.")


def add_viewing_angles(cloud_cover_filepath, viewing_angle_filepath, outpath):
    """Add variable MEAN_AZIMUTH and MEAN_ZENITH to each cloud cover observation."""
    # Load the CSV files
    cloud_cover_df = pd.read_csv(cloud_cover_filepath)
    angles_df = pd.read_csv(viewing_angle_filepath)

    # Only keep viewing angles for MGRS_TILE == 32VKM
    angles_df = angles_df[angles_df["MGRS_TILE"] == "32VKM"].copy()

    # Helper function to check if all values in a column are the same
    def all_same(series):
        return series.nunique() == 1

    # Create dictionaries to quickly look up mean azimuth/zenith per date
    azimuth_dict = {}
    zenith_dict = {}

    for date, group in angles_df.groupby("date"):
        if len(group) == 1:
            # Only one observation, take values directly
            azimuth_dict[date] = group["MEAN_AZIMUTH"].values[0]
            zenith_dict[date] = group["MEAN_ZENITH"].values[0]
        else:
            if (all_same(group["SPACECRAFT_NAME"]) and
                all_same(group["SENSING_ORBIT_DIRECTION"]) and
                all_same(group["SENSING_ORBIT_NUMBER"]) and
                (group["MEAN_ZENITH"].max() - group["MEAN_ZENITH"].min() < 1) and
                (group["MEAN_AZIMUTH"].max() - group["MEAN_AZIMUTH"].min() < 1)):
                azimuth_dict[date] = group["MEAN_AZIMUTH"].values[0]
                zenith_dict[date] = group["MEAN_ZENITH"].values[0]
            else:
                azimuth_dict[date] = np.nan
                zenith_dict[date] = np.nan

    # Map the values back to the cloud_cover_df
    cloud_cover_df["MEAN_AZIMUTH"] = cloud_cover_df["date"].map(azimuth_dict)
    cloud_cover_df["MEAN_ZENITH"] = cloud_cover_df["date"].map(zenith_dict)

    # Save back to the same filepath
    cloud_cover_df.to_csv(outpath, index=False)

    # Print the head
    print(cloud_cover_df.head())
    # Print shape of the DataFrame
    print("Shape of cloud_cover_df:", cloud_cover_df.shape)

    # Print summary statistics for selected columns
    cols_of_interest = ["MEAN_AZIMUTH", "MEAN_ZENITH", "cloud_cover_large", "blue_sky_albedo_median"]
    print("\nSummary statistics:")
    print(cloud_cover_df[cols_of_interest].describe())

    # Optional: check number of missing values
    print("\nMissing values per column:")
    print(cloud_cover_df[cols_of_interest].isna().sum())


if __name__ == "__main__": 
    # Import dataframes 
    # Paths
    viewing_angles_folder_path = "data/processed/S2_viewing_angles"
    viewing_angles_table_path = "data/processed/S2_viewing_angles_full_table.csv"
    s2_inpath = "data/processed/s2_cloud_cover_table_small_and_large.csv"
    
    # Single files in folder into one file 
    #merge_all_in_path(viewing_angles_folder_path, viewing_angles_table_path)
    
    plot_std_hist_of_daily_groups(viewing_angles_table_path)
    
    # Merge into new file
    add_viewing_angles(cloud_cover_filepath=s2_inpath, 
                       viewing_angle_filepath=viewing_angles_table_path, 
                       outpath="data/processed/s2_cloud_cover_table_small_and_large.csv")
    
