import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

cloud_cover_table_filepath = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"

def mean_sat_angles_distribution(cloud_cover_filepath, outpath=None):
    cloud_props = pd.read_csv(cloud_cover_table_filepath)
    cloud_props = cloud_props[["MEAN_AZIMUTH", "MEAN_ZENITH"]]
    # Count NaN values per column
    nan_counts = cloud_props.isna().sum()

    print("NaN counts per column:")
    print(nan_counts)
    
    # Remove NaN values for plotting
    zenith_values = cloud_props["MEAN_ZENITH"].dropna()
    azimuth_values = cloud_props["MEAN_AZIMUTH"].dropna()

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for MEAN_ZENITH
    axes[0].hist(zenith_values, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Distribution of MEAN_ZENITH")
    axes[0].set_xlabel("Zenith Angle [deg]")
    axes[0].set_ylabel("Count")

    # Histogram for MEAN_AZIMUTH
    axes[1].hist(azimuth_values, bins=30, color='salmon', edgecolor='black')
    axes[1].set_title("Distribution of MEAN_AZIMUTH")
    axes[1].set_xlabel("Azimuth Angle [deg]")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if outpath is None: 
        outpath = f"output/mean_sat_viewing_angles_hist.png"
    plt.savefig(outpath)
    print(f"Histogram saved to {outpath}.")
    
    
def mean_sat_angles_per_doy(cloud_cover_filepath, start_date=None, end_date=None, outpath=None):
    """
    Plot mean MEAN_ZENITH and MEAN_AZIMUTH per DOY (day of year) for a sample date range.
    Saves figure to file instead of showing.
    """
    cloud_props = pd.read_csv(cloud_cover_filepath)
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    
    # Filter date range if provided
    if start_date is not None:
        cloud_props = cloud_props[cloud_props["date"] >= start_date]
    if end_date is not None:
        cloud_props = cloud_props[cloud_props["date"] <= end_date]

    # Extract DOY
    cloud_props["doy"] = cloud_props["date"].dt.dayofyear

    # Group by DOY and compute mean angles
    mean_angles = cloud_props.groupby("doy")[["MEAN_ZENITH", "MEAN_AZIMUTH"]].mean().reset_index()

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Mean Zenith
    axes[0].plot(mean_angles["doy"], mean_angles["MEAN_ZENITH"], color="skyblue", marker='o', linestyle='-')
    axes[0].set_ylabel("Mean Zenith Angle [deg]")
    axes[0].set_title("Mean Zenith Angle per DOY")

    # Mean Azimuth
    axes[1].plot(mean_angles["doy"], mean_angles["MEAN_AZIMUTH"], color="salmon", marker='o', linestyle='-')
    axes[1].set_xlabel("Day of Year (DOY)")
    axes[1].set_ylabel("Mean Azimuth Angle [deg]")
    axes[1].set_title("Mean Azimuth Angle per DOY")

    plt.tight_layout()

    if outpath is None:
        outpath = "output/mean_sat_viewing_angles_per_doy.png"
    plt.savefig(outpath)
    print(f"Mean angles per DOY plot saved to {outpath}.")
    
def satellite_viewing_phases(cloud_cover_filepath):
    df = pd.read_csv(cloud_cover_table_filepath)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Detect phase changes with tolerance of 1 degree
    zenith = df["MEAN_ZENITH"].values
    azimuth = df["MEAN_AZIMUTH"].values
    
    # Initialize phase_id array
    phase_id = [0]

    for i in range(1, len(df)):
        if not (np.isclose(zenith[i], zenith[i-1], atol=1) and np.isclose(azimuth[i], azimuth[i-1], atol=1)):
            phase_id.append(phase_id[-1] + 1)  # new phase
        else:
            phase_id.append(phase_id[-1])  # same phase

    df["phase_id"] = phase_id

    # Calculate phase lengths in days
    phase_lengths = df.groupby("phase_id").agg(
        start_date=("date", "first"),
        end_date=("date", "last"),
        zenith=("MEAN_ZENITH", "first"),
        azimuth=("MEAN_AZIMUTH", "first")
    ).reset_index()

    phase_lengths["length_days"] = (phase_lengths["end_date"] - phase_lengths["start_date"]).dt.days + 1

    print(phase_lengths)
    

def scatter_viewing_angles(cloud_cover_filepath, outpath=None):
    cloud_props = pd.read_csv(cloud_cover_filepath)

    unique_angles = cloud_props[["MEAN_ZENITH", "MEAN_AZIMUTH"]].drop_duplicates()

    plt.figure(figsize=(6,6))
    plt.scatter(unique_angles["MEAN_ZENITH"], unique_angles["MEAN_AZIMUTH"], color="blue", s=100)
    plt.xlabel("Zenith Angle [deg]")
    plt.ylabel("Azimuth Angle [deg]")
    plt.title("Zenith vs Azimuth Viewing Geometry")
    plt.grid(True)
    if outpath is None:
        outpath = "output/scatter_sat_zenith_azimuth.png"
    plt.savefig(outpath)
    print(f"Scatter plot saved to {outpath}.")

if __name__ == "__main__": 
    #mean_sat_angles_distribution(cloud_cover_table_filepath)
    #mean_sat_angles_per_doy(cloud_cover_table_filepath, start_date="2024-01-01", end_date="2025-12-31")
    #scatter_viewing_angles(cloud_cover_table_filepath)
    satellite_viewing_phases(cloud_cover_table_filepath)