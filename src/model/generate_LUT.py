# ==================================================================================
# Generate the lookup-table (LUT) with libRadtran radiative transfer solver Disort 
# ==================================================================================

import os
import glob
import subprocess
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm  # Progress bar
from astral import LocationInfo
from astral.sun import sun
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_sunshine_hours_for_doy(doy, location):
    """Get hours with sunshine for day of year doy and location."""
    # Convert DOY to date in a non-leap year (e.g., 2024)
    year = 2024  # Non-leap year is fine unless you're using 366
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)

    # Get sunrise and sunset (local time)
    s = sun(location.observer, date=date, tzinfo=location.timezone)
    sunrise_utc = s["sunrise"].astimezone(pytz.utc)
    sunset_utc = s["sunset"].astimezone(pytz.utc)
    
    print(f"Bergen sunshine DOY {doy} : {sunrise_utc.hour}:{sunrise_utc.minute} - {sunset_utc.hour}:{sunset_utc.minute} UTC")

    # Round up/down to nearest full hour in UTC
    start_hour = int(np.round(sunrise_utc.hour + sunrise_utc.minute / 60))
    end_hour = int(np.round(sunset_utc.hour + sunset_utc.minute / 60))

    # Create list of full UTC hours between sunrise and sunset
    if end_hour >= start_hour:
        return list(range(start_hour, end_hour + 1))
    else:
        return []  # Polar night fallback


# Parameter discretizations
# Day of year: 1st and 15th of each month 
# Reference year (non-leap year assumed, e.g. 2021)
year = 2021

# Generate DOY for the 1st and 15th of each month
DOY = [1, 15, 32, 46, 60, 74, 91, 105, 121, 135, 152, 166, 182, 196, 213, 227, 244, 258, 274, 288, 305, 319, 335, 349]
doy_to_date = {}
for doy in DOY:
    dt = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    doy_to_date[doy] = (dt.month, dt.day)

print(doy_to_date)

# Hour of day: all values that have sunshine throughout the year UTC
# Bergen timezones: Apr-Oct: UTC+2, Nov-Mar UTC+1. Calculate from astral model and convert to UTC
# Define Bergen location
bergen_location = LocationInfo(name="Bergen", region="Norway", timezone="Europe/Oslo", latitude=60.39, longitude=5.33)
HOD_DICT = {doy: get_sunshine_hours_for_doy(doy, bergen_location) for doy in DOY}
print(HOD_DICT)

# Monthly Aerosol optical depth values at 550nm from Modis MCD19A2.061 
# compare Nikitidou et al. (2014)
AOD_MONTHLY = {1: 0.077, 2:0.081, 3:0.075, 4:0.087,
               5: 0.110, 6:0.103, 7:0.104, 8:0.109,
               9: 0.094, 10:0.061, 11:0.067, 12:0.072}

# Surface albedo values
# 50 percentile, from claas3 small region subset
ALBEDO_VALUES = [0.129] 

# Altitude in km 
# Range in Bergen is 0 (sea level) - 800 with a peak around 50m. From Copernicus DEM 30m resolution
# 50%, 75%, 95% percentiles from DEM: [0.08, 0.226, 0.521]
# Set constant value to 50% percentile. Set to median height because it does not have a large effect on GHI data. 
ALTITUDE = 0.08 

# Effective droplet radius
# For ice clouds
IC_REFF = 50 # from Hong and Liu (2015) for 60° Latitude, 5 km altitude, vs. 20 from libRadtran example 
# For water clouds 
WC_REFF = 10 # from libRadtran example & Han (1994)

# Liquid/ice water content
LWC = 0.1 # from libRadtran simple water cloud example and Lee et al. (2010)
IWC = 0.015 # from Hong and Liu (2015) for 60° Latitude 5km cloud altitude, vs. 0.015 from libRadtran simple ice cloud example

# Cloud optical depth at 760nm, from Sentinel-5P
# log-scaled 14 values between 0 and 60, 
# plus values 65, 75, 90, 100, 140, 150, empirical values from claas cot small region
COT =  [0.1, 0.31, 0.75, 1.31, 2.17, 3.4, 4.87, 7.11, 9.54, 14.03, 19.66, 29.09, 38.73, 53.51, 65, 75, 90, 100, 140, 150]
 
# Cloud base height in km, from CLAAS-3
# Altitude + Percentiles 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, empirical values from CLAAS-3 
CLOUD_TOP_HEIGHT = [0.575,  1.291,  2.008, 2.724,  3.440,  4.157,  4.873,  5.589,  6.306,  7.022, 
                    7.738, 8.454, 9.171, 9.887, 10.603, 11.320, 12.036, 12.752, 13.469, 14.185]

# Cloud thickness (vertical extent) in km, from CLAAS-3
# fixed at 1km, according to satellite data valid for large majority of clouds 
CLOUD_GEOGRAPHICAL_THICKNESS = 1 

# Cloud types
CLOUD_PHASE = ['ice', 'water']


# Output directory 
OUTPUT_DIR = "data/processed/LUT/claas3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def descriptive_stats_claas3(df):
    """Generate descriptive statistics of cloud properties in Bergen based on claas 3 data."""
    print(df.describe())
    
    # Calculate percentiles
    # Define percentiles
    albedo_percentiles = [5, 25, 50, 75, 95]
    cloud_percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    # Compute and round percentiles
    albedo_stats = df["blue_sky_albedo_median"].quantile([p / 100 for p in albedo_percentiles]).round(3)
    cth_stats = df["cth_median_small"].quantile([p / 100 for p in cloud_percentiles]).round(0)
    cot_stats = df["cot_median_small"].quantile([p / 100 for p in cloud_percentiles]).round(2)
    
    # Add linearly spaced values for CTH
    cth_min, cth_max = df["cth_median_small"].min(), df["cth_median_small"].max()
    cth_linspace = np.linspace(cth_min, cth_max, 20).round(0)

    # Add logarithmically spaced values for COT (exclude zero/negative)
    cot_subset = df.loc[(df["cot_median_small"] >= 0) & (df["cot_median_small"] <= 60), "cot_median_small"]

    cot_percentiles = cot_subset.quantile([p / 100 for p in cloud_percentiles]).round(2).tolist()
    cot_min = np.round(df["cot_median_small"].min(),2)

    cot_custom = sorted(set(cot_percentiles + [cot_min, 65, 75, 90, 100, 140, 150]))

    print("\nAlbedo percentiles:\n", albedo_stats)
    print("\nCloud Top Height percentiles:\n", cth_stats)
    print("\nCloud Optical Thickness percentiles:\n", cot_stats)

    print("\nLinearly spaced Cloud Top Height values:\n", cth_linspace)
    print("\nCustom Cloud Optical Thickness values:\n", cot_custom)
    
    # Plot 4 histograms in a 2x2 grid
    fig, axs = plt.subplots(1, 3, figsize=(14, 8))
    fig.suptitle("Distributions of Cloud and Surface Properties", fontsize=16)

    # Plot each feature
    axs[0].hist(df["blue_sky_albedo_median"], bins=50, color='skyblue', edgecolor='black')
    axs[0].axvline(albedo_stats.loc[0.5], color="red", linestyle="--", label="50% percentile")
    axs[0].set_title("Surface Albedo")
    axs[0].set_xlabel("Albedo")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(df["cth_median_small"], bins=50, color='lightgreen', edgecolor='black')
    for val in cth_linspace:
        axs[1].axvline(val, color="black", linestyle="--", alpha=0.5)
    axs[1].set_title("Cloud Top Height (m)")
    axs[1].set_xlabel("Height (m)")
    axs[1].set_ylabel("Frequency")

    axs[2].hist(df["cot_median_small"], bins=50, color='orange', edgecolor='black')
    for val in cot_custom:
        axs[2].axvline(val, color="black", linestyle="--", alpha=0.5)
    axs[2].set_title("Cloud Optical Depth")
    axs[2].set_xlabel("Optical Depth")
    axs[2].set_ylabel("Frequency")

    # Layout adjustment
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath="output/claas3_cloud_properties_descriptive_stats.png"
    plt.savefig(outpath)
    print(f"Descriptive claas stats saved to {outpath}.")


def generate_LUT():
    """Generate the look-up table with radiative transfer calculations from libradtran (former uvspec) DISORT routine."""
    os.makedirs("cloudfiles", exist_ok=True)
    
    # Flattened combinations including hour
    all_combos = []
    for doy in DOY:
        for hour in HOD_DICT[doy]:
            for albedo in ALBEDO_VALUES:
                all_combos.append((doy, hour, albedo))


    for doy, hour, albedo in tqdm(all_combos, desc="🌍 Generate LUT ", unit="config"):
        # File naming convention
        filename = f"LUT_doy{doy}_hod{hour}_alb{albedo}.csv"
        output_path = os.path.join(OUTPUT_DIR, filename.replace(" ", "_"))
        
        # Skip if output file already exists
        if os.path.exists(output_path):
            tqdm.write(f"⏩ Skipping existing file: {output_path}")
            continue
        
        # Select atmospheric profile: midlatitudes is default for this region
        # midlatitudes_summer for April - September, else midlatitudes_winter
        profile = "midlatitude_summer" if 105 <= doy <= 258 else "midlatitude_winter"

        # Clear sky simulation
        clear_input = generate_uvspec_input(doy, hour, albedo, profile)
        ghi_clear, direct_clear, diffuse_clear = run_uvspec(clear_input)
        
        # Skip if the sun is not visible
        if ghi_clear == 0.0: 
            tqdm.write(f"No sunshine for DOY {doy}, hour {hour}; skip")
            continue

        tqdm.write(f"\n🔵 CLEAR SKY | Atmos={profile}, DOY={doy}, Hour= {hour}, Albedo={albedo}")
        tqdm.write(f"    → GHI_clear={ghi_clear:.2f} W/m², Direct_clear={direct_clear:.2f}, Diffuse_clear={diffuse_clear:.2f}")

        results = []

        for cth, cot, cloud_phase in tqdm(product(CLOUD_TOP_HEIGHT, COT, CLOUD_PHASE),
            desc=f"☁ Inner loop (doy={doy}, h={hour})", leave=False):                                   
            base = cth - CLOUD_GEOGRAPHICAL_THICKNESS
            # Set base to 0 if cth-vertical extent is smaller than 0 
            base = 0.0 if base < 0.0 else base 

            # Determine cloud phase
            reff = IC_REFF if cloud_phase == "ice" else WC_REFF
            lwc = IWC if cloud_phase == "ice" else LWC

            cloud_file = f"cloudfiles/{cloud_phase}_top{cth*1000}_cot{cot:.2f}.dat"
            write_cloud_file(cloud_file, base, cth, lwc, reff)

            # Cloudy input with cloud file and modify tau550
            cloudy_input = generate_uvspec_input(
                doy, hour, albedo, profile,
                cloud_file=cloud_file,
                cloud_phase=cloud_phase,
                cot=cot
            )
            
            ghi_cloudy, direct_cloudy, diffuse_cloudy = run_uvspec(cloudy_input)

            # CAF = cloudy / clear
            caf_ghi = ghi_cloudy / ghi_clear if ghi_clear > 0 else np.nan
            caf_dir = direct_cloudy / direct_clear if direct_clear > 0 else np.nan
            caf_dif = diffuse_cloudy / diffuse_clear if diffuse_clear > 0 else np.nan
            
            #tqdm.write(f"☁️  CLOUDY SKY | Base={base} km, Type={cloud_phase}, COT={cot}")
            #tqdm.write(f"    → GHI_cloudy={ghi_cloudy:.2f} W/m², Direct_cloudy={direct_cloudy:.2f}, Diffuse_cloudy={diffuse_cloudy:.2f}")
            #print(f"📉 CAF        | CAF_GHI={caf_ghi:.3f}, CAF_Direct={caf_dir:.3f}, CAF_Diffuse={caf_dif:.3f}")

            results.append([
                profile, doy, hour, albedo, ALTITUDE, AOD_MONTHLY[doy_to_date[doy][0]],
                base, cth, cot, cloud_phase,
                ghi_clear, direct_clear, diffuse_clear,
                ghi_cloudy, direct_cloudy, diffuse_cloudy,
                caf_ghi, caf_dir, caf_dif
            ])
        
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "atmosphere", "doy", "hour", "albedo", "altitude_km",
                "aod",
                "cloud_base_km", "cloud_top_km", "cot", "cloud_phase",
                "ghi_clear", "direct_clear", "diffuse_clear",
                "ghi_cloudy", "direct_cloudy", "diffuse_cloudy",
                "caf_ghi", "caf_direct", "caf_diffuse"
            ])
            writer.writerows(results)

        print(f"✅ Saved LUT: {output_path}")



def write_cloud_file(filename, base, top, lwc, reff):
    """Save to filename the cloud parameter input file to uvspec."""
    with open(filename, "w") as f:
        f.write("# z LWC/IWC R_eff\n")
        f.write(f"{top:.3f} 0 0\n")
        f.write(f"{base:.3f} {lwc} {reff}\n")

def generate_uvspec_input(doy, hour, albedo, profile,
                          cloud_file=None, cloud_phase=None, cot=None):
    """Generate uvspec input string from parameters."""
    month = doy_to_date[doy][0]
    day = doy_to_date[doy][1]
    lines = [
        #"verbose", #disable for speed-up
        "latitude N 60.39", # Bergen latitude and longitude, combine with time to select SZA and atm_profile
        "longitude E 5.33",
        f"time 2024 {month} {day} {hour} 00 00",
        "rte_solver disort",
        f"atmosphere_file {profile}",
        "number_of_streams 6", #default is 6 for fluxes, 16 for radiances 
        "wavelength 250 2500",
        "aerosol_default",
        f"aerosol_set_tau_at_wvl 550 {AOD_MONTHLY[month]}",
        "mol_abs_param KATO",
        "output_user eglo edir edn",
        "output_process sum",
        f"albedo {albedo}",
        f"altitude {ALTITUDE}"
    ]
    if cloud_file and cloud_phase and cot:
        if cloud_phase == "ice":
            lines.append(f"ic_file 1D {cloud_file}")
            lines.append(f"ic_modify tau550 set {cot}")
        else:
            lines.append(f"wc_file 1D {cloud_file}")
            lines.append(f"wc_modify tau550 set {cot}")

    return "\n".join(lines)


def run_uvspec(input_str, out_path="output/uvspec.out", err_path="output/verbose.txt"):
    """Run uvspec routine given the input string and save results to out_path and print errors to err_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out_file, open(err_path, "w") as err_file:
        result = subprocess.run(
            ["uvspec"],
            input=input_str,
            text=True,
            stdout=out_file,
            stderr=err_file
        )

    if result.returncode != 0:
        print(f"uvspec error. See {err_path}")
        return np.nan, np.nan, np.nan

    try:
        with open(out_path, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    wl, edir, edn = map(float, line.split()[:3])
                    eglo = edir + edn
                    return eglo, edir, edn
        return np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Failed to parse output: {e}")
        return np.nan, np.nan, np.nan

def merge_LUT_files(lut_folder, output_file): 
    """Read all LUT files in lut_folder into one pd dataframe, 
    concatenate and write to one LUT file in output_file."""
    # Pattern to match filenames
    pattern = os.path.join(lut_folder, 'LUT_doy*_hod*_alb*.csv')

    # List of dataframes
    df_list = []

    for file_path in glob.glob(pattern):
        print(f"Append file {file_path}...")
        df = pd.read_csv(file_path)            
        df_list.append(df)
        
    # Merge all into one DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Merged LUT saved to: {output_file}") 


if __name__ == "__main__":
    # Read cloud properties from claas-3
    df_claas = pd.read_csv("data/processed/s2_cloud_cover_table_small_and_large_with_simulated_florida_flesland_ghi.csv")
    df_claas = df_claas[["date", "blue_sky_albedo_median", "cot_median_small", "cth_median_small", "cph_median_small"]]
    descriptive_stats_claas3(df_claas)

    # Generate LUT (runtime 32 hours)
    #generate_LUT()
    
    # Merge all lut files to one big csv
    #merge_LUT_files("data/processed/LUT/claas3", "data/processed/LUT/claas3/LUT.csv")
