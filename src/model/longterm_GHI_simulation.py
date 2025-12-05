import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm 
from scipy.stats import dirichlet
from src.model import FLORIDA_LAT, FLORIDA_LON, FLESLAND_LAT, FLESLAND_LON
from src.model.instantaneous_GHI_model import build_ghi_clear_lut
from src.model.generate_LUT import HOD_DICT


# ---------------------------------------------------------
# Clear-sky-index bootstrap samplers
# ---------------------------------------------------------
def prepare_csi_bootstrap(clear_sky_index_monthly_mixed_sky_filepath):
    """
    Loads Clear-sky index observations and returns:
        per_month[(sky_type, month)] -> array
        pooled[sky_type] -> array
    """
    df = pd.read_csv(clear_sky_index_monthly_mixed_sky_filepath)
    df["sky_type"] = df["sky_type"].astype(str)
    df["month"] = df["month"].astype(int)

    per_month = {}
    pooled = {"clear": [], "mixed": [], "overcast": []}

    for _, row in df.iterrows():
        s = row["sky_type"]
        m = int(row["month"])
        c = float(row["mean_clear_sky_index"])

        pooled[s].append(c)

        key = (s, m)
        if key not in per_month:
            per_month[key] = []
        per_month[key].append(c)

    # Convert lists to numpy arrays
    pooled = {s: np.array(vals) for s, vals in pooled.items()}
    per_month = {k: np.array(vals) for k, vals in per_month.items()}

    return per_month, pooled


def bootstrap_csi_annual(pooled_dict, sky_type):
    """Draw CSI from pooled (all-month) distribution."""
    if sky_type == "clear":
        return 1.0
    arr = pooled_dict[sky_type]
    return float(arr[np.random.randint(0, len(arr))])


def bootstrap_csi_monthly(per_month_dict, sky_type, month):
    """Draw CSI from per-month distribution."""
    if sky_type == "clear":
        return 1.0
    arr = per_month_dict.get((sky_type, month), None)
    if arr is None or len(arr) == 0:
        return np.nan
    return float(arr[np.random.randint(0, len(arr))])

# ---------------------------------------------------------
# Sky type dirichlet sampling
# ---------------------------------------------------------
def sample_dirichlet(sky_type_counts_path, n_samples):
    # Example counts per month (replace with your actual counts)
    # Each row: month, count_clear, count_mixed, count_overcast
    counts = pd.read_csv(sky_type_counts_path)

    # Choose a Dirichlet prior alpha (pseudo-counts).
    # Use alpha=1 (uniform prior) or put a small extra mass on 'clear' to avoid zeros
    alpha_prior = np.array([1.0, 1.0, 1.0])
    # If you want to nudge winter clear up a bit:
    # alpha_prior = np.array([2.0, 1.0, 1.0])

    # For each month sample probability vectors
    samples_by_month = {}
    for _, row in counts.iterrows():
        month = int(row['month'])
        obs_counts = np.array([row['clear'], row['mixed'], row['overcast']])
        posterior_alpha = obs_counts + alpha_prior
        draws = dirichlet.rvs(posterior_alpha, size=n_samples)  # shape (n_samples, 3)
        samples_by_month[month] = draws

    # Usage in simulation:
    # For each simulation year and month, draw one p-vector:
    # p_vec = samples_by_month[month][np.random.randint(0, n_samples)]
    # then draw sky type ~ Categorical(p_vec)
    return samples_by_month


# ---------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------
def simulate_annual_ghi(monthly_sky_type_count_filepath, area_mean_clear_sky_index_filepath, 
                        model="annual",
                        n_years=100, verbose = False):
    """Given monthly sky type probabilities and clear sky index for each sky type, 
    simulate n_years of the area-wide monthly and annual GHI. Returns a dataframe with one
    row per observation, including columns GHI (integral), sky type (clear/mixed/overcast),
    clear-sky index, month. """
    # Build clear sky lut 
    GHI_CLEAR_LUT = build_ghi_clear_lut()
    
    # Preload sky type sampling
    n_sky_type_samples = n_years * 50
    sky_type_probs = sample_dirichlet(monthly_sky_type_count_filepath, n_sky_type_samples)
    
    # Preload CSI bootstrapping lookups
    per_month_csi, pooled_csi = prepare_csi_bootstrap(area_mean_clear_sky_index_filepath)

    if verbose:
        print("Loaded CSI observations:")
        for key in list(per_month_csi.keys()):
            print(f"  Month {key[1]}, {key[0]}: {len(per_month_csi[key])} obs")
        print("Annual pooled sizes:", {k: len(v) for k, v in pooled_csi.items()})
        print()
    
    results = []
    month_range = range(0, 12*n_years)
    days_per_month = [31, 28, 31, 30, 31, 30,
                  31, 31, 30, 31, 30, 31]
    hod_dict_all_keys = list(HOD_DICT.keys())
            
    for month_i in tqdm(month_range, total=len(month_range), desc=f"Simulating {model} ghi for {n_years} years ... "): 
        month = (month_i % 12) + 1
        year_index = (month_i // 12) + 1
        
        # Get number of days in this month
        n_days = days_per_month[month-1] # Adjust using calendar
            
        # -------------------------------------------------------
        # Draw sky type 
        # -------------------------------------------------------
        p_vec = sky_type_probs[month][np.random.randint(0, n_sky_type_samples)]
        sky_types = np.random.choice(a=["clear", "mixed", "overcast"], size=n_days,
                                    p=p_vec)
        
        assert len(p_vec) == 3, f"There are {len(p_vec)} sky types (expected 3)."
        assert abs(sum(p_vec) - 1.0) < 10e-5, f"Sky type probabilites must add up to 1.0, but sum is {sum(p_vec)}."
        assert 0 < p_vec[0] < 0.5, f"Clear sky type probability assumes unrealistic value ({p_vec[0]})!"
        assert all(p_vec > 0), f"Sky type probabilities must not be 0: {p_vec}."
        
        if verbose: 
            print(f"---------------- Month {month} ------------- ")
            print(f"pvec: {p_vec}")
            print(f"sky types: {sky_types}")
        
        for day in range(1,n_days+1): 
            # Get doy 
            doy = sum(days_per_month[:month-1]) + day
            assert 1 <= doy <= 365, f"DOY out of range: {doy}."  
        
            sky_type = sky_types[day-1]
            
            # -----------------------------------------
            # Draw Clear-sky index using bootstrap depending on model
            # -----------------------------------------
            if model == "annual":
                k = bootstrap_csi_annual(pooled_csi, sky_type)
            elif model == "monthly":
                k = bootstrap_csi_monthly(per_month_csi, sky_type, month)             
            else:
                raise ValueError(f"Unknown model: {model}")

            if verbose:
                print(f" --- Day {doy}, sky={sky_type}, k={k:.3f} --- ")
                
            assert 0 <= k <= 1.0, f"Clear-sky index has unexpected value: {k}. "
            assert k is not None, f"Clear-sky index must not be None. "
            if sky_type == "clear": 
                assert abs(k - 1.0) < 10e-5, f"Clear-sky index for clear-days should be 1.0, but is {k}."
            
            # -------------------------------------
            # Get Sunshine hours 
            # -------------------------------------
            closest_key = min(hod_dict_all_keys, key=lambda i: abs(i - doy))
            assert 1 <= closest_key <= 365, f"Closest key out of range: {closest_key}"
            # Get hours between sunset and sunrise from HOD_DICT
            hours = HOD_DICT[closest_key]
            assert 5 <= len(hours) <= 21, f"Unexpected value of hours of sunshine per day: {len(hours)}."
            assert all(1 < h < 22 for h in hours), f"Unexpected value of hour with sunshine: {hours}."
            
            if verbose: 
                print(f"Hours of doy {doy}, closest doy: {closest_key}: {hours}")
            
            # Init spatially uniform ghi
            daily_GHI_Wh = 0.0
                
            # -----------------------------------------------------    
            # Compute hourly GHI 
            # -----------------------------------------------------
            clear_hours = GHI_CLEAR_LUT[doy-1, hours]
            daily_GHI_Wh = np.nansum(clear_hours * k)
            
            if verbose: 
                print(f"clear_hours: {clear_hours}")
                print(f"daily_ghi_wh: ", daily_GHI_Wh)
                     
            # Calculate daily mean GHI 
            daily_GHI_mean = daily_GHI_Wh / 24
            
            assert daily_GHI_Wh is not None, f"Mean daily GHI is None: {year_index}-{doy}."
            assert 0 < daily_GHI_mean < 10000, f"Unrealistic value for daily GHI: {daily_GHI_mean}."
            
            if verbose: 
                print(f"Daily GHI Wh/m²: {daily_GHI_Wh:.1f}, Mean GHI: {daily_GHI_mean:.1f}.")
            
            # ------------------------------------
            # Append to results 
            # ------------------------------------
            results.append({
                "day_count": day,
                "doy": doy,
                "month": month,
                "year": year_index,
                "GHI_daily_Wh": daily_GHI_Wh,
                "GHI_daily_mean_Wh": daily_GHI_mean,
                "k": k,
                "sky_type": sky_type
            })

    return pd.DataFrame(results)
    
    
    
def spatially_resolved_model(longterm_sim_results_spatially_uniform_filepath, clear_sky_index_monthly_mixed_sky_filepath, outpath_nc, 
                             verbose = False):
    """Multiply monthly results times normalized monthly clear-sky index maps (for mixed sky conditions). Keep 
    the results for overcast and clear sky conditions. Output: 12 maps (one for each month) with 
    clear-sky index for mixed-sky and all-sky."""
    
    longterm_sim_df = pd.read_csv(longterm_sim_results_spatially_uniform_filepath)
    longterm_sim_df["GHI_daily_kWh"] = longterm_sim_df["GHI_daily_Wh"] / 1000
    
    # Load monthly clear-sky index maps 
    ds = xr.open_dataset(clear_sky_index_monthly_mixed_sky_filepath)
    # Import variable 
    k_map = ds["clear_sky_index"].isel(month=0).values 
    k_map_size1, k_map_size2 = k_map.shape 
    lat = ds["lat"].values 
    lon = ds["lon"].values # Find nearest index for location 
                
    # Remove old file 
    if os.path.exists(outpath_nc): 
        os.remove(outpath_nc)

    # init (month, lat, lon) arrays 
    mixed_sky_ghi = np.zeros((12, k_map_size1, k_map_size2), dtype=np.float32) 
    all_sky_ghi = np.zeros((12, k_map_size1, k_map_size2), dtype=np.float32)
    monthly_counts_mixed = np.zeros(12, dtype=np.int32)
    monthly_counts_all_sky = np.zeros(12, dtype=np.int32)
    
    # create new dataset 
    out_ds = xr.Dataset( { "mixed_sky_ghi": (("month", "lat", "lon"), mixed_sky_ghi), 
                          "all_sky_ghi": (("month", "lat", "lon"), all_sky_ghi), }, 
                        coords={ "month": np.arange(1, 13), 
                                "lat": lat, 
                                "lon": lon } )
    
    # Pre-compute grouping 
    grouped = longterm_sim_df.groupby("month")
    
    for month in range(1,13): 
        print(f"\n--- Processing month {month} ---")
        
        # Normalize monthly sky maps 
        monthly_map_k = ds["clear_sky_index"].isel(month=month-1).values 
        monthly_area_mean_k = monthly_map_k.mean() 
        monthly_map_k_norm = monthly_map_k/monthly_area_mean_k 
        assert np.all(0 < monthly_map_k_norm), f"Some values in k_map are out of range (0-infty): {monthly_map_k_norm}." 
        print(f"k_map norm min: {np.min(monthly_map_k_norm):.3f}," 
              f"norm max: {np.max(monthly_map_k_norm):.3f}, mean={monthly_map_k_norm.mean():.3f}")

        # -----------------------------------
        # Monthly sky ghi  - Mixed sky 
        # -----------------------------------

        # Get monthly aggregated data 
        df_m = grouped.get_group(month)
        
        # Mask once
        mask_m = (df_m["sky_type"] == "mixed")
        mask_c = (df_m["sky_type"] == "clear")
        mask_o = (df_m["sky_type"] == "overcast")

        # Get counts per sky type
        n_m = mask_m.sum()
        n_c = mask_c.sum()
        n_o = mask_o.sum()
        
        # Count monthly mixed-sky days
        monthly_counts_mixed[month - 1] = n_m

        # Monthly mean GHI for mixed sky
        mean_ghi_m = df_m.loc[mask_m, "GHI_daily_kWh"].mean() if n_m > 0 else 0.0

        # Spatial mixed-sky GHI pattern
        mixed_map = mean_ghi_m * monthly_map_k_norm

        # Store into dataset
        out_ds["mixed_sky_ghi"].loc[dict(month=month)] = mixed_map.astype(np.float32)

        if verbose:
            print(f"  Mixed sky: mean_GHI={mean_ghi_m:.1f} Wh, days={n_m}")

        
        # --------------------------------------------------
        # Spatially resolved ghi  - All sky weighted mean
        # --------------------------------------------------

        # Mean daily GHI for each sky type
        mean_ghi_c = df_m.loc[mask_c, "GHI_daily_kWh"].mean() if n_c > 0 else 0.0
        mean_ghi_o = df_m.loc[mask_o, "GHI_daily_kWh"].mean() if n_o > 0 else 0.0

        total_obs = n_m + n_c + n_o
        monthly_counts_all_sky[month - 1] = total_obs

        assert 28*n_years <= total_obs <= 31 * n_years, f"Unexpected value for total obs: {total_obs}, expected {28*n_years}-{31*n_years}."
        
        # Weighted mean:
        # (M*n_m + C*n_c + O*n_o) / N_total
        # M is spatial map; C, O uniform scalars
        allsky_map = (
            mixed_map * n_m +
            mean_ghi_c * n_c +
            mean_ghi_o * n_o
        ) / total_obs

        # Store
        out_ds["all_sky_ghi"].loc[dict(month=month)] = allsky_map.astype(np.float32)

        if verbose:
            print(f"  All-sky: mixed={n_m}, clear={n_c}, overcast={n_o}, total={total_obs}")
            print(f"  mean clear={mean_ghi_c:.1f} Wh, mean overcast={mean_ghi_o:.1f} Wh")
            print(f" Mean all-sky: {allsky_map.mean():.1f}")
        
    # ------------------------------------------------------
    # Add metadata / units to the variables & coordinates
    # ------------------------------------------------------
    out_ds["month_mixed_count"] = (("month",), monthly_counts_mixed)
    out_ds["month_all_sky_count"] = (("month",), monthly_counts_all_sky)

    out_ds["month_mixed_count"].attrs.update({
        "units": "count",
        "long_name": "Number of mixed-sky observations per month over all simulation years",
        "description": "Total number of simulated mixed-sky GHI observations contributing to each month's statistics"
    })
    
    out_ds["month_all_sky_count"].attrs.update({
        "units": "count",
        "long_name": "Number of daily (all-sky) observations per month over all simulation years",
        "description": "Total number of simulated daily (all-sky) GHI observations contributing to each month's statistics"
    })

    out_ds["mixed_sky_ghi"].attrs.update({
        "units": "kWh m-2",
        "long_name": "Spatially resolved daily GHI for mixed-sky conditions",
        "description": "Mean mixed-sky GHI multiplied by normalized clear-sky index map"
    })

    out_ds["all_sky_ghi"].attrs.update({
        "units": "kWh m-2",
        "long_name": "Spatially resolved daily GHI for all-sky conditions",
        "description": "Weighted combination of mixed, clear, and overcast GHI using spatial pattern from mixed sky"
    })

    out_ds["lat"].attrs.update({
        "units": "degrees_north"
    })

    out_ds["lon"].attrs.update({
        "units": "degrees_east"
    })

    out_ds["month"].attrs.update({
        "units": "1–12",
        "long_name": "Month number"
    })

    # Save netCDF
    print(f"\nWriting output to {outpath_nc} ...")
    out_ds.to_netcdf(outpath_nc)
    print("Done.")
    
    


def spatially_resolved_simulation_timeseries(
        simulation_results_csv,
        clear_sky_index_monthly_filepath,
        verbose=False
    ):
    """
    Fast version:
      - Preloads 12×2 pixel k-values (Florida, Flesland)
      - No per-day append
      - Sums and means per sky type
    """

    # ----------------------------------------------------------
    # Load simulation dataframe
    # ----------------------------------------------------------
    df = pd.read_csv(simulation_results_csv)
    df["GHI_daily_kWh"] = df["GHI_daily_Wh"] / 1000

    # ----------------------------------------------------------
    # Load k-maps dataset
    # ----------------------------------------------------------
    ds = xr.open_dataset(clear_sky_index_monthly_filepath)
    lat = ds["lat"].values
    lon = ds["lon"].values
    k_monthly = ds["clear_sky_index"].values   # shape: (12, lat, lon)

    # ----------------------------------------------------------
    # Locate stations
    # ----------------------------------------------------------
    Florida_ilat = np.abs(lat - FLORIDA_LAT).argmin() 
    Florida_ilon = np.abs(lon - FLORIDA_LON).argmin() 
    Flesland_ilat = np.abs(lat - FLESLAND_LAT).argmin() 
    Flesland_ilon = np.abs(lon - FLESLAND_LON).argmin()

    if verbose:
        print(f"Florida pixel:  ({Florida_ilat}, {Florida_ilon})")
        print(f"Flesland pixel: ({Flesland_ilat}, {Flesland_ilon})")

    # ----------------------------------------------------------
    # Precompute k_norm values for each month (12 × 2)
    # ----------------------------------------------------------
    k_station = np.zeros((12, 2))  # month index 0..11, station 0=Florida,1=Flesland

    for m in range(12):
        k_map = k_monthly[m]
        k_norm = k_map / k_map.mean()

        k_station[m, 0] = float(k_norm[Florida_ilat, Florida_ilon])
        k_station[m, 1] = float(k_norm[Flesland_ilat, Flesland_ilon])

    # ----------------------------------------------------------
    # Prepare looping
    # ----------------------------------------------------------
    years = sorted(df["year"].unique())
    grouped_year = df.groupby("year")

    results = []
    pbar = tqdm(total=len(years) * 12, desc="Processing (fast version)")

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    for year in years:

        df_year = grouped_year.get_group(year)
        df_year_month = df_year.groupby("month")

        for month in range(1, 13):

            pbar.update(1)

            if month not in df_year_month.groups:
                continue

            df_m = df_year_month.get_group(month)

            # Masks
            mask_m = df_m["sky_type"] == "mixed"
            mask_c = df_m["sky_type"] == "clear"
            mask_o = df_m["sky_type"] == "overcast"

            # Observations per sky type
            n_m = int(mask_m.sum())
            n_c = int(mask_c.sum())
            n_o = int(mask_o.sum())
            n_all = n_m + n_c + n_o

            # Extract arrays
            ghi = df_m["GHI_daily_Wh"].values

            ghi_m = ghi[mask_m]
            ghi_c = ghi[mask_c]
            ghi_o = ghi[mask_o]

            # Station-specific k factors
            kF = k_station[month-1, 0]  # Florida
            kS = k_station[month-1, 1]  # Flesland

            # ------------------------------------------------------
            # Compute aggregated (SUM and MEAN)
            # ------------------------------------------------------

            # --- Mixed ---
            fl_m_sum = (ghi_m * kF).sum()
            fs_m_sum = (ghi_m * kS).sum()

            # --- Clear ---
            fl_c_sum = ghi_c.sum()
            fs_c_sum = ghi_c.sum()

            # --- Overcast ---
            fl_o_sum = ghi_o.sum()
            fs_o_sum = ghi_o.sum()

            # --- All-sky ---
            fl_all_sum = fl_m_sum + fl_c_sum + fl_o_sum
            fs_all_sum = fs_m_sum + fs_c_sum + fs_o_sum

            # Means
            fl_m_mean = fl_m_sum / n_m if n_m else np.nan
            fs_m_mean = fs_m_sum / n_m if n_m else np.nan

            fl_c_mean = fl_c_sum / n_c if n_c else np.nan
            fs_c_mean = fs_c_sum / n_c if n_c else np.nan

            fl_o_mean = fl_o_sum / n_o if n_o else np.nan
            fs_o_mean = fs_o_sum / n_o if n_o else np.nan

            fl_all_mean = fl_all_sum / n_all if n_all else np.nan
            fs_all_mean = fs_all_sum / n_all if n_all else np.nan

            # ------------------------------------------------------
            # Append a single row per sky type
            # ------------------------------------------------------
            results.append({
                "year": year,
                "month": month,
                "sky_type": "mixed",
                "n_clear": n_c, "n_mixed": n_m, "n_overcast": n_o,
                "Florida_GHI_daily_Wh_sum": fl_m_sum,
                "Florida_GHI_daily_Wh_mean": fl_m_mean,
                "Flesland_GHI_daily_Wh_sum": fs_m_sum,
                "Flesland_GHI_daily_Wh_mean": fs_m_mean
            })

            results.append({
                "year": year,
                "month": month,
                "sky_type": "clear",
                "n_clear": n_c, "n_mixed": n_m, "n_overcast": n_o,
                "Florida_GHI_daily_Wh_sum": fl_c_sum,
                "Florida_GHI_daily_Wh_mean": fl_c_mean,
                "Flesland_GHI_daily_Wh_sum": fs_c_sum,
                "Flesland_GHI_daily_Wh_mean": fs_c_mean
            })

            results.append({
                "year": year,
                "month": month,
                "sky_type": "overcast",
                "n_clear": n_c, "n_mixed": n_m, "n_overcast": n_o,
                "Florida_GHI_daily_Wh_sum": fl_o_sum,
                "Florida_GHI_daily_Wh_mean": fl_o_mean,
                "Flesland_GHI_daily_Wh_sum": fs_o_sum,
                "Flesland_GHI_daily_Wh_mean": fs_o_mean
            })

            results.append({
                "year": year,
                "month": month,
                "sky_type": "all-sky",
                "n_clear": n_c, "n_mixed": n_m, "n_overcast": n_o,
                "Florida_GHI_daily_Wh_sum": fl_all_sum,
                "Florida_GHI_daily_Wh_mean": fl_all_mean,
                "Flesland_GHI_daily_Wh_sum": fs_all_sum,
                "Flesland_GHI_daily_Wh_mean": fs_all_mean
            })

    pbar.close()

    return pd.DataFrame(results)

    
    
def describe_resuts(results): 
    print(results.head())
    
    # Convert daily GHI from Wh/m² → kWh/m²
    results["GHI_daily_kWh"] = results["GHI_daily_Wh"] / 1000
    results["GHI_daily_mean_kWh"] = results["GHI_daily_mean_Wh"] / 1000

    # ---------------------------
    # 1. Summary per year
    # ---------------------------
    yearly_stats = results.groupby("year").agg(
        mean_daily_GHI_kWh=("GHI_daily_kWh", "mean"),
        std_daily_GHI_kWh=("GHI_daily_kWh", "std")
    ).reset_index()

    print("Yearly GHI statistics (kWh/m²/day):")
    print(yearly_stats)

    # ---------------------------
    # 2. Summary per month (aggregated over all years)
    # ---------------------------
    month_stats = results.groupby("month").agg(
        mean_daily_GHI_kWh=("GHI_daily_kWh", "mean"),
        std_daily_GHI_kWh=("GHI_daily_kWh", "std"),
        mean_k=("k", "mean"),
        std_k=("k", "std"),
        n_clear=("sky_type", lambda x: sum(np.array(x) == "clear")),
        n_mixed=("sky_type", lambda x: sum(np.array(x) == "mixed")),
        n_overcast=("sky_type", lambda x: sum(np.array(x) == "overcast")),
        n_days=("doy", "count")
    ).reset_index()

    print("\nMonthly aggregated statistics over all years:")
    print(month_stats)
   

if __name__ == "__main__": 
    monthly_sky_type_counts_filepath = "data/processed/monthly_sky_type_counts.csv"
    area_mean_clear_sky_index_filepath = "data/processed/area_mean_clear_sky_index_per_obs.csv"
    clear_sky_index_monthly_mixed_sky_filepath = "data/processed/simulated_clear_sky_index_monthly_mixed_sky.nc"
    ghi_monthly_spatial_longterm_sim_outpath = f"data/processed/longterm_ghi_spatially_resolved_monthly.nc"
    
    n_years = 5000
    model = "monthly"
    
    # Run simulation
    #results = simulate_annual_ghi(monthly_sky_type_counts_filepath, area_mean_clear_sky_index_filepath,
    #                    model=model, n_years=n_years, 
    #                    verbose=False)
    ## Save to outpath
    outpath = f"data/processed/longterm_sim_ghi_{n_years}_k={model}.csv"
    #results.to_csv(outpath, index=False)
    #results = pd.read_csv(outpath)
    
        
    # ----------------------------------------------------------
    # Spatially resolved simulation monthly 
    # ----------------------------------------------------------
    #spatially_resolved_model(outpath, clear_sky_index_monthly_mixed_sky_filepath,
    #                         ghi_monthly_spatial_longterm_sim_outpath, verbose=True)
    
    # ---------------------------------------------------------
    # Florida Flesland pixels monthly timeseries
    # --------------------------------------------------------
    results_df = spatially_resolved_simulation_timeseries(outpath, 
                                                          clear_sky_index_monthly_mixed_sky_filepath, 
                                                          verbose = True)
    outpath_fl_fs_monthly = f"data/processed/longterm_sim_Florida_Flesland_monthly_pixels_{n_years}_k={model}.csv"
    results_df.to_csv(outpath_fl_fs_monthly, index=False)

    
