import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm 
from scipy.stats import dirichlet
from src.model.surface_GHI_model import get_closest_lut_entry
from src.model import FLORIDA_LAT, FLORIDA_LON, FLESLAND_LAT, FLESLAND_LON
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
                        LUT_filepath,
                        clear_sky_index_monthly_mixed_sky_filepath=None, 
                        ghi_spatially_resolved_outpath = None,
                        model="annual",
                        n_years=100, verbose = False):
    """Given monthly sky type probabilities and clear sky index for each sky type, 
    simulate n_years of the area-wide monthly and annual GHI. Returns a dataframe with one
    row per observation, including columns GHI (integral), sky type (clear/mixed/overcast),
    clear-sky index, month. """
    
    # Can only do spatial simulations if both k-maps-input and outpath are given as parameters
    if clear_sky_index_monthly_mixed_sky_filepath is None: 
        assert ghi_spatially_resolved_outpath is None, "Please provide monthly maps of k as input."
    if ghi_spatially_resolved_outpath is None: 
        assert clear_sky_index_monthly_mixed_sky_filepath is None, "Please provide an outpath for the spatially resolved simulation results."
    
    lut = pd.read_csv(LUT_filepath)
    variables = ["doy", "hour", "albedo", "altitude_km", "cloud_top_km", "cot", "cloud_phase"]
    unique_values = {var: lut[var].unique() for var in variables if var in lut.columns}
    surface_albedo = 0.129
    altitude = 0.08 
    
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
    
    if clear_sky_index_monthly_mixed_sky_filepath is not None: 
        # Open .nc Dataset
        ds = xr.open_dataset(clear_sky_index_monthly_mixed_sky_filepath)
        # Import variable 
        k_map = ds["clear_sky_index"].isel(month=0).values
        k_map_size1, k_map_size2 = k_map.shape
        lat = ds["lat"].values
        lon = ds["lon"].values
        # Find nearest index for location
        Florida_ilat = np.abs(lat - FLORIDA_LAT).argmin()
        Florida_ilon = np.abs(lon - FLORIDA_LON).argmin()
        Flesland_ilat = np.abs(lat - FLESLAND_LAT).argmin()
        Flesland_ilon = np.abs(lon - FLESLAND_LON).argmin()
        
        if verbose: 
            print(f"Florida pixel location: x={Florida_ilat}, y={Florida_ilon}")
            print(f"Flesland pixel location: x={Flesland_ilat}, y={Flesland_ilon}")
        
    if ghi_spatially_resolved_outpath is not None: 
        # Remove old file
        if os.path.exists(ghi_spatially_resolved_outpath):
            os.remove(ghi_spatially_resolved_outpath)

        # init (month, lat, lon) arrays
        monthly_sum_GHI = np.zeros((12, k_map_size1, k_map_size2), dtype=np.float32)
        monthly_count   = np.zeros(12, dtype=np.int32)
        
        # create new dataset
        out_ds = xr.Dataset(
            {
                "monthly_mean_GHI": (("month", "lat", "lon"), monthly_sum_GHI),
                "monthly_count": (("month"), monthly_count),
            },
            coords={
                "month": np.arange(1, 13),
                "lat": lat,
                "lon": lon
            }
        )
        
    values_out_of_range = 0
    for month_i in tqdm(month_range, total=len(month_range), desc=f"Simulating annual ghi for {n_years} years ... "): 
        month = (month_i % 12) + 1
        year_index = (month_i // 12) + 1
        
        # Get number of days in this month
        n_days = days_per_month[month-1] # Adjust using calendar
        
        # Pre-load monthly map k 
        if clear_sky_index_monthly_mixed_sky_filepath is not None:
            monthly_map_k = ds["clear_sky_index"].isel(month=month-1).values
            # Pre-compute area-mean only once per month
            monthly_area_mean_k = monthly_map_k.mean()
            monthly_map_k_norm = monthly_map_k/monthly_area_mean_k
            assert np.all(0 < monthly_map_k_norm), f"Some values in k_map are out of range (0-infty): {monthly_map_k_norm}."
            print(f"k_map norm min: {np.min(monthly_map_k_norm):.3f}, norm max: {np.max(monthly_map_k_norm):.3f}")

            
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
            # Get spatially resolved k map 
            # -------------------------------------
            if clear_sky_index_monthly_mixed_sky_filepath is not None: 
                if sky_type == "mixed": 
                    # Calculate daily error 
                    diff_k = monthly_area_mean_k - k 
                    print(f"Daily diff k: {diff_k:.3f} (Monthly area mean k={monthly_area_mean_k:.3f})")
                    # Add daily error to mean monthly map
                    k_map = monthly_map_k_norm * k 
                    min_k = np.min(k_map)
                    max_k = np.max(k_map)
                    print(f"k_map min: {min_k:.3f}, max: {max_k:.3f}")
                    if (min_k < 0) or (max_k > 1.0): 
                        values_out_of_range += 1
                        tqdm.write(f"Some values in k_map are out of range (0-1]: min {min_k}, max {max_k}.")
                    k_map = np.clip(k_map, 0.00001, 1.0) # clip to values 0 and 1.0
                elif sky_type == "clear": 
                    k_map = np.ones(shape=(k_map_size1, k_map_size2)) # uniform map of 1.0
                elif sky_type == "overcast": 
                    k_map = np.full(shape=(k_map_size1, k_map_size2), fill_value=k) # uniform map of k

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
            
            if clear_sky_index_monthly_mixed_sky_filepath is not None: 
                # Spatially resolved k 
                daily_GHI_Wh = np.zeros(shape=(k_map_size1, k_map_size2), dtype=np.float32) 
            else: 
                # Spatially uniform k 
                daily_GHI_Wh = 0.0
                
            # -----------------------------------------------------    
            # Compute hourly GHI 
            # -----------------------------------------------------
            for hour in hours:
                # Get clear-sky irradiance for this day and time from LUT
                res_dict = get_closest_lut_entry(lut, unique_values, doy, hour, surface_albedo, altitude)
                
                if res_dict["direct_clear"] is not None and res_dict["diffuse_clear"] is not None: 
                    ghi_clear = res_dict["direct_clear"] + res_dict["diffuse_clear"]
                else : 
                    # Skip if hour is outside of sunrise/sunset
                    continue
                
                # -------------------------------------------------
                # Apply clear-sky index
                # -------------------------------------------------
                if clear_sky_index_monthly_mixed_sky_filepath is not None: 
                    # Calculate GHI_hourly as array
                    GHI_hourly = ghi_clear * k_map   
                else : 
                    GHI_hourly = ghi_clear * k  # Hourly mean in W/m²
                    
                daily_GHI_Wh += GHI_hourly * 1.0 # "convert" to Wh/m²
                
            # Calculate daily mean GHI 
            daily_GHI_mean = daily_GHI_Wh / 24
            
            # Update monthly spatially resolved mean
            if ghi_spatially_resolved_outpath is not None:
                monthly_sum_GHI[month - 1] += daily_GHI_mean
                monthly_count[month - 1] += 1
         
            if clear_sky_index_monthly_mixed_sky_filepath is not None:
                # spatial model
                Florida_val  = daily_GHI_Wh[Florida_ilat, Florida_ilon] 
                Flesland_val = daily_GHI_Wh[Flesland_ilat, Flesland_ilon]
                Florida_mean = daily_GHI_mean[Florida_ilat, Florida_ilon]
                Flesland_mean = daily_GHI_mean[Flesland_ilat, Flesland_ilon]
                model_type = "spatial"
            else:
                # scalar
                Florida_val = Flesland_val = daily_GHI_Wh
                Florida_mean = Flesland_mean = daily_GHI_mean
                model_type = model   # "annual" or "monthly"

            assert Florida_val is not None, f"Mean daily GHI is None: {year_index}-{doy}."
            assert 0 < Florida_mean < 10000, f"Unrealistic value for daily GHI: {Florida_mean}."
            
            if verbose: 
                print(f"Daily GHI Wh/m²: {Florida_val:.1f}, Mean GHI: {Florida_mean:.1f}.")
            
            # ------------------------------------
            # Append to results 
            # ------------------------------------
            results.append({
                "day_count": day,
                "doy": doy,
                "month": month,
                "year": year_index,
                "Florida_GHI_daily_Wh": Florida_val,
                "Flesland_GHI_daily_Wh": Flesland_val,
                "Florida_GHI_daily_mean_Wh": Florida_mean,
                "Flesland_GHI_daily_mean_Wh": Flesland_mean,
                "k": k,
                "sky_type": sky_type,
                "model": model_type
            })
        
    if verbose: 
        total_days = n_years * 365
        percentage_out_of_range = values_out_of_range / total_days
        print(f"Number of values out of range: {values_out_of_range} ({percentage_out_of_range:.3f} %).")       
    # End of simulation loop
    # Write to netcdf 
    out_ds.to_netcdf(ghi_spatially_resolved_outpath)

    return pd.DataFrame(results)
    
    

def plot_uncertainty_stacked(samples_by_month):
    months = sorted(samples_by_month.keys())

    clear_mean, mixed_mean, over_mean = [], [], []
    clear_sd, mixed_sd, over_sd = [], [], []

    for m in months:
        draws = samples_by_month[m]
        clear_mean.append(draws[:,0].mean())
        mixed_mean.append(draws[:,1].mean())
        over_mean.append(draws[:,2].mean())

        clear_sd.append(draws[:,0].std())
        mixed_sd.append(draws[:,1].std())
        over_sd.append(draws[:,2].std())

    # Start building the stacked barplot
    x = np.arange(len(months))

    plt.figure(figsize=(12,6))

    # Clear (bottom layer)
    plt.bar(x, clear_mean, 
            yerr=clear_sd, capsize=3, 
            label="Clear", color="#4c72b0")

    # Mixed (shifted up by clear mean)
    plt.bar(x, mixed_mean, bottom=clear_mean,
            yerr=mixed_sd, capsize=3, 
            label="Mixed", color="#55a868")

    # Overcast (shifted up by clear+mixed means)
    bottom_for_over = np.array(clear_mean) + np.array(mixed_mean)
    plt.bar(x, over_mean, bottom=bottom_for_over,
            yerr=over_sd, capsize=3,
            label="Overcast", color="#c44e52")

    plt.xticks(x, months)
    plt.xlabel("Month")
    plt.ylabel("Posterior Probability (mean ± sd)")
    plt.title("Posterior Uncertainty of Monthly Sky Type Probabilities (Dirichlet)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    monthly_sky_type_counts_filepath = "data/processed/monthly_sky_type_counts.csv"
    area_mean_clear_sky_index_filepath = "data/processed/area_mean_clear_sky_index_per_obs.csv"
    clear_sky_index_monthly_mixed_sky_filepath = "data/processed/simulated_clear_sky_index_monthly_mixed_sky.nc"
    lut_filepath = "data/processed/LUT/claas3/LUT.csv" # new LUT with Claas3 cloud properties
    ghi_monthly_spatial_longterm_sim_outpath = "data/processed/simulated_ghi_longterm_monthly.nc"
    
    n_years = 1
    model = "monthly"
    
    # Run simulation
    results = simulate_annual_ghi(monthly_sky_type_counts_filepath, area_mean_clear_sky_index_filepath,
                        lut_filepath, clear_sky_index_monthly_mixed_sky_filepath, 
                        ghi_spatially_resolved_outpath=ghi_monthly_spatial_longterm_sim_outpath,
                        model=model, n_years=n_years, 
                        verbose=True)
    # Save to outpath
    outpath = f"data/processed/longterm_sim_ghi_{n_years}_k={model}.csv"
    results.to_csv(outpath, index=False)
    
    # ---------------- Describe results ----------------
    print(results.head())
    
    # Convert daily GHI from Wh/m² → kWh/m²
    results["GHI_daily_kWh"] = results["Florida_GHI_daily_Wh"] / 1000
    results["GHI_daily_mean_kWh"] = results["Florida_GHI_daily_mean_Wh"] / 1000

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
