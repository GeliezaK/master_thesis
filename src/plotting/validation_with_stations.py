import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import pvlib
from pvlib.location import Location
from sklearn.metrics import r2_score
from src.model.surface_GHI_model import CENTER_LAT, CENTER_LON, get_closest_lut_entry

def print_basic_info(sim_vs_obs):
    # Correlations 
    sub_merged = sim_vs_obs[[
        "date",
        "sky_type",
        "Florida_ghi_sim",
        "Flesland_ghi_sim",
        "Florida_ghi_1M",
        "Flesland_ghi_1M"
    ]]
    # Basic info
    print("Columns:", list(sim_vs_obs))
    print("\nHead of subset:")
    print(sub_merged.head())

    print("\nDescribe:")
    print(sub_merged.describe())

    # Missing values per column
    print("\nMissing values per column:")
    print(sub_merged.isna().sum())

    # Correlations
    print("\nCorrelations:")
    corrs = {
        "Florida_sim_vs_obs": sub_merged["Florida_ghi_sim"].corr(sub_merged["Florida_ghi_1M"]),
        "Flesland_sim_vs_obs": sub_merged["Flesland_ghi_sim"].corr(sub_merged["Flesland_ghi_1M"]),
        "Florida_vs_Flesland_sim": sub_merged["Florida_ghi_sim"].corr(sub_merged["Flesland_ghi_sim"]),
        "Florida_vs_Flesland_obs": sub_merged["Florida_ghi_1M"].corr(sub_merged["Flesland_ghi_1M"]),
    }
    for k, v in corrs.items():
        print(f"{k}: {v}")
        

def scatter_with_fit(x, y, xlabel="", ylabel="", title="", outpath="output/scatterplot.png", 
    ax=None, show_stats=True, sky_type=None):
    """
    Scatter plot with 1:1 dashed line, optional R² and correlation display.
    Optionally color points by sky type (categorical variable).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if sky_type is None:
        # Plain scatter
        ax.scatter(x, y, alpha=0.7, label="Data")
    else:
        # Color by sky type
        categories = np.unique(sky_type[~pd.isna(sky_type)])
        for cat in categories:
            mask = (sky_type == cat) & ~np.isnan(x) & ~np.isnan(y)
            ax.scatter(
                x[mask], y[mask], alpha=0.7, label=str(cat)
            )

    # 1:1 dashed line
    min_val = min(np.nanmin(x), np.nanmin(y))
    max_val = max(np.nanmax(x), np.nanmax(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal 1:1")

    # Compute stats
    if show_stats:
        mask_all = ~np.isnan(x) & ~np.isnan(y)
        r2_all = r2_score(x[mask_all], y[mask_all])
        corr_all = np.corrcoef(x[mask_all], y[mask_all])[0, 1]

        stats_text = f"Overall:\nR²={r2_all:.2f}\nCorr={corr_all:.2f}"

        if sky_type is not None:
            stats_text += "\n\nBy sky type:"
            for cat in categories:
                mask = (sky_type == cat) & mask_all
                if np.sum(mask) > 1:  # at least 2 points
                    r2 = r2_score(x[mask], y[mask])
                    corr = np.corrcoef(x[mask], y[mask])[0, 1]
                    stats_text += f"\n{cat}: R²={r2:.2f}, Corr={corr:.3f}"

        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            fontsize=9
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Save scatter plot to {outpath}.")


def lineplot_doy(x, y1, y2=None, xlabel="DOY", ylabel="", title="", 
                 outpath="output/lineplot.png", labels=("sim", "obs"), ax=None):
    """Plot DOY vs variable. Optionally plot two lines (sim and obs). Only plot points where both are not NaN."""
    
    x = np.array(x)
    y1 = np.array(y1)
    
    if y2 is not None:
        y2 = np.array(y2)
        # Keep only entries where both y1 and y2 are not NaN
        mask = ~np.isnan(y1) & ~np.isnan(y2)
        x_plot = x[mask]
        y1_plot = y1[mask]
        y2_plot = y2[mask]
    else:
        # Keep only non-NaN y1
        mask = ~np.isnan(y1)
        x_plot = x[mask]
        y1_plot = y1[mask]
        y2_plot = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(x_plot, y1_plot, marker='o', linestyle='-', label=labels[0])
    if y2_plot is not None:
        ax.plot(x_plot, y2_plot, marker='x', linestyle=':', label=labels[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.savefig(outpath)
    print(f"Saved lineplot to {outpath}.")
    

def get_toa_irradiance(lat, lon, datetime):
    """Get extraterrestrial (top of atmosphere) irradiance [W/m²] for given lat/lon/time"""
    site = Location(lat, lon, 'Europe/Oslo', 12, 'Bergen')
    solpos = site.get_solarposition(datetime)

    # Extra-terrestrial radiation normal to sun
    I0_normal = pvlib.irradiance.get_extra_radiation(datetime.dayofyear)

    # Project onto horizontal plane
    I0_horizontal = I0_normal * np.maximum(0, np.sin(np.radians(solpos['apparent_elevation'])))

    return I0_horizontal.values[0]

def flag_observations(sim_vs_obs, lut_path):
    # Get theoretical maximum irradiance 
    sim_vs_obs["datetime"] = pd.to_datetime(
        sim_vs_obs["system:time_start_large"], unit="ms", utc=True
    )
    
    sim_vs_obs["TOA_irradiance"] = sim_vs_obs.apply(
        lambda row: get_toa_irradiance(CENTER_LAT, CENTER_LON, row["datetime"]),
        axis=1
    )
    
    # Set default flag "OK", then "T" if irradiance larger than Top of Atmosphere irradiance
    sim_vs_obs["Flesland_flag"] = np.where(
        sim_vs_obs["Flesland_ghi_1M"] > sim_vs_obs["TOA_irradiance"], "T", "OK"
    )

    sim_vs_obs["Florida_flag"] = np.where(
        sim_vs_obs["Florida_ghi_1M"] > sim_vs_obs["TOA_irradiance"], "T", "OK"
    )

    # Count how many "T" flags per station
    print("Flesland flagged:", (sim_vs_obs["Flesland_flag"] == "T").sum())
    print("Florida flagged:", (sim_vs_obs["Florida_flag"] == "T").sum())
    
    # Update Flesland_flag to "N" if Flesland_ghi_1M < 0
    sim_vs_obs["Flesland_flag"] = np.where(
        sim_vs_obs["Flesland_ghi_1M"] < 0, "N", sim_vs_obs["Flesland_flag"]
    )

    # Update Florida_flag to "N" if Florida_ghi_1M < 0
    sim_vs_obs["Florida_flag"] = np.where(
        sim_vs_obs["Florida_ghi_1M"] < 0, "N", sim_vs_obs["Florida_flag"]
    )

    # Count how many "N" flags per station
    print("Flesland negative flagged:", (sim_vs_obs["Flesland_flag"] == "N").sum())
    print("Florida negative flagged:", (sim_vs_obs["Florida_flag"] == "N").sum())
    
    # Now flag any values that are Remove data with GHI > f ∗ ICS + a, where
    # f = 2, a = 0 if ICS ≤ 100 W /m2
    # f = 1.05, a = 95 if ICS > 100 W /m
    # Get clear-sky value from lut 
    # Compute DOY and hour from datetime
    sim_vs_obs["DOY"] = sim_vs_obs["datetime"].dt.dayofyear
    sim_vs_obs["Hour"] = sim_vs_obs["datetime"].dt.hour + sim_vs_obs["datetime"].dt.minute / 60.0

    # Define constants
    altitude_km = 0.08
    surface_albedo = sim_vs_obs["blue_sky_albedo_median"].mean()

    # Read LUT
    lut = pd.read_csv(lut_path) 
    variables = ["DOY", "Hour", "Albedo", "Altitude_km", "CloudTop_km", "Tau550", "CloudType"] 
    unique_values = {var: lut[var].unique() for var in variables if var in lut.columns}
    
    # Function to get clear sky GHI (direct + diffuse)
    def get_clear_sky_ghi(row):
        res = get_closest_lut_entry(
            lut=lut,
            unique_values=unique_values,
            doy=row["DOY"],
            hour=row["Hour"],
            albedo=surface_albedo,
            altitude_km=altitude_km,
            cloud_top_km=None,  # clear sky
            cot=None,
            cloud_phase=None
        )
        # Return sum of direct + diffuse
        if res["direct_clear"] is None or res["diffuse_clear"] is None:
            return np.nan
        return res["direct_clear"] + res["diffuse_clear"]
    
    # Apply to each row 
    sim_vs_obs["clear_sky"] = sim_vs_obs.apply(get_clear_sky_ghi, axis=1)
    print(sim_vs_obs[["datetime", "sky_type", "clear_sky"]].head())
    print(sim_vs_obs[["clear_sky"]].describe())
    
    # Define a function to compute the threshold for a given ICS value
    def threshold(ics):
        if ics <= 100:
            return 2 * ics + 0
        else:
            return 1.05 * ics + 95

    # Apply the threshold to Flesland
    sim_vs_obs["Flesland_flag"] = np.where(
        sim_vs_obs["Flesland_ghi_1M"] > sim_vs_obs["clear_sky"].apply(threshold),
        "CS",
        sim_vs_obs["Flesland_flag"]
    )

    # Apply the threshold to Florida
    sim_vs_obs["Florida_flag"] = np.where(
        sim_vs_obs["Florida_ghi_1M"] > sim_vs_obs["clear_sky"].apply(threshold),
        "CS",
        sim_vs_obs["Florida_flag"]
    )

    # Print counts of "CS" flag per station
    print("Flesland CS flagged:", (sim_vs_obs["Flesland_flag"] == "CS").sum())
    print("Florida CS flagged:", (sim_vs_obs["Florida_flag"] == "CS").sum())
    
    return sim_vs_obs

def compute_error_metrics(sim, obs, tolerance_pct=10):
    # Remove NaNs
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]
    
    mbe = np.mean(sim - obs)
    mae = np.mean(np.abs(sim - obs))
    rmse = np.sqrt(np.mean((sim - obs)**2))
    
    # Percentage within tolerance
    pct_within_tol = np.mean(np.abs(sim - obs) <= (tolerance_pct/100) * obs) * 100
    
    return {"MBE": mbe, "MAE": mae, "RMSE": rmse, f"% within ±{tolerance_pct}%": pct_within_tol}



if __name__ == "__main__": 
    sim_vs_obs_path = "data/processed/s2_cloud_cover_table_small_and_large_with_simulated_florida_flesland_ghi.csv"
    lut_path = "data/processed/LUT/LUT.csv"

    # Correlations 
    # frost coordinates: 0.723 Florida, 0.723 Flesland
    # plotting coordinates: 0.723 Florida, 0.886 Flesland
    # shifted coordinates: 0.626 Florida, 0.886 Flesland
    # ECAD coordinates: 0.718 Florida, 0.8709 Flesland
    # Read
    sim_vs_obs = pd.read_csv(sim_vs_obs_path)
    # Define conditions
    conditions = [
        sim_vs_obs["cloud_cover_large"] <= 1,      # clear
        sim_vs_obs["cloud_cover_large"] >= 99,     # overcast
    ]

    choices = ["clear", "overcast"]

    # Default is "mixed"
    sim_vs_obs["sky_type"] = np.select(conditions, choices, default="mixed")  
    
    sim_vs_obs = flag_observations(sim_vs_obs, lut_path)
    
    
    # Make new subset 
    # Copy subset without modifying the original
    sim_vs_obs_sub = sim_vs_obs[[
        "date", "sky_type", "cloud_cover_large", 
        "Florida_ghi_sim", "Flesland_ghi_sim", 
        "Flesland_ghi_1M", "Florida_ghi_1M"
    ]].copy()


    # Overwrite GHI values with NaN where flag is not OK
    sim_vs_obs_sub["Flesland_ghi_1M"] = np.where(
        sim_vs_obs["Flesland_flag"] != "OK", 
        np.nan, 
        sim_vs_obs_sub["Flesland_ghi_1M"]
    )

    sim_vs_obs_sub["Florida_ghi_1M"] = np.where(
        sim_vs_obs["Florida_flag"] != "OK", 
        np.nan, 
        sim_vs_obs_sub["Florida_ghi_1M"]
    )

    # Compute error: sim - obs
    sim_vs_obs_sub["error_flesland"] = sim_vs_obs_sub["Flesland_ghi_sim"] - sim_vs_obs_sub["Flesland_ghi_1M"]
    sim_vs_obs_sub["error_florida"] = sim_vs_obs_sub["Florida_ghi_sim"] - sim_vs_obs_sub["Florida_ghi_1M"]

    # Drop NaNs in error or cloud_fraction
    flesland_corr_data = sim_vs_obs_sub[["error_flesland", "cloud_cover_large"]].dropna()
    florida_corr_data = sim_vs_obs_sub[["error_florida", "cloud_cover_large"]].dropna()
    
    # Filter for mixed sky type
    flesland_mixed = flesland_corr_data[sim_vs_obs_sub["sky_type"] == "mixed"]
    florida_mixed = florida_corr_data[sim_vs_obs_sub["sky_type"] == "mixed"]

    # Compute correlation
    corr_flesland_mixed = flesland_mixed["error_flesland"].corr(flesland_mixed["cloud_cover_large"])
    corr_florida_mixed = florida_mixed["error_florida"].corr(florida_mixed["cloud_cover_large"])

    print(f"Correlation between cloud cover and error (Flesland, mixed skies): {corr_flesland_mixed:.3f}")
    print(f"Correlation between cloud cover and error (Florida, mixed skies): {corr_florida_mixed:.3f}")


    # Make sure date is datetime
    sim_vs_obs_sub["date"] = pd.to_datetime(sim_vs_obs_sub["date"])
    
    # Print basic info and statistics
    print_basic_info(sim_vs_obs_sub)
    
    # Compute error metrics 
    # Flesland
    flesland_metrics = compute_error_metrics(
        sim_vs_obs_sub["Flesland_ghi_sim"].values,
        sim_vs_obs_sub["Flesland_ghi_1M"].values
    )

    # Florida
    florida_metrics = compute_error_metrics(
        sim_vs_obs_sub["Florida_ghi_sim"].values,
        sim_vs_obs_sub["Florida_ghi_1M"].values
    )

    print("Flesland metrics:", flesland_metrics)
    print("Florida metrics:", florida_metrics)
    
    # Plot scatter Flesland sim - Flesland obs
    """ scatter_with_fit(
        sim_vs_obs_sub["Flesland_ghi_sim"].values,
        sim_vs_obs_sub["Flesland_ghi_1M"].values,
        xlabel="Flesland simulated GHI",
        ylabel="Flesland observed GHI",
        title="Flesland sim vs obs",
        outpath="output/scatter_flesland_sim_flesland_obs_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )

    # Plot scatter Florida sim - Florida obs
    scatter_with_fit(
        sim_vs_obs_sub["Florida_ghi_sim"].values,
        sim_vs_obs_sub["Florida_ghi_1M"].values,
        xlabel="Florida simulated GHI",
        ylabel="Florida observed GHI",
        title="Florida sim vs obs",
        outpath="output/scatter_florida_sim_florida_obs_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )
    
    # Plot scatter Florida sim - Florida sim
    scatter_with_fit(
        sim_vs_obs_sub["Florida_ghi_sim"].values,
        sim_vs_obs_sub["Flesland_ghi_sim"].values,
        xlabel="Florida simulated GHI",
        ylabel="Flesland simulated GHI",
        title="Florida vs Flesland sim", 
        outpath="output/scatter_flesland_sim_florida_sim_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )
    
    # Plot scatter Flesland obs - Florida obs 
    scatter_with_fit(
        sim_vs_obs_sub["Florida_ghi_1M"].values,
        sim_vs_obs_sub["Flesland_ghi_1M"].values,
        xlabel="Florida observed GHI",
        ylabel="Flesland observed GHI",
        title="Florida vs Flesland obs",
        outpath="output/scatter_flesland_obs_florida_obs_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )
    
    # Convert to DOY for line plots    
    sim_vs_obs_sub["doy"] = sim_vs_obs_sub["date"].dt.dayofyear
  
    # Plot doy - Flesland sim, Flesland obs
    for year in range(2015, 2026):
        single_year = sim_vs_obs_sub[sim_vs_obs_sub["date"].dt.year == year]
        lineplot_doy(
            single_year["doy"],
            single_year["Flesland_ghi_sim"],
            single_year["Flesland_ghi_1M"],
            ylabel="GHI (W/m2)",
            title=f"DOY vs Flesland GHI ({year})",
            outpath=f"output/lineplot_flesland_sim_obs_doy_{year}.png"
        )
        # Plot doy - Florida sim, Florida obs  
        lineplot_doy(
            single_year["doy"],
            single_year["Florida_ghi_sim"],
            single_year["Florida_ghi_1M"],
            ylabel="GHI (W/m2)",
            title=f"DOY vs Florida GHI ({year})",
            outpath=f"output/lineplot_florida_sim_obs_doy_{year}.png"
        )  """
    
    
    
    