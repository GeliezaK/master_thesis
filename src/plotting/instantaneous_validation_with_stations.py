# ======================================================================
# Compute error metrics, plot simulations vs. observations, describe
# simulated and observed data distributions. 
# ======================================================================

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from scipy.stats import linregress
from sklearn.linear_model import TheilSenRegressor
from src.model import MIXED_THRESHOLD, OVERCAST_THRESHOLD, COARSE_RESOLUTIONS
from src.plotting import SIMULATION_COLOR, SIMULATION_LS, SIMULATION_M, OBSERVATION_COLOR, OBSERVATION_LS, OBSERVATION_M
from src.plotting import SKY_TYPE_COLORS, STATION_COLORS, set_paper_style
from src.preprocessing.quality_control_flag_stations_data import flag_observations

set_paper_style()

def print_basic_info(sim_vs_obs):
    """Print descriptive information of table with florida and flesland 
    ghi measurements and simulated data."""
    
    # Correlations 
    sub_merged = sim_vs_obs[[
        "date",
        "sky_type",
        "florida_ghi_sim_horizontal",
        "flesland_ghi_sim_horizontal",
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
        "Florida_sim_vs_obs": sub_merged["florida_ghi_sim_horizontal"].corr(sub_merged["Florida_ghi_1M"]),
        "Flesland_sim_vs_obs": sub_merged["flesland_ghi_sim_horizontal"].corr(sub_merged["Flesland_ghi_1M"]),
        "Florida_vs_Flesland_sim": sub_merged["florida_ghi_sim_horizontal"].corr(sub_merged["flesland_ghi_sim_horizontal"]),
        "Florida_vs_Flesland_obs": sub_merged["Florida_ghi_1M"].corr(sub_merged["Flesland_ghi_1M"]),
    }
    for k, v in corrs.items():
        print(f"{k}: {v}")
        

def scatter_with_fit(sim, obs, xlabel="", ylabel="", title="", outpath="output/scatterplot.png", 
    ax=None, show_stats=True, sky_type=None):
    """
    Scatter plot between simulations and observations with 1:1 dashed line, optional R² and correlation display.
    Optionally color points by sky type (categorical variable).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if sky_type is None:
        # Plain scatter
        ax.scatter(sim, obs, alpha=0.7, label="Data")
    else:
        # Color by sky type
        categories = np.unique(sky_type[~pd.isna(sky_type)])
        for cat in categories:
            mask = (sky_type == cat) & ~np.isnan(sim) & ~np.isnan(obs)
            ax.scatter(
                sim[mask], obs[mask], alpha=0.7, label=str(cat), color=SKY_TYPE_COLORS[cat]
            )

    # 1:1 dashed line
    min_val = min(np.nanmin(sim), np.nanmin(obs))
    max_val = max(np.nanmax(sim), np.nanmax(obs))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal 1:1")

    # Compute stats
    if show_stats:
        mask_all = ~np.isnan(sim) & ~np.isnan(obs)
        corr_all = np.corrcoef(sim[mask_all], obs[mask_all])[0, 1]
        lam, delta, sigma, r2 = compute_error_model_tian2016(obs[mask_all], sim[mask_all])

        stats_text = f"Corr={corr_all:.2f}\nλ={lam:.2f}\nδ={delta:.2f}\nσ={sigma:.2f}\nR²={r2:.2f}"

        
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            fontsize=14
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
    """Plot DOY (x-axis) vs variable (y-axis). Optionally plot two lines (e.g. sim and obs). 
    Only plot points where both are not NaN."""
    
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

    ax.plot(x_plot, y1_plot, marker=SIMULATION_M, linestyle=SIMULATION_LS, color=SIMULATION_COLOR, label=labels[0])
    if y2_plot is not None:
        ax.plot(x_plot, y2_plot, marker=OBSERVATION_M, linestyle=OBSERVATION_LS, color=OBSERVATION_COLOR, label=labels[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved lineplot to {outpath}.")
    

def compute_error_model_tian2016(x, y):
    """Compute error metrics based on error model in Tian et al (2016). DOI:
    https://doi.org/10.1175/MWR-D-15-0087.1 """
    # Align data
    df = np.column_stack([x, y])
    df = df[~np.isnan(df).any(axis=1)]
    x, y = df[:,0], df[:,1]

    res = linregress(x, y)
    lam = res.slope      # λ
    delta = res.intercept  # δ
    residuals = y - (lam * x + delta)
    sigma = np.std(residuals, ddof=1)  # σ, random error std

    return lam, delta, sigma, res.rvalue**2


def compute_error_metrics(df, sim_col="flesland_ghi_sim_horizontal_log", 
                          obs_col="Flesland_1M_ghi_log", sky_col="sky_type", tolerance_pct=10):
    """
    Compute performance metrics (MBE, MAE, RMSE, % within tolerance,
    NSE, Willmott d, Legates E1, R², and Kolmogorov–Smirnov test integral KSI)
    for modeled vs observed solar irradiance, both overall and per sky type.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns [sim_col, obs_col, sky_col].
    sim_col : str
        Name of column with modeled values.
    obs_col : str
        Name of column with observed values.
    sky_col : str
        Name of column indicating sky type ('clear', 'mixed', 'overcast').
    tolerance_pct : float
        Percentage tolerance for “% within ±X%” metric.

    Returns
    -------
    dict
        Nested dictionary with metrics for 'total' and each sky type.
    """


    def metrics(sim, obs):
        """Compute all metrics for one subset of data."""
        mask = ~np.isnan(sim) & ~np.isnan(obs)
        sim = np.asarray(sim[mask])
        obs = np.asarray(obs[mask])

        if len(obs) == 0:
            return {k: np.nan for k in [
                "lambda", "delta", "sigma",
                "MBE", "MAE", "RMSE", f"% within ±{tolerance_pct}%",
                "NSE", "Willmott_d", "Legates_E1", "R2", "KSI"
            ]}

        # --- Basic metrics ---
        mbe = np.mean(sim - obs)
        mae = np.mean(np.abs(sim - obs))
        rmse = np.sqrt(np.mean((sim - obs)**2))
        pct_within_tol = np.mean(np.abs(sim - obs) <= (tolerance_pct / 100) * obs) * 100
        
        # --- Fit error model by Tian et al (2016) --- 
        lam, delta, sigma, r2 = compute_error_model_tian2016(obs, sim)

        # --- Nash–Sutcliffe Efficiency (NSE) ---
        nse = 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)

        # --- Willmott’s Index of Agreement (d) ---
        willmott_d = 1 - np.sum((sim - obs)**2) / np.sum((np.abs(sim - np.mean(obs)) + np.abs(obs - np.mean(obs)))**2)

        # --- Legates & McCabe Coefficient of Efficiency (E1) ---
        legates_E1 = 1 - np.sum(np.abs(sim - obs)) / np.sum(np.abs(obs - np.mean(obs)))

        # --- Kolmogorov–Smirnov test integral (KSI) ---
        ksi = compute_ksi(sim, obs)

        return {
            "MBE": mbe,
            "MAE": mae,
            "RMSE": rmse,
            f"% within ±{tolerance_pct}%": pct_within_tol,
            "lambda": lam, 
            "delta": delta, 
            "sigma": sigma, 
            "NSE": nse,
            "Willmott_d": willmott_d,
            "Legates_E1": legates_E1,
            "R2": r2,
            "KSI": ksi,
        }

    # --- Compute for total and each sky type ---
    results = {"total": metrics(df[sim_col], df[obs_col])}
    for sky in df[sky_col].dropna().unique():
        subset = df[df[sky_col] == sky]
        results[sky] = metrics(subset[sim_col], subset[obs_col])

    return results


def compute_ksi(sim, obs, nbins=50):
    """
    Compute Kolmogorov–Smirnov test integral (KSI) following Espinar et al. (2010). 
    DOI: https://doi.org/10.1016/j.solener.2008.07.009

    Returns KSI [%].
    """
    sim = np.asarray(sim)
    obs = np.asarray(obs)
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim, obs = sim[mask], obs[mask]
    if len(obs) < 5:
        return np.nan

    # Normalize both to [0,1] range
    xmin = min(sim.min(), obs.min())
    xmax = max(sim.max(), obs.max())
    sim_norm = (sim - xmin) / (xmax - xmin)
    obs_norm = (obs - xmin) / (xmax - xmin)

    # Empirical CDFs
    hist_sim, bin_edges = np.histogram(sim_norm, bins=nbins, range=(0, 1), density=True)
    hist_obs, _ = np.histogram(obs_norm, bins=nbins, range=(0, 1), density=True)
    cdf_sim = np.cumsum(hist_sim) / np.sum(hist_sim)
    cdf_obs = np.cumsum(hist_obs) / np.sum(hist_obs)

    Dn = np.abs(cdf_sim - cdf_obs)
    dx = 1 / nbins
    N = len(obs)
    Phi = 1.63  # asymptotic value
    Dc = Phi / np.sqrt(N)
    Ac = Dc * (xmax - xmin)
    KSI = 100 * (1 / Ac) * np.sum(Dn * dx * (xmax - xmin))

    return KSI


def plot_error_vs_cloud_cover(df, x_col="cloud_cover_large",
                              cat_col="florida_cloud_shadow",
                              error_col="error", 
                              title="Florida Absolute Simulation Error vs. Cloud Cover",
                              xlabel="Cloud Cover (%)",
                              ylabel="Sim - Obs",
                              plot_regression=True,
                              outpath="output/florida_shadow_cloud_cover_vs_error.png"):
    """
    Plot simulation error versus some continuous control variable (x_col) and cloud shadow (cat_col).

    Optionally fits Theil–Sen regressions for both clear and shadow pixels and annotates slopes and intercepts.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing columns for control variable, simulation error, and categorical variable.
    x_col : str, optional
        Column name for control variable (default: "cloud_cover_large").
    cat_col : str, optional
        Column name for binary cloud shadow variable (default: "florida_cloud_shadow").
    error_col : str, optional
        Column name for model error (Simulated - Observed) (default: "error").
    title : str, optional
        Plot title (default: "Florida Absolute Simulation Error vs. Cloud Cover").
    xlabel : str, optional
        Label for x-axis (default: "Cloud Cover (%)").
    ylabel : str, optional
        Label for y-axis (default: "Sim - Obs").
    plot_regression : bool, optional
        Whether to compute and plot Theil–Sen regression lines (default: True).
    outpath : str, optional
        File path to save the figure (default: "output/florida_shadow_cloud_cover_vs_error.png").

    Outputs
    -------
    Saves a scatter plot with regression lines (if enabled) to the specified `outpath`.
    """


    # Compute error
    df = df.copy()

    # Separate shadow vs clear pixels
    clear_mask = df[cat_col] == 0
    shadow_mask = df[cat_col] == 1

    plt.figure(figsize=(8,6))
    
    # Plot clear pixels
    plt.scatter(df.loc[clear_mask, x_col],
                df.loc[clear_mask, error_col],
                color="blue", marker="o", alpha=0.6, label="Clear (shadow=0)")
    
    # Plot shadow pixels
    plt.scatter(df.loc[shadow_mask, x_col],
                df.loc[shadow_mask, error_col],
                color="red", marker="^", alpha=0.6, label="Shadow (shadow=1)")
    
    if plot_regression:
        # --- Theil–Sen regression: Clear pixels ---
        x_clear = df.loc[clear_mask, x_col].values.reshape(-1, 1)
        y_clear = df.loc[clear_mask, error_col].values
        mask_valid = ~np.isnan(x_clear.flatten()) & ~np.isnan(y_clear)
        x_clear = x_clear[mask_valid].reshape(-1, 1)
        y_clear = y_clear[mask_valid]

        if len(x_clear) > 10:
            model_clear = TheilSenRegressor().fit(x_clear, y_clear)
            slope_clear = model_clear.coef_[0]
            intercept_clear = model_clear.intercept_
            x_fit = np.linspace(x_clear.min(), x_clear.max(), 100).reshape(-1, 1)
            y_fit = model_clear.predict(x_fit)
            plt.plot(x_fit, y_fit, color="darkturquoise", linestyle="--", linewidth=2, label="Theil–Sen fit (clear)")
        else:
            slope_clear = intercept_clear = np.nan

        # --- Theil–Sen regression: Shadow pixels ---
        x_shadow = df.loc[shadow_mask, x_col].values.reshape(-1, 1)
        y_shadow = df.loc[shadow_mask, error_col].values
        mask_valid = ~np.isnan(x_shadow.flatten()) & ~np.isnan(y_shadow)
        x_shadow = x_shadow[mask_valid].reshape(-1, 1)
        y_shadow = y_shadow[mask_valid]

        if len(x_shadow) > 10:
            model_shadow = TheilSenRegressor().fit(x_shadow, y_shadow)
            slope_shadow = model_shadow.coef_[0]
            intercept_shadow = model_shadow.intercept_
            x_fit_shadow = np.linspace(x_shadow.min(), x_shadow.max(), 100).reshape(-1, 1)
            y_fit_shadow = model_shadow.predict(x_fit_shadow)
            plt.plot(x_fit_shadow, y_fit_shadow, color="orange", linestyle="--", linewidth=2, label="Theil–Sen fit (shadow)")
        else:
            slope_shadow = intercept_shadow = np.nan
            
        # Add slope/intercept text
        textstr = (f"Clear: a={slope_clear:.2f}, b={intercept_clear:.2f}\n"
                f"Shadow: a={slope_shadow:.2f}, b={intercept_shadow:.2f}")
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                fontsize=14, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Scatter plot saved to {outpath}.")
    

def print_sky_type_percentages(sim_vs_obs): 
    """Print counts and percentages of each sky type in Sentinel-2 observations."""
    # Total observations
    total_obs = len(sim_vs_obs)
    
    # Count values per sky type
    counts = sim_vs_obs["sky_type"].value_counts(dropna=False)

    # Percentages
    percentages = sim_vs_obs["sky_type"].value_counts(normalize=True, dropna=False) * 100

    # Combine into a single DataFrame for nice printing
    summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })
    
    print(summary)
    print(f"\nTotal observations: {total_obs}")
    

def plot_obs_vs_sim_distributions(
    florida_obs, flesland_obs,
    florida_sim, flesland_sim,
    outpath,
    xlabel="Global Horizontal Irradiance [W/m²]",
    title="Distribution of Measured and Simulated GHI"
):
    """
    Plot side-by-side histograms (normalized) of observed vs simulated GHI
    for Flesland and Florida.

    Parameters
    ----------
    florida_obs : array-like
        Observed GHI values for Florida.
    flesland_obs : array-like
        Observed GHI values for Flesland.
    florida_sim : array-like
        Simulated GHI values for Florida.
    flesland_sim : array-like
        Simulated GHI values for Flesland.
    outpath : str
        Path to save the figure.
    xlabel : str, optional
        Label for x-axis.
    title : str, optional
        Main title of the figure.
    """

    # Compute bins based on both observations and simulations
    global_max = max(
        np.nanmax(florida_obs),
        np.nanmax(flesland_obs),
        np.nanmax(florida_sim),
        np.nanmax(flesland_sim)
    )
    bins = np.linspace(0, global_max, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute histograms
    def normalized_counts(data):
        counts, _ = np.histogram(data, bins=bins)
        return counts / counts.sum()

    counts_flesland_obs = normalized_counts(flesland_obs)
    counts_flesland_sim = normalized_counts(flesland_sim)
    counts_florida_obs  = normalized_counts(florida_obs)
    counts_florida_sim  = normalized_counts(florida_sim)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    plt.subplots_adjust(wspace=0.25)

    # ---------------- Left: Flesland ----------------
    ax = axes[0]
    ax.plot(bin_centers, counts_flesland_obs,
            color=OBSERVATION_COLOR, linestyle=OBSERVATION_LS, marker=OBSERVATION_M,
            label="Observed")
    ax.plot(bin_centers, counts_flesland_sim,
            color=SIMULATION_COLOR, linestyle=SIMULATION_LS, marker=SIMULATION_M,
            label="Simulated")

    ax.set_title("Flesland")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    # ---------------- Right: Florida ----------------
    ax = axes[1]
    ax.plot(bin_centers, counts_florida_obs,
            color=OBSERVATION_COLOR, linestyle=OBSERVATION_LS, marker=OBSERVATION_M,
            label="Observed")
    ax.plot(bin_centers, counts_florida_sim,
            color=SIMULATION_COLOR, linestyle=SIMULATION_LS, marker=SIMULATION_M,
            label="Simulated")

    ax.set_title("Florida")
    ax.set_xlabel(xlabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    # Main title & save
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✅ Distribution comparison plots saved to {outpath}.")
    
    
def plot_obs_distributions(ghi_florida, ghi_flesland, outpath,
                           xlabel="Measured GHI [W/m²]",
                           title="Distribution of Measured GHI"):
    """Plot histogram of distribution of observations at Flesland and 
    Florida stations. Save figure to outpath."""
    
    florida_min, florida_max = np.nanmin(ghi_florida), np.nanmax(ghi_florida)
    flesland_min, flesland_max = np.nanmin(ghi_flesland), np.nanmax(ghi_flesland)

    # Compute histogram values
    bins = np.linspace(0, max(florida_max, flesland_max), 30)

    counts_florida, edges = np.histogram(ghi_florida, bins=bins)
    counts_flesland, _    = np.histogram(ghi_flesland, bins=bins)

    # Convert to bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Normalize if you want to compare shapes (frequency density)
    counts_florida = counts_florida / counts_florida.sum()
    counts_flesland = counts_flesland / counts_flesland.sum()

    # Plot as line plots
    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, counts_florida, color=STATION_COLORS["Florida"], marker="^", label="Florida")
    plt.plot(bin_centers, counts_flesland, color=STATION_COLORS["Flesland"], marker="o", label="Flesland")

    plt.xlabel(xlabel)
    plt.ylabel("Frequency (%)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Distribution plot saved to {outpath}.")
    
def test_heteroscedasticity_white(flesland_sim, flesland_obs, 
                                  florida_sim, florida_obs): 
    """Perform White's test of heteroscedasticity of flesland and florida
    simulations vs. observations."""
    
    # Flesland: observed vs simulated (log-transformed)
    y_flesland = flesland_obs
    X_flesland = sm.add_constant(flesland_sim)
    model_flesland = sm.OLS(y_flesland, X_flesland).fit()
    resid_flesland = model_flesland.resid

    # Florida: observed vs simulated (log-transformed)
    y_florida = florida_obs
    X_florida = sm.add_constant(florida_sim)
    model_florida = sm.OLS(y_florida, X_florida).fit()
    resid_florida = model_florida.resid

    # ---------------------------------------------------------------------
    # Perform White's test
    # ---------------------------------------------------------------------
    labels = ['LM Stat', 'LM p-value', 'F-Stat', 'F p-value']

    white_flesland = het_white(resid_flesland, model_flesland.model.exog)
    white_florida  = het_white(resid_florida, model_florida.model.exog)

    print("=== White Test for Heteroscedasticity ===")
    print("Flesland:")
    print(dict(zip(labels, white_flesland)))
    print(f"→ Interpretation: p-value = {white_flesland[3]:.4f} "
        f"{'→ heteroscedasticity detected' if white_flesland[3] < 0.05 else '→ homoscedastic'}")

    print("\nFlorida:")
    print(dict(zip(labels, white_florida)))
    print(f"→ Interpretation: p-value = {white_florida[3]:.4f} "
        f"{'→ heteroscedasticity detected' if white_florida[3] < 0.05 else '→ homoscedastic'}")



def main(): 
    sim_ECAD_path = "data/processed/s2_cloud_cover_table_small_and_large_with_simulated_florida_flesland_ghi.csv"
    sim_vs_obs_path = "data/processed/s2_cloud_cover_with_stations_with_pixel_sim.csv"

    # Correlations 
    # Read
    sim_ECAD = pd.read_csv(sim_ECAD_path)
    sim_vs_obs = pd.read_csv(sim_vs_obs_path)
    sim_vs_obs["flesland_ghi_sim_ECAD"] = sim_ECAD["Flesland_ghi_sim_ECAD"]
    sim_vs_obs["florida_ghi_sim_ECAD"] = sim_ECAD["Florida_ghi_sim_ECAD"]
    required_cols = ["date",
        "cloud_cover_large_thresh50", "cloud_cover_small_thresh50",
        "cloud_cover_large", "cloud_cover_small", 
        "cth_median_large", "cot_median_large", "cph_median_large",
        "cth_median_small", "cot_median_small", "cph_median_small",
        "Florida_ghi_1M", "Flesland_ghi_1M",
        "MEAN_ZENITH", "MEAN_AZIMUTH", "month",
        "florida_ghi_sim_ECAD", "flesland_ghi_sim_ECAD", "total_clear_sky",
        "florida_ghi_sim_horizontal", "flesland_ghi_sim_horizontal", "florida_cloud_shadow", 
        "flesland_cloud_shadow", "Florida_ghi_sim_ECAD_100m", "Flesland_ghi_sim_ECAD_100m", 
        "Florida_ghi_sim_ECAD_500m", "Flesland_ghi_sim_ECAD_500m",
        "Florida_ghi_sim_ECAD_1000m", "Flesland_ghi_sim_ECAD_1000m",
        "Florida_ghi_sim_ECAD_5000m", "Flesland_ghi_sim_ECAD_5000m",
        "Florida_ghi_sim_ECAD_25000m", "Flesland_ghi_sim_ECAD_25000m"
    ]

    assert all(col in sim_vs_obs.columns for col in required_cols), \
        f"Missing columns: {set(required_cols) - set(sim_vs_obs.columns)}"


    # ------------------------------ Add sky type ---------------------------
    # Define conditions
    conditions = [
        sim_vs_obs["cloud_cover_large"] <= MIXED_THRESHOLD,      # clear
        sim_vs_obs["cloud_cover_large"] >= OVERCAST_THRESHOLD,     # overcast
    ]

    choices = ["clear", "overcast"]

    # Default is "mixed"
    sim_vs_obs["sky_type"] = np.select(conditions, choices, default="mixed")  
    #print_sky_type_percentages(sim_vs_obs)
    
    # ------------------------------ Exclude outliers and missing data -----------------------
    sim_vs_obs["datetime"] = pd.to_datetime(
        sim_vs_obs["system:time_start_large"], unit="ms", utc=True
    )
    sim_vs_obs = flag_observations(sim_vs_obs, obs_col="Florida_ghi_1M", datetime_col="datetime")
    sim_vs_obs = flag_observations(sim_vs_obs, obs_col="Flesland_ghi_1M", datetime_col="datetime")
    
    print(list(sim_vs_obs))
    
    # Make new subset 
    # Copy subset without modifying the original
    subset_cols = ["sky_type"] + required_cols
    sim_vs_obs_sub = sim_vs_obs[subset_cols].copy()
    
    # Print basic info and statistics
    print_basic_info(sim_vs_obs_sub)

    # Overwrite GHI values with NaN where flag is not OK
    sim_vs_obs_sub["Flesland_ghi_1M"] = np.where(
        sim_vs_obs["Flesland_ghi_1M_flag"] != "OK", 
        np.nan, 
        sim_vs_obs_sub["Flesland_ghi_1M"]
    )

    sim_vs_obs_sub["Florida_ghi_1M"] = np.where(
        sim_vs_obs["Florida_ghi_1M_flag"] != "OK", 
        np.nan, 
        sim_vs_obs_sub["Florida_ghi_1M"]
    )
    
    
    # ---------------------- Log Transformation ---------------------------
    # Transform everything to log because of non-Gaussian distribution 
    cols_to_check = [
        "flesland_ghi_sim_horizontal",
        "florida_ghi_sim_horizontal",
        "Flesland_ghi_1M",
        "Florida_ghi_1M",
        "Florida_ghi_sim_ECAD_100m",
        "Flesland_ghi_sim_ECAD_100m",
        "Florida_ghi_sim_ECAD_500m",
        "Flesland_ghi_sim_ECAD_500m",
        "Florida_ghi_sim_ECAD_1000m",
        "Flesland_ghi_sim_ECAD_1000m",
        "Florida_ghi_sim_ECAD_5000m",
        "Flesland_ghi_sim_ECAD_5000m",
        "Florida_ghi_sim_ECAD_25000m",
        "Flesland_ghi_sim_ECAD_25000m"
    ]

    # Assert no exact zeros in the data
    for col in cols_to_check:
        n_zeros = (sim_vs_obs_sub[col] == 0).sum()
        n_negative = (sim_vs_obs_sub[col] < 0).sum()
        if n_zeros > 0:
            raise ValueError(f"Column '{col}' contains {n_zeros} zero values — cannot take log.")
        elif n_negative > 0:
            raise ValueError(f"Column '{col}' contains {n_negative} negative values — cannot take log.")
        else:
            print(f"✅ No zeros or negative values in column '{col}'")

    # Apply log(x/s) with standard deviation s
    for col in cols_to_check:
        s = sim_vs_obs_sub[col].std()
        log_col = f"{col}_log"
        sim_vs_obs_sub[log_col] = np.log(sim_vs_obs_sub[col] /s)
        print(f"Applied log-transform to '{col}' using s = {s:.4f}")
    
    
    # ------------------------- Plot measurement distributions ------------------------
    # Drop NaNs for plotting
    stations = ["Florida", "Flesland"]

    # Define variable pairs for non-log and log data
    var_pairs = [
        ("ghi_1M", "ghi_sim_horizontal"),  # normal data
        ("ghi_1M_log", "ghi_sim_horizontal_log")  # log-transformed data
    ]

    for obs_suffix, sim_suffix in var_pairs:
        obs_data = {}
        sim_data = {}

        # Collect valid data per station (drop rows where sim or obs is NaN)
        for station_name in stations:
            obs_col = f"{station_name}_{obs_suffix}"
            sim_col = f"{station_name.lower()}_{sim_suffix}"

            valid = sim_vs_obs_sub[[obs_col, sim_col]].dropna()
            obs_data[station_name] = valid[obs_col]
            sim_data[station_name] = valid[sim_col]

            print(f"✅ {station_name}: using {len(valid)} valid pairs after NaN removal for {obs_col} vs {sim_col}")

        # -------------------- Plot distributions --------------------
        outpath = (
            f"output/{stations[1].lower()}_{stations[0].lower()}_"
            f"{'log_' if 'log' in obs_suffix else ''}measurements_vs_sim_distribution.png"
        )

        xlabel = (
            r"$\log\!\left(\frac{I}{\sigma}\right)$"
            if "log" in obs_suffix
            else "Global Horizontal Irradiance [W/m²]"
        )

        title = (
            "Distribution of log-transformed measurements"
            if "log" in obs_suffix
            else "Distribution of Measured and Simulated GHI"
        )

        plot_obs_vs_sim_distributions(
            obs_data["Florida"], obs_data["Flesland"],
            sim_data["Florida"], sim_data["Flesland"],
            outpath=outpath,
            xlabel=xlabel,
            title=title
        )

        # -------------------- White test for heteroscedasticity --------------------
        print(f"\n=== White test ({'log' if 'log' in obs_suffix else 'linear'} data) ===")
        test_heteroscedasticity_white(
            sim_data["Flesland"], obs_data["Flesland"],
            sim_data["Florida"],  obs_data["Florida"]
        )
        print("\n" + "-"*75 + "\n")
        
    # --------------------------- Explore error correlations --------------------------
    # Compute error: sim - obs
    # Compute model error (simulation minus observation)
    sim_vs_obs_sub["error_flesland"] = (sim_vs_obs_sub["Flesland_ghi_1M_log"] - sim_vs_obs_sub["flesland_ghi_sim_horizontal_log"])
    sim_vs_obs_sub["error_florida"] = (sim_vs_obs_sub["Florida_ghi_1M_log"] - sim_vs_obs_sub["florida_ghi_sim_horizontal_log"])
    # Filter for mixed sky type
    mixed = sim_vs_obs_sub[sim_vs_obs_sub["sky_type"] == "mixed"].copy()

    predictors = [
        "cloud_cover_large", "florida_cloud_shadow", "flesland_cloud_shadow",
        "cth_median_small", "cot_median_small", "cph_median_small",
        "MEAN_ZENITH", "MEAN_AZIMUTH", "total_clear_sky"
    ]

    print("=== Error Correlations (Mixed Skies) ===")
    for var in predictors:
        valid_florida = mixed[["error_florida", var]].dropna()  # only drop NaNs in these two
        valid_flesland = mixed[["error_flesland", var]].dropna()  # only drop NaNs in these two
        corr_fles = valid_flesland["error_flesland"].corr(valid_flesland[var])
        corr_flor = valid_florida["error_florida"].corr(valid_florida[var])
        print(f"{var:25s} | Flesland: {corr_fles:6.3f} | Florida: {corr_flor:6.3f}")

    # Plot error vs cloud cover and cloud shadow
    coarse_sims = [f"_ghi_sim_ECAD_{res}m" for res in COARSE_RESOLUTIONS]
    coarse_log_sims = [sim_scen + "_log" for sim_scen in coarse_sims]
    
    for sim_scen in coarse_sims: 
        print(f"\n======== {sim_scen} ========")
        for station_name in stations:
            sim_col = station_name + sim_scen
            obs_col = f"{station_name}_ghi_1M"
            error_col = f"{station_name.lower()}_error"

            sim_vs_obs_sub[error_col] = sim_vs_obs_sub[obs_col] - sim_vs_obs_sub[sim_col]
            
            print(f"\n=== {station_name} ===")
            
            for sky in sim_vs_obs_sub["sky_type"].unique():
                subset = sim_vs_obs_sub.loc[sim_vs_obs_sub["sky_type"] == sky].copy()

                # Compute normalized absolute error (relative to total_clear_sky)
                subset["mae_norm"] = (subset[error_col].abs() / subset["total_clear_sky"]) * 100

                # Drop NaNs or invalid (e.g. clear sky = 0)
                subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["mae_norm"])

                if len(subset) == 0:
                    print(f"  {sky}: no valid data")
                    continue

                mean_val = subset["mae_norm"].mean()
                median_val = subset["mae_norm"].median()
                std_val = subset["mae_norm"].std()
                min_val = subset["mae_norm"].min()
                max_val = subset["mae_norm"].max()

                print(f"  {sky:>15s} → mean: {mean_val:6.2f}%, median: {median_val:6.2f}%, std: {std_val:6.2f}%, "
                    f"min: {min_val:6.2f}%, max: {max_val:6.2f}%  (N={len(subset)})")
        
        
    for station_name in stations:
        sim_col = f"{station_name.lower()}_ghi_sim_horizontal"
        obs_col = f"{station_name}_ghi_1M"
        error_col = f"{station_name.lower()}_error"
        
        # Plot error vs third variable 
        xlabel = r"$I_{\mathrm{sim}}$"
        ylabel = r"$I_{\mathrm{obs}} - I_{\mathrm{sim}}$"

        plot_error_vs_cloud_cover(sim_vs_obs_sub,
                            x_col=sim_col,
                            cat_col=f"{station_name.lower()}_cloud_shadow",
                            error_col=error_col,
                            title=f"{station_name} Error vs. Simulated Irradiance", 
                            xlabel=xlabel,
                            ylabel=ylabel, 
                            plot_regression=False,
                            outpath=f"output/{station_name.lower()}_shadow_irradiance_vs_error.png")
        
        sim_col_log = f"{station_name.lower()}_ghi_sim_horizontal_log"
        obs_col_log = f"{station_name}_ghi_1M_log"
        error_col_log = f"{station_name.lower()}_error_log"

        sim_vs_obs_sub[error_col_log] = sim_vs_obs_sub[obs_col_log] - sim_vs_obs_sub[sim_col_log]
        xlabel = r"$\log\!\left(\frac{I_{\mathrm{sim}}}{\sigma_{\mathrm{sim}}}\right)$"
        ylabel = r"$\log\!\left(\frac{I_{\mathrm{obs}}}{\sigma_{\mathrm{obs}}}\right) - \log\!\left(\frac{I_{\mathrm{sim}}}{\sigma_{\mathrm{sim}}}\right)$"

        plot_error_vs_cloud_cover(sim_vs_obs_sub,
                            x_col=sim_col_log,
                            cat_col=f"{station_name.lower()}_cloud_shadow",
                            error_col=error_col_log,
                            title=f"{station_name} Error vs. Simulated Irradiance (Log-Transformed)", 
                            xlabel=xlabel,
                            ylabel=ylabel, 
                            plot_regression=False,
                            outpath=f"output/{station_name.lower()}_shadow_irradiance_vs_error_log.png")


    # Make sure date is datetime
    sim_vs_obs_sub["date"] = pd.to_datetime(sim_vs_obs_sub["date"])
    
    
    # ---------------------------------- Compute error metrics -----------------------------------
    coarse_sims = [f"_ghi_sim_ECAD_{res}m" for res in COARSE_RESOLUTIONS]
    coarse_log_sims = [sim_scen + "_log" for sim_scen in coarse_sims]
    
    for sim_scenario in coarse_sims: 
        for station_name in stations: 
            sim_col = station_name + sim_scenario
            obs_col = station_name + "_ghi_1M"
            metrics = compute_error_metrics(sim_vs_obs_sub, sim_col=sim_col, obs_col=obs_col, tolerance_pct=10)

            # Filter to keep only the selected metrics
            selected_keys = ["lambda", "delta", "sigma", "R2", "NSE", "Legates_E1", "KSI"]
            filtered_metrics = {
                k: {m: v for m, v in metrics[k].items() if m in selected_keys}
                for k in metrics
            }

            print(f"\n📊 {station_name} — {sim_scenario}:")
            print(pd.DataFrame(filtered_metrics).T)
    

    
    # Plot scatter Flesland sim - Flesland obs
    scatter_with_fit(
        sim=sim_vs_obs_sub["flesland_ghi_sim_horizontal_log"].values,
        obs=sim_vs_obs_sub["Flesland_ghi_1M_log"].values,
        xlabel=r"$\log\!\left(\frac{I_{\mathrm{sim}}}{\sigma_{\mathrm{sim}}}\right)$",
        ylabel=r"$\log\!\left(\frac{I_{\mathrm{obs}}}{\sigma_{\mathrm{obs}}}\right)$",
        title="Flesland Observed vs. Simulated Irradiance",
        outpath="output/scatter_flesland_sim_flesland_obs_log_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )

    # Plot scatter Florida sim - Florida obs
    scatter_with_fit(
        sim=sim_vs_obs_sub["florida_ghi_sim_horizontal_log"].values,
        obs=sim_vs_obs_sub["Florida_ghi_1M_log"].values,
        xlabel=r"$\log\!\left(\frac{I_{\mathrm{sim}}}{\sigma_{\mathrm{sim}}}\right)$",
        ylabel=r"$\log\!\left(\frac{I_{\mathrm{obs}}}{\sigma_{\mathrm{obs}}}\right)$",
        title="Florida Observed vs. Simulated Irradiance",
        outpath="output/scatter_florida_sim_florida_obs_log_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )
    
    # Plot scatter Florida sim - Florida sim
    scatter_with_fit(
        sim_vs_obs_sub["florida_ghi_sim_horizontal_log"].values,
        sim_vs_obs_sub["flesland_ghi_sim_horizontal_log"].values,
        xlabel="Florida log(I_sim)",
        ylabel="Flesland log(I_sim)",
        title="Simulated Irradiance at Florida vs. Flesland", 
        outpath="output/scatter_flesland_sim_florida_sim_log_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    )
    
    # Plot scatter Flesland obs - Florida obs 
    scatter_with_fit(
        sim_vs_obs_sub["Florida_ghi_1M_log"].values,
        sim_vs_obs_sub["Flesland_ghi_1M_log"].values,
        xlabel="Florida log(I_obs)",
        ylabel="Flesland log(I_obs)",
        title="Observed Irradiance at Florida vs. Flesland",
        outpath="output/scatter_flesland_obs_florida_obs_log_with_sky_type.png",
        sky_type=sim_vs_obs_sub["sky_type"]
    ) 
    
    # Convert to DOY for line plots    
    sim_vs_obs_sub["doy"] = sim_vs_obs_sub["date"].dt.dayofyear
  
    # Plot doy - Flesland sim, Flesland obs
    for year in range(2015, 2026):
        single_year = sim_vs_obs_sub[sim_vs_obs_sub["date"].dt.year == year]
        lineplot_doy(
            single_year["doy"],
            single_year["flesland_ghi_sim_horizontal"],
            single_year["Flesland_ghi_1M"],
            ylabel="GHI (W/m²)",
            title=f"DOY vs Flesland GHI ({year})",
            outpath=f"output/lineplot_flesland_sim_obs_doy_{year}.png"
        )
        # Plot doy - Florida sim, Florida obs  
        lineplot_doy(
            single_year["doy"],
            single_year["florida_ghi_sim_horizontal"],
            single_year["Florida_ghi_1M"],
            ylabel="GHI (W/m²)",
            title=f"DOY vs Florida GHI ({year})",
            outpath=f"output/lineplot_florida_sim_obs_doy_{year}.png"
        )   
    
    
if __name__ == "__main__": 
    main()
    
    
    