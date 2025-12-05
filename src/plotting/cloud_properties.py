import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
from src.plotting import CLOUD_PROP_COLORS, DEFAULT_ALPHA, set_paper_style
from src.model.generate_LUT import COT, CLOUD_TOP_HEIGHT, CLOUD_GEOGRAPHICAL_THICKNESS

set_paper_style()

def plot_range_and_median(cloud_props_filepath, outpath, var_name="cgt", roi="small",
                          title="Cloud Geometrical Thickness", ylabel="Thickness (m)"): 
    """Plot min max range and median histogram of CLAAS-3 cloud property value. """
    cloud_props = pd.read_csv(cloud_props_filepath)
    suffixes = [f"_min_{roi}", f"_max_{roi}", f"_median_{roi}"]
    col_names = [var_name + suf for suf in suffixes] + ["date"]
    cloud_props = cloud_props[col_names]
    # Parse date and extract month
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    cloud_props["month"] = cloud_props["date"].dt.month
    
    # Group by month and compute mean of min, max, median
    monthly_stats = cloud_props.groupby("month")[[var_name + s for s in suffixes]].mean()
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(monthly_stats.index, monthly_stats[var_name + f"_min_{roi}"], label="Min", marker="o", ls="--")
    plt.plot(monthly_stats.index, monthly_stats[var_name + f"_median_{roi}"], label="Median", marker="x", ls='-')
    plt.plot(monthly_stats.index, monthly_stats[var_name + f"_max_{roi}"], label="Max", marker="^", ls="--")
    
    # Reference line at 500, 1000 m
    if var_name == "cgt": 
        lut_bins = CLOUD_GEOGRAPHICAL_THICKNESS * 1000
    elif var_name == "cot": 
        lut_bins = COT
    elif var_name == "cth":
        lut_bins = [x * 1000 for x in CLOUD_TOP_HEIGHT]
        
    for i, bin_value in enumerate(lut_bins): 
        print(bin_value)
        if i == 0:
            plt.axhline(y=bin_value, color="red", linestyle=":", linewidth=1, label="LUT bins")
        else:
            plt.axhline(y=bin_value, color="red", linestyle=":", linewidth=1)
            
    # Labels and title
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1,13))
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}.")
    
def plot_hists_and_bins(cloud_props_filepath, outpath, roi="small"):
    var_names = [f"cgt_median_{roi}", f"cth_median_{roi}", f"cot_median_{roi}"]
    cloud_props = pd.read_csv(cloud_props_filepath)
    col_names = var_names + ["date"]
    cloud_props = cloud_props[col_names]
    cloud_props["date"] = pd.to_datetime(cloud_props["date"], format="%Y-%m-%d")
    cloud_props["month"] = cloud_props["date"].dt.month

    # Mapping for LUT bins
    lut_map = {
        "cgt": [CLOUD_GEOGRAPHICAL_THICKNESS * 1000],
        "cth": [x * 1000 for x in CLOUD_TOP_HEIGHT],
        "cot": COT
    }
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, full_var_name in zip(axes, var_names):
        # Extract base name (cgt, cth, cot)
        var_base = full_var_name.split("_")[0]
        
        # Data
        data = cloud_props[full_var_name].dropna()
        
        # Histogram
        ax.hist(data, bins=50, color=CLOUD_PROP_COLORS[var_base], edgecolor="black", alpha=DEFAULT_ALPHA)
        
        # LUT bin lines
        for i, bin_value in enumerate(lut_map[var_base]):
            if i == 0:
                ax.axvline(x=bin_value, color="red", linestyle=":", linewidth=1, label="LUT bins")
            else:
                ax.axvline(x=bin_value, color="red", linestyle=":", linewidth=1)
        
        # Titles and labels
        ax.set_title(f"{var_base.upper()} median")
        ax.set_xlabel(var_base.upper())
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    
    print(f"✅ Histogram plots saved to {outpath}")
    

if __name__ == "__main__": 
    s2_csv = "data/processed/s2_cloud_cover_table_small_and_large_with_cloud_props.csv"
    var_name = "cth"
    roi = "small"
    cloud_props = pd.read_csv(s2_csv)
    # Select columns of interest and drop rows with any NaNs
    cols = ["cot_median_small", "cgt_median_small", "cth_median_small", "cph_median_small",
            "cot_median_large", "cgt_median_large", "cth_median_large", "cph_median_large"]
    # Count missing values
    missing_counts = cloud_props[cols].isna().sum()

    print("Number of missing values per variable:")
    print(missing_counts)
    
    subset = cloud_props[["cot_median_small", "cgt_median_small", "cth_median_small", "cph_median_small"]].dropna()
    print(subset.describe())
    # Compute correlation matrix (Pearson by default)
    corr_matrix = subset.corr()

    print("Pairwise correlation matrix:")
    print(corr_matrix)
    
    # Define dependent and independent variables
    y = subset["cgt_median_small"]
    X = subset[["cot_median_small", "cth_median_small", "cph_median_small"]]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit linear regression
    model = sm.OLS(y, X).fit()

    print(model.summary())
    #plot_range_and_median(s2_csv, f"output/{var_name}_{roi}_range_and_median_monthly.png", 
    #                      var_name=var_name, roi=roi)
    #plot_hists_and_bins(s2_csv, f"output/cloud_props_hist_lut_bins.png", roi="small")