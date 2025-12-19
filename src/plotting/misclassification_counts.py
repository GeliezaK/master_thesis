# ===============================================================================
# Plot number/percentage of misclassifications for different coarse resolutions
# and cloud cover percentages. 
# ===============================================================================

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

from src.plotting import set_paper_style

set_paper_style()

def plot_misclassification_vs_scale(misclassification_counts_csv_filepath, outpath):
    """
    Plot misclassification percentage of cloud classification as a function of 
    deviation from 50% cloud cover, for multiple spatial resolutions.

    Parameters
    ----------
    misclassification_counts_csv_filepath : str
        Path to a CSV file containing columns:
        - 'date': observation date
        - 'cloud_cover_10m': cloud cover percentage at 10m scale
        - 'misclassified_percentage': percentage of misclassified pixels
        - 'resolution': spatial resolution in meters
    outpath : str
        File path to save the generated plot.

    Outputs
    -------
    Saves a line plot with shaded 95% confidence intervals showing median 
    misclassification versus |cloud_cover - 50| for each resolution.
    Includes a theoretical maximum misclassification reference line.
    """

    # Load data
    df = pd.read_csv(misclassification_counts_csv_filepath)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # -------------------------
    # Compute distance from 50%
    # -------------------------
    df["dist_from_50"] = np.abs(df["cloud_cover_10m"] - 50)

    # Ensure resolution is treated as categorical
    df["resolution"] = df["resolution"].astype(int)
    resolutions = sorted(df["resolution"].unique())
    print(f"Unique resolutions: {resolutions}")

    # -------------------------
    # Define bins in (abs(cloud_cover - 50))
    # -------------------------
    n_bins = 10
    bins = np.linspace(0, df["dist_from_50"].max(), n_bins + 1)
    df["bin"] = pd.cut(df["dist_from_50"], bins=bins, include_lowest=True)

    # Midpoints for plotting on x-axis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    print(f"bin_centers: {bin_centers}")

    # -------------------------
    # Plot: one line per resolution
    # -------------------------
    plt.figure(figsize=(10, 6))

    for res in resolutions:
        sub = df[df["resolution"] == res]

        grouped = sub.groupby("bin")["misclassified_percentage"]

        med = grouped.median().values
        std = grouped.std().values
        counts = grouped.count().values
        print(f"medians {res}m: {med}")
        print(f"stds {res}m: {std}")
        print(f"counts {res}m: {counts}")

        # 95% CI: ± 1.96 * std/sqrt(n)
        ci95 = 1.96 * std / np.sqrt(counts)
        print(f"ci95 {res}m : {ci95}")

        # Plot median line
        plt.plot(
            bin_centers,
            med,
            marker="o",
            label=f"{res} m",
            linewidth=2
        )

        # Shaded CI band
        plt.fill_between(
            bin_centers,
            med - ci95,
            med + ci95,
            alpha=0.25
        )

    # Add theoretical maximum of misclassifications
    x_max = range(0,51)
    y_max = range(50,-1, -1)
    plt.plot(x_max, y_max, marker="", linewidth=2, ls="--", color="grey", label="Max")
    plt.xlabel("|Cloud Cover - 50| (%)")
    plt.ylabel("Misclassified Percentage")
    plt.title("Misclassification vs Cloud-Cover Distance from 50% (per resolution)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Resolution")

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Figure saved to {outpath}.")

    
if __name__ == "__main__": 
    misclassification_count_csv = "data/processed/misclassification_counts_upscaled_cloud_mask.csv"
    
    plot_misclassification_vs_scale(misclassification_count_csv, "output/misclassification_count_cloud_cover.png")
