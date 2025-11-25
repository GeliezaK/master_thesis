import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helper function: plot histogram for a subset with mean & sd
# ============================================================
def plot_hist_with_stats(ax, data, title):
    """Plot histogram on the given axis with mean and mean ± sd lines."""
    if len(data) == 0:
        ax.set_title(f"{title}\n(no data)")
        return

    mean_val = data.mean()
    sd_val = data.std()

    # Histogram
    ax.hist(data, bins=30, edgecolor="black")

    # Vertical lines
    ax.axvline(mean_val, color="red", linewidth=2)
    ax.axvline(mean_val - sd_val, color="grey", linestyle="--", linewidth=1)
    ax.axvline(mean_val + sd_val, color="grey", linestyle="--", linewidth=1)

    ax.set_title(title)

    # Print stats
    print(f"{title}: mean = {mean_val:.6f}, sd = {sd_val:.6f}")


# ============================================================
# Plot histograms per sky_type in 3 subplots
# ============================================================
def plot_histograms_by_sky_type(df, outpath="output/clear_sky_index_histogram_sky_type.png"):
    sky_types = ["clear", "mixed", "overcast"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, sky in zip(axes, sky_types):
        subset = df[df["sky_type"] == sky]["mean_clear_sky_index"]
        plot_hist_with_stats(ax, subset, title=f"Sky type: {sky}")

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved histogram to {outpath}.")


# ============================================================
# Plot per month and sky type (36 subplots)
# ============================================================
def plot_histograms_by_month_and_sky(df, outpath="output/clear_sky_index_histogram_monthly_sky_type.png"):
    sky_types = ["clear", "mixed", "overcast"]
    months = range(1, 13)

    fig, axes = plt.subplots(12, 3, figsize=(12, 28))
    fig.suptitle("Histograms by Month and Sky Type", y=1.02, fontsize=16)

    for i, month in enumerate(months):
        for j, sky in enumerate(sky_types):
            ax = axes[i, j]
            subset = df[(df["month"] == month) & (df["sky_type"] == sky)]["mean_clear_sky_index"]
            title = f"Month {month}, {sky}"
            plot_hist_with_stats(ax, subset, title)

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved histogram to {outpath}.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    area_mean_k_path = "data/processed/area_mean_clear_sky_index_per_obs.csv"
    df = pd.read_csv(area_mean_k_path)

    # Ensure correct types if needed
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["month"].astype(int)

    print("=== Plotting per sky type ===")
    plot_histograms_by_sky_type(df)

    print("\n=== Plotting per month and sky type ===")
    plot_histograms_by_month_and_sky(df)
