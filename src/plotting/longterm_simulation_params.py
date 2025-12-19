# ====================================================================
# Plot distribution of clear sky index for sky type and month. 
# Plot probability (with uncertainty) for each sky type per month. 
# ====================================================================

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.plotting import set_paper_style, SKY_TYPE_COLORS
from src.model.longterm_GHI_simulation import sample_dirichlet

set_paper_style()

# -------------------------------------------------------------
# Helper function: plot histogram for a subset with mean & sd
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Plot histograms per sky_type in 3 subplots
# -------------------------------------------------------------
def plot_histograms_by_sky_type(df, outpath="output/clear_sky_index_histogram_sky_type.png"):
    """Plot histograms of clear sky index per sky type."""
    sky_types = ["clear", "mixed", "overcast"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, sky in zip(axes, sky_types):
        subset = df[df["sky_type"] == sky]["mean_clear_sky_index"]
        plot_hist_with_stats(ax, subset, title=f"Sky type: {sky}")

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved histogram to {outpath}.")


# -------------------------------------------------------------
# Plot per month and sky type (36 subplots)
# -------------------------------------------------------------
def plot_histograms_by_month_and_sky(df, outpath="output/clear_sky_index_histogram_monthly_sky_type.png"):
    """Plot histogram of clear sky index per sky type and month."""
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


def plot_sky_type_probs_with_sd(samples_by_month, outpath="output/sky_type_probabilities_uncertainty.png"):
    """
    Plot monthly sky type probabilities with uncertainty (mean ± standard deviation) as grouped bars.

    Parameters
    ----------
    samples_by_month : dict
        Dictionary mapping month (int) to a NumPy array of Dirichlet samples
        with shape (n_samples, 3) for [clear, mixed, overcast].
    outpath : str, optional
        File path to save the plot (default: "output/sky_type_probabilities_uncertainty.png").

    Outputs
    -------
    Saves a grouped bar plot showing posterior mean and standard deviation of sky type probabilities for each month.
    """
    
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

    # Convert to arrays for easier arithmetic
    clear_mean = np.array(clear_mean)
    mixed_mean = np.array(mixed_mean)
    over_mean = np.array(over_mean)

    clear_sd   = np.array(clear_sd)
    mixed_sd   = np.array(mixed_sd)
    over_sd    = np.array(over_sd)

    # Prepare grouped bar positions
    x = np.arange(len(months))
    bar_width = 0.25

    plt.figure(figsize=(14,6))

    # Clear bars
    plt.bar(
        x - bar_width, clear_mean, 
        width=bar_width,
        yerr=clear_sd, capsize=4,
        label="Clear", 
        color=SKY_TYPE_COLORS["clear"]
    )

    # Mixed bars
    plt.bar(
        x, mixed_mean,
        width=bar_width,
        yerr=mixed_sd, capsize=4,
        label="Mixed",
        color=SKY_TYPE_COLORS["mixed"]
    )

    # Overcast bars
    plt.bar(
        x + bar_width, over_mean,
        width=bar_width,
        yerr=over_sd, capsize=4,
        label="Overcast",
        color=SKY_TYPE_COLORS["overcast"]
    )

    # Formatting
    plt.xticks(x, months)
    plt.xlabel("Month")
    plt.ylabel("Posterior Probability (mean ± sd)")
    plt.title("Posterior Uncertainty of Monthly Sky Type Probabilities (Dirichlet)")
    plt.legend(loc="center right", bbox_to_anchor=(1.2, 0.5))

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved grouped uncertainty sky type probability plot to {outpath}.")


if __name__ == "__main__":
    area_mean_k_path = "data/processed/area_mean_clear_sky_index_per_obs.csv"
    monthly_sky_type_counts_filepath = "data/processed/monthly_sky_type_counts.csv"

    df = pd.read_csv(area_mean_k_path)

    # Ensure correct types if needed
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["month"].astype(int)

    print("=== Plotting per sky type ===")
    plot_histograms_by_sky_type(df)

    print("\n=== Plotting per month and sky type ===")
    plot_histograms_by_month_and_sky(df)

    samples_by_month = sample_dirichlet(monthly_sky_type_counts_filepath, 5000)
    plot_sky_type_probs_with_sd(samples_by_month)