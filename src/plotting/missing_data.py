# =================================================================
# Plot the missing values/available data from ground stations 
# and sentinel-2 for the studied time period as heatmaps. 
# =================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar
from src.plotting import set_paper_style

set_paper_style()


def heatmap_missing_values_frost(
    frost_data_path,
    title="Missing values at Flesland and Florida stations 2015-2025 10:30-11:30 UTC",
    outpath="output/florida_flesland_na_2015-2025_heatmap.png"
):
    """Plot heatmap of missing values from ground stations (downloaded from frost api) for each 
    year and month of the study time period. """
    
    # Load data
    df = pd.read_csv(frost_data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    # Count available values
    grouped = (
        df.groupby(["station", "year", "month"])
          .agg(num_vals=("value", lambda x: x.notna().sum()))
          .reset_index()
    )

    # Compute theoretical max (= 60 * number_of_days_in_month)
    grouped["num_days"] = grouped.apply(
        lambda r: calendar.monthrange(r["year"], r["month"])[1], axis=1
    )
    grouped["max_possible"] = grouped["num_days"] * 60
    grouped["missing"] = grouped["max_possible"] - grouped["num_vals"]
    grouped["missing_pct"] = grouped["missing"] / grouped["max_possible"] * 100

    stations = ["Flesland", "Florida"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    im = None  # store last image handle

    for ax, station in zip(axes, stations):
        sub = grouped[grouped["station"] == station]

        # Full year×month grid
        years = sorted(sub["year"].unique())
        full_index = pd.MultiIndex.from_product(
            [years, range(1, 13)], names=["year", "month"]
        )

        heat = (
            sub.set_index(["year", "month"])["missing_pct"]
               .reindex(full_index)
               .unstack(level=0)
        )

        heat = heat.fillna(100)

        # plot heatmap
        im = ax.imshow(heat.values, aspect="auto", origin="upper", vmin=0, vmax=100)

        ax.set_title(station)
        ax.set_xlabel("Year")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns)

        ax.set_yticks(range(12))
        ax.set_yticklabels(range(1, 13))
        ax.set_ylabel("Month")

    # --------------------------------------------------------
    # Colorbar positioned OUTSIDE to the right of both subplots
    # --------------------------------------------------------
    # Add a new axes to the right of the last subplot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Missing percentage (%)")

    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0, 0.9, 0.95])  # leave space for colorbar
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"Figure was saved to {outpath}")


def heatmap_available_values_s2(
    s2_data_path,
    title="Number of available Sentinel-2 observations per month",
    outpath="output/s2_available_obs_heatmap.png"
):
    """Plot available data from Sentinel-2 for each month and year of the 
    studied time period as heatmap. """
    # Load S2 data
    df = pd.read_csv(s2_data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Count available S2 observations per (year, month)
    grouped = (
        df.groupby(["year", "month"])
          .agg(num_vals=("date", "count"))
          .reset_index()
    )

    # Build full grid of years × months
    years = sorted(grouped["year"].unique())
    full_index = pd.MultiIndex.from_product(
        [years, range(1, 13)], names=["year", "month"]
    )

    heat = (
        grouped.set_index(["year", "month"])["num_vals"]
               .reindex(full_index)      # include months without observations
               .unstack(level=0)         # columns = years
    )

    # If no obs in a month → fill with zero
    heat = heat.fillna(0)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Different colormap to make clear this shows data availability
    im = ax.imshow(
        heat.values,
        aspect="auto",
        origin="upper",
        cmap="inferno"
    )

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)

    ax.set_yticks(range(12))
    ax.set_yticklabels(range(1, 13))
    ax.set_ylabel("Month")

    fig.colorbar(im, ax=ax, label="Number of S2 observations")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"Figure was saved to {outpath}")


def main():
    frost_data = "data/processed/frost_ghi_1M_Flesland_Florida_10:30-11:30UTC.csv"
    s2_data = "data/processed/s2_cloud_cover_large_thresh_40.csv"
    heatmap_missing_values_frost(frost_data)
    heatmap_available_values_s2(s2_data)

if __name__ == "__main__":
    main()
