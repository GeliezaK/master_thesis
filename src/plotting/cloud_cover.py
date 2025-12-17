# ====================================================================
# Plotting functions to compare small vs. large ROI cloud cover 
# And satellite vs stations cloud cover. 
# ====================================================================
# The cloud cover data from Florida and Flesland stations can be downloaded
# manually via the https://seklima.met.no/ api. 

import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

colors = {'s2': '#F0A202', 'Flesland': '#550527', 'Florida': '#A10702'}


def plot_hist_cloud_cover(df, title, outpath,
                          cloud_cover_variable="cloud_cover", 
                          mixed_thresh = 1.0, overcast_thresh = 99.0):
    # Plot histogram of cloud cover
    print(df[cloud_cover_variable].describe())
    total_obs = df[cloud_cover_variable].count()
    print("Total observations:", total_obs)

    print("Overcast sky threshold: ", mixed_thresh)
    print("mixed sky threshold: ", overcast_thresh)
    n_overcast = df[df[cloud_cover_variable] >= overcast_thresh].shape[0]
    n_mixed = df[(df[cloud_cover_variable] > mixed_thresh) & (df[cloud_cover_variable] < overcast_thresh)].shape[0]
    n_clear = df[(df[cloud_cover_variable] <= mixed_thresh)].shape[0]
    print("Number of overcast obs: ", n_overcast)
    print("percentage overcast obs: ", n_overcast/total_obs)
    print("Number of mixed obs: ", n_mixed)
    print("percentage mixed obs: ", n_mixed/total_obs)
    print("Number of clear obs: ", n_clear)
    print("percentage clear obs: ", n_clear/total_obs)

    plt.figure(figsize=(10, 6))
    plt.hist(df[cloud_cover_variable], bins=range(0, 102), edgecolor='black', align='left')
    plt.axvline(overcast_thresh, ls = '--', color='gray')
    plt.axvline(mixed_thresh, ls = '--', color='gray')
    plt.title(title)
    plt.xlabel("Cloud Cover (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(outpath)
    print(f"Saved histogramm to {outpath}")

def plot_monthly_cloud_cover(df):
    # Monthly averages 
    monthly_avg = df.groupby('month')['cloud_cover'].mean()
    monthly_avg.plot(kind='bar', title='Average Cloud Cover per Month', ylabel='% Cloud Cover')
    plt.tight_layout()
    plt.savefig("output/cloud_cover_monthly_avg.png")

def plot_monthly_cloud_cover_boxplots(df):
    # Monthly boxplots 
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='month', y='cloud_cover', data=df)
    plt.title('Cloud Cover Distribution per Month')
    plt.xlabel('Month')
    plt.ylabel('% Cloud Cover')
    plt.tight_layout()
    plt.savefig("output/cloud_cover_monthly_boxplots.png")

def plot_seasonal_cloud_cover_boxplots(df):
    # Boxplot per season
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='season', y='cloud_cover', data=df)
    plt.title('Cloud Cover Distribution per Season')
    plt.xlabel('Season')
    plt.ylabel('% Cloud Cover')
    plt.tight_layout()
    plt.savefig("output/cloud_cover_seasonal_boxplot.png")

def plot_yearly_cloud_cover(df):
    # Yearly trends 
    plt.figure()
    yearly_avg = df.groupby('year')['cloud_cover'].mean()
    yearly_avg.plot(marker='o', title='Average Cloud Cover per Year', ylabel='% Cloud Cover')
    plt.tight_layout()
    plt.savefig("output/cloud_cover_yearly_avg.png")

def plot_doy_cloud_cover(df):
    # Doy cloud cover 
    plt.figure()
    yearly_avg = df.groupby('doy')['cloud_cover'].mean()
    yearly_avg.plot(marker='o', title='Average Cloud Cover per Day of Year', ylabel='% Cloud Cover')
    plt.tight_layout()
    plt.savefig("output/cloud_cover_doy_avg.png")


def plot_yearly_cloud_cover(df):
    # Plot for multiple years in several subplots
    years = sorted(df['year'].unique())
    n_years = len(years)

    # Set up 2-column subplot layout
    ncols = 3
    nrows = math.ceil(n_years / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.5 * nrows), sharex=True)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot for each year
    for i, year in enumerate(years):
        ax = axes[i]
        df_year = df[df['year'] == year]

        monthly_avg = df_year.groupby('month').agg({
            'cloud_cover': 'mean',
            'avg_cloud_prob': 'mean',
            'doy': 'count'
        }).rename(columns={'doy': 'count'})

        months = sorted(df_year['month'].unique())

        ax.plot(months, monthly_avg['cloud_cover'], label='Cloud Cover (%)', marker='o')
        ax.plot(months, monthly_avg['avg_cloud_prob'], label='Avg Cloud Prob (%)', marker='s')
        ax.bar(months, monthly_avg['count'], alpha=0.3, label='Data Points', color='gray')

        ax.set_title(f"{year}")
        ax.set_xticks(months)
        ax.set_xlim(1, 12)
        ax.set_ylabel("Value")
        ax.legend(loc='lower right', fontsize='small')

    # Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.xlabel("Month")
    plt.suptitle("Monthly Cloud Cover and Cloud Probability (2015–2025)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("output/yearly_monthly_cloud_cover_avg_cloud_prob.png")


def plot_mean_with_std_cloud_cover_s2_vs_stations(combined_df):  
    # Group by month and compute mean and std
    monthly_stats = combined_df.groupby('month').agg({
        's2_cloud_cover': ['mean', 'std'],
        'Florida_cloud_cover': ['mean', 'std'],
        'Flesland_cloud_cover': ['mean', 'std']
    })

    # Flatten MultiIndex columns
    monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
    months = monthly_stats.index

    # Plot setup
    plt.figure(figsize=(10, 6))

    # Sentinel-2
    plt.plot(months, monthly_stats["s2_cloud_cover_mean"], label="Sentinel-2", color=colors['s2'], marker='o', markersize=6)
    plt.fill_between(months,
                    monthly_stats["s2_cloud_cover_mean"] - monthly_stats["s2_cloud_cover_std"],
                    monthly_stats["s2_cloud_cover_mean"] + monthly_stats["s2_cloud_cover_std"],
                    color=colors['s2'], alpha=0.2)

    # Flesland
    plt.plot(months, monthly_stats["Flesland_cloud_cover_mean"], label="Flesland", color=colors['Flesland'], marker='x', markersize=6)
    plt.fill_between(months,
                    monthly_stats["Flesland_cloud_cover_mean"] - monthly_stats["Flesland_cloud_cover_std"],
                    monthly_stats["Flesland_cloud_cover_mean"] + monthly_stats["Flesland_cloud_cover_std"],
                    color=colors['Flesland'], alpha=0.2)

    # Florida
    plt.plot(months, monthly_stats["Florida_cloud_cover_mean"], label="Florida", color=colors['Florida'], marker='x', markersize=6)
    plt.fill_between(months,
                    monthly_stats["Florida_cloud_cover_mean"] - monthly_stats["Florida_cloud_cover_std"],
                    monthly_stats["Florida_cloud_cover_mean"] + monthly_stats["Florida_cloud_cover_std"],
                    color=colors['Florida'], alpha=0.2)



    # Styling
    plt.title("Monthly Average Cloud Cover and Standard Deviation — Florida, Flesland & Sentinel-2")
    plt.xlabel("Month")
    plt.ylabel("Cloud Cover (%)")
    plt.ylim(0,100)
    plt.xticks(ticks=range(1, 13), labels=[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/mean_with_std_cloud_cover_Florida_vs_Flesland_vs_Satellite_pct_PT10M-PT1H.png")

 
def plot_yearly_monthly_cloud_cover_s2_vs_stations(combined_df):
    # Plot for multiple years in several subplots
    years = sorted(combined_df['year'].unique())
    n_years = len(years)

    # Set up 2-column subplot layout
    ncols = 3
    nrows = math.ceil(n_years / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.5 * nrows), sharex=True)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot for each year
    for i, year in enumerate(years):
        ax = axes[i]
        df_year = combined_df[combined_df['year'] == year]

        # Compute monthly mean and std
        monthly_stats = df_year.groupby('month').agg({
            's2_cloud_cover': ['mean', 'std'],
            'Flesland_cloud_cover': ['mean', 'std'],
            'Florida_cloud_cover': ['mean', 'std'],
            'doy': 'count'
        })
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        monthly_stats = monthly_stats.rename(columns={'doy_count': 'count'})
        
        months = sorted(df_year['month'].unique())

        # Plot means
        ax.plot(months, monthly_stats['s2_cloud_cover_mean'], label='S2', marker='o', color=colors['s2'])
        ax.plot(months, monthly_stats['Flesland_cloud_cover_mean'], label='Flesland', marker='x', color=colors['Flesland'])
        ax.plot(months, monthly_stats['Florida_cloud_cover_mean'], label='Florida', marker='x', color=colors['Florida'])

        # Plot std deviation shaded bands
        ax.fill_between(months,
                        monthly_stats['s2_cloud_cover_mean'] - monthly_stats['s2_cloud_cover_std'],
                        monthly_stats['s2_cloud_cover_mean'] + monthly_stats['s2_cloud_cover_std'],
                        color=colors['s2'], alpha=0.2)

        ax.fill_between(months,
                        monthly_stats['Flesland_cloud_cover_mean'] - monthly_stats['Flesland_cloud_cover_std'],
                        monthly_stats['Flesland_cloud_cover_mean'] + monthly_stats['Flesland_cloud_cover_std'],
                        color=colors['Flesland'], alpha=0.2)

        ax.fill_between(months,
                        monthly_stats['Florida_cloud_cover_mean'] - monthly_stats['Florida_cloud_cover_std'],
                        monthly_stats['Florida_cloud_cover_mean'] + monthly_stats['Florida_cloud_cover_std'],
                        color=colors['Florida'], alpha=0.2)
    
        # Plot bar for observation count
        #ax.bar(months, monthly_stats['count'], label='# obs', color='lightgray', alpha=0.4)

        ax.set_title(f"{year}")
        ax.set_xticks(range(1,13))
        ax.set_xlim(1, 12)
        ax.set_ylim(0,100)
        ax.set_ylabel("Cloud Cover (%)")

    # Add legend in the last (empty) subplot if there is one
    n_subplots = nrows * ncols
    n_used = len(years)

    if n_used < n_subplots:
        empty_ax = axes[-1]  # Last subplot (row 4, col 3 if 4x3)
        empty_ax.axis('off')  # Hide axis
        legend_handles = [
            plt.Line2D([0], [0], color=colors['s2'], label='S2', marker='o', linestyle='-'),
            plt.Line2D([0], [0], color=colors['Flesland'], label='Flesland', marker='x', linestyle='-'),
            plt.Line2D([0], [0], color=colors['Florida'], label='Florida', marker='x', linestyle='-'),
            plt.Rectangle((0, 0), 1, 1, color='lightgray', alpha=0.4, label='+- 1 std')
        ]

        empty_ax.legend(
            handles=legend_handles,
            loc='center',
            fontsize=16,              # bigger font
            title='Legend',
            title_fontsize=18,        # bigger title
            frameon=False,
            markerscale=2,
            handlelength=2
        )

    # Remove any remaining unused axes (if more than one empty)
    for j in range(n_used, n_subplots - 1):  # exclude the last one
        fig.delaxes(axes[j])


    plt.xlabel("Month")
    plt.suptitle("Monthly Cloud Cover and Uncertainty (2015–2025)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("output/yearly_monthly_cloud_cover_s2_vs_florida_flesland_with_uncertainty.png")


def correlate_s2_vs_stations(combined_df):
    # Correlations analyses between s2, Florida and Flesland
    res_s2_florida = spearmanr(combined_df['s2_cloud_cover'], combined_df['Florida_cloud_cover'], alternative='greater')
    res_s2_flesland = spearmanr(combined_df['s2_cloud_cover'], combined_df['Flesland_cloud_cover'], alternative='greater')
    res_florida_flesland = spearmanr(combined_df['Florida_cloud_cover'], combined_df['Flesland_cloud_cover'], alternative='greater')

    print("S2 vs. Florida: ", res_s2_florida.statistic, res_s2_florida.pvalue)
    print("S2 vs. Flesland: ", res_s2_flesland.statistic, res_s2_flesland.pvalue)
    print("Florida vs Flesland: ", res_florida_flesland.statistic, res_florida_flesland.pvalue)

    # Data references
    s2 = combined_df['s2_cloud_cover']
    florida = combined_df['Florida_cloud_cover']
    flesland = combined_df['Flesland_cloud_cover']

    def compute_metrics(y_true, y_pred):
        r, p = spearmanr(y_true, y_pred, alternative='greater')
        mae = mean_absolute_error(y_true, y_pred)
        mbe = np.median(y_pred - y_true)
        rmse = root_mean_squared_error(y_true, y_pred)
        return r, p, mae, mbe, rmse

    # Compute all metrics
    r1, p1, mae1, mbe1, rmse1 = compute_metrics(florida, s2)
    r2, p2, mae2, mbe2, rmse2 = compute_metrics(flesland, s2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # Common plot settings
    def make_scatter(ax, x, y, title, r, p, mae, mbe, rmse):
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.7)
        ax.plot([0, 100], [0, 100], ls='--', color='gray')  # Diagonal line
        ax.set_title(title)
        ax.set_xlabel("Sentinel-2 Cloud Cover (%)")
        ax.set_ylabel("Ground Cloud Cover (%)")
        ax.text(0.05, 0.90, f"ρ = {r:.2f}\np = {p:.2e}\nMAE = {mae:.2f}\nMBE = {mbe:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    make_scatter(axes[0], florida, s2, "S2 vs Florida", r1, p1, mae1, mbe1, rmse1)
    make_scatter(axes[1], flesland, s2, "S2 vs Flesland", r2, p2, mae2, mbe2, rmse2)

    plt.tight_layout()
    plt.savefig("output/correlation_s2_florida_flesland_cloud_cover.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    h1 = axes[0].hist2d(florida,s2, bins=(9, 20), cmap='Blues')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title(f"S2 vs Florida \n ρ = {r1:.2f}, p = {p1:.2e}\nMAE = {mae1:.2f}, MBE = {mbe1:.2f}, RMSE = {rmse1:.2f} ")
    axes[0].set_ylabel("Sentinel-2 cloud cover (%)")
    axes[0].set_xlabel("Florida cloud cover (oktas converted to %)")

    plt.colorbar(h1[3], ax=axes[0], label='Count', orientation='vertical')

    h2 = axes[1].hist2d(flesland, s2, bins=(9, 20), cmap='Greens')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title(f"S2 vs Flesland \n ρ = {r2:.2f}, p = {p2:.2e}\nMAE = {mae2:.2f}, MBE = {mbe2:.2f}, RMSE = {rmse2:.2f}")
    axes[1].set_xlabel("Flesland cloud cover (oktas converted to %)")
    plt.colorbar(h2[3], ax=axes[1], label='Count', orientation='vertical')

    plt.tight_layout()
    plt.savefig("output/correlation_hist_2D_s2_florida_flesland_cloud_cover.png")


# Scatter plot function
def scatter_compare(df, col_x, col_y, title="Cloud Cover Comparison", outpath=None):
    plt.figure(figsize=(7,7))
    plt.scatter(df[col_x], df[col_y], alpha=0.6, edgecolor="k")

    # Diagonal line
    min_val = min(df[col_x].min(), df[col_y].min())
    max_val = max(df[col_x].max(), df[col_y].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Fit")

    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Saved scatter to {outpath}.")
    plt.close()

    # Pearson correlation
    corr = df[col_x].corr(df[col_y])
    print(f"Correlation between {col_x} and {col_y}: {corr:.3f}")
    
 
if __name__ == "__main__": 
    # To plot small vs large roi cloud cover: 
    merged_s2_cloud_cover_filepath = "data/processed/s2_cloud_cover_large_thresh_40.csv"
    # to plot stations vs s2 cloud cover: 
    stations_s2_cloud_cover_filepath = "data/processed/cloud_cover_2015-06-01_2025-05-01_s2_Flesland_Florida_paired.csv"

    # Load merged dataset
    # required columns: "system:time_start_large", "cloud_cover_large", "cloud_cover_small", "start_time_range_small", "start_time_range_large"
    df = pd.read_csv(merged_s2_cloud_cover_filepath)

    # Convert to datetime
    df["datetime"] = pd.to_datetime(df["system:time_start_large"], unit="ms")

    # Extract only time of day
    df["time_of_day"] = df["datetime"].dt.time

    # Get min and max time of day
    min_time = df["time_of_day"].min()
    max_time = df["time_of_day"].max()

    print("Min time of day:", min_time)
    print("Max time of day:", max_time)
    
    print(df.head())
    print(df[["time_of_day", "cloud_cover_large", "start_time_range_small", "start_time_range_large"]].describe())
    
    
    # Run scatter comparison
    scatter_compare(df, "cloud_cover_small", "cloud_cover_large", 
                    title="Scatter Plot: Cloud Cover Small vs Large ROI (2015-2025)", 
                    outpath="output/cloud_cover_scatter_thresh_40.png")
    plot_hist_cloud_cover(df, "Small ROI Cloud Cover Distribution", 
                          "output/s2_cloud_cover_small_roi_hist_thresh_40.png",
                          "cloud_cover_small")
    
    plot_hist_cloud_cover(df, "Large ROI Cloud Cover Distribution", 
                          "output/s2_cloud_cover_large_roi_hist_thresh_40.png",
                          "cloud_cover_large")  