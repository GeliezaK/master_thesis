import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import calendar
import statsmodels.api as sm
from scipy.stats import wasserstein_distance
from preprocessing.quality_control_flag_stations_data import flag_observations
from preprocessing.merge_station_obs_with_sim import extract_pixel_by_location
from src.plotting import STATION_COLORS, set_paper_style, SKY_TYPE_COLORS

set_paper_style()

def flag_stations_data(stations_data_path):
    # Load full dataset
    stations_df = pd.read_csv(stations_data_path)
    stations_df["datetime"] = pd.to_datetime(stations_df["timestamp"], utc=True)

    # Work on a copy so we don't modify original accidentally
    df_out = stations_df.copy()

    # Prepare a list to collect flagged subsets
    flagged_parts = []

    # Loop through stations
    for station_name in ["Flesland", "Florida"]:
        print(f"\n--- Processing station: {station_name} ---")
        
        # Extract subset
        df_station = stations_df[stations_df["station"] == station_name].copy()
        
        # ----------------------------------
        # Print number of missing values 
        # ----------------------------------
        # Ensure datetime is sorted
        df_station = df_station.sort_values("datetime")
        
        # Compute theoretical number of hourly observations
        start = df_station["datetime"].iloc[0]
        end = df_station["datetime"].iloc[-1]
        
        # total hours between start and end, inclusive
        theoretical_n_obs = int(((end - start).total_seconds() / 3600) + 1)
        
        # Missing values
        n_missing = theoretical_n_obs - len(df_station)
        
        print(f"Station: {station_name}")
        print(f"Start: {start}, End: {end}")
        print(f"Theoretical number of hourly observations: {theoretical_n_obs}")
        print(f"Number of missing values: {n_missing}")

        # Apply flagging on this subset
        df_station = flag_observations(
            df_station,
            obs_col="value",
            datetime_col="datetime",
        )

        flagged_parts.append(df_station)

    # Concatenate all flagged station data
    flagged_all = pd.concat(flagged_parts, axis=0)

    # Merge flags back into full original dataframe by index
    df_out = df_out.merge(
        flagged_all[["value_flag"]],
        left_index=True,
        right_index=True,
        how="left"
    )
    
    return df_out


def plot_doy_vs_daily_mean_ghi(stations_data_path, daily_sim_monthly_path, daily_sim_annual_path):

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_path, parse_dates=["datetime"])
    sim_monthly = pd.read_csv(daily_sim_monthly_path)
    sim_annual = pd.read_csv(daily_sim_annual_path)

    # ---------------------------------------------------------
    # Prepare STATION daily GHI data
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["doy"] = stations["datetime"].dt.dayofyear
    stations["month"] = stations["datetime"].dt.month

    # Compute *daily* mean GHI for each station
    station_daily = (
        stations.groupby(["station", "date", "doy"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "GHI_daily_mean"})
    )
    station_daily["GHI_daily_kWh"] = station_daily["GHI_daily_mean"] * 24 / 1000 # convert to kWh

    # ---------------------------------------------------------
    # Compute percentiles per DOY for each station
    # ---------------------------------------------------------
    def compute_GHI_daily_Wh_percentiles(df):
        return (
            df.groupby("doy")["GHI_daily_kWh"]
            .agg(
                min=lambda x: np.nanmin(x),
                p25=lambda x: np.percentile(x, 25),
                p50=lambda x: np.percentile(x, 50),  
                p75=lambda x: np.percentile(x, 75),
                max=lambda x: np.nanmax(x),
            )
            .reset_index()
        )

    stat_flesland = compute_GHI_daily_Wh_percentiles(
        station_daily[station_daily["station"] == "Flesland"]
    )
    stat_florida = compute_GHI_daily_Wh_percentiles(
        station_daily[station_daily["station"] == "Florida"]
    )

    # ---------------------------------------------------------
    # Prepare SIMULATION daily GHI data
    # --------------------------------------------------------- 
    
    sim_monthly["GHI_daily_kWh"] = sim_monthly["GHI_daily_Wh"] / 1000 # convert to kWh
    sim_annual["GHI_daily_kWh"] = sim_annual["GHI_daily_Wh"] / 1000
    
    sim_monthly_stats = compute_GHI_daily_Wh_percentiles(sim_monthly)
    sim_annual_stats = compute_GHI_daily_Wh_percentiles(sim_annual)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))

    def add_line_and_shade(df, label, color):
        # median line
        plt.plot(df["doy"], df["p50"], color=color, label=label, linewidth=2)
        # shaded IQR
        #plt.fill_between(df["doy"], df["p25"], df["p75"], color=color, alpha=0.2)
        # outer percentile band
        plt.fill_between(df["doy"], df["min"], df["max"], color=color, alpha=0.12)

    # Stations
    add_line_and_shade(stat_flesland, "Flesland", "tab:red")
    add_line_and_shade(stat_florida, "Florida", "tab:orange")

    # Simulations
    add_line_and_shade(sim_monthly_stats, "Model Monthly Mean k", "tab:green")
    add_line_and_shade(sim_annual_stats, "Model Annual Mean k", "tab:blue")

    plt.xlabel("Day of Year")
    plt.ylabel("Daily Mean Solar Irradiation (kWh/m²)")
    plt.title("Daily Mean Solar Irradiation vs Day of Year — Stations vs. Simulations")
    
    # ---------------------------------------------------------
    # Add vertical month grid lines
    # ---------------------------------------------------------
    # DOY for first day of each month (non-leap year)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    for doy in month_starts:
        plt.axvline(doy, color="grey", linestyle="--", alpha=0.35, linewidth=1)

    # Add month labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for doy, label in zip(month_starts, month_labels):
        plt.text(doy + 1, plt.ylim()[1] * 0.98, label,
                 rotation=0, fontsize=12, color="dimgray", va="top")
    
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------
    outpath = "output/monthly_mean_ghi_vs_doy_longterm_sim_vs_stations_minmax.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Figure saved to {outpath}.")
    
    
    
def plot_doy_vs_daily_mean_ghi_10day(stations_data_path,
                                     daily_sim_model1_path,
                                     daily_sim_model2_path, 
                                     daily_sim_model3_path):

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_path, parse_dates=["datetime"])
    sim_model1 = pd.read_csv(daily_sim_model1_path)
    sim_model2 = pd.read_csv(daily_sim_model2_path)
    sim_model3 = pd.read_csv(daily_sim_model3_path)

    # ---------------------------------------------------------
    # Prepare STATION daily GHI data
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["doy"] = stations["datetime"].dt.dayofyear

    # Compute *daily* mean GHI per station
    station_daily = (
        stations.groupby(["station", "date", "doy"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "GHI_daily_mean"})
    )

    station_daily["GHI_daily_kWh"] = station_daily["GHI_daily_mean"] * 24 / 1000

    # ---------------------------------------------------------
    # Helper: Aggregate into 10-day bins
    # ---------------------------------------------------------
    def aggregate_10day(df, daily_ghi_var = "GHI_daily_kWh"):
        df = df.copy()
        df["doy_bin"] = ((df["doy"] - 1) // 10) * 10 + 1  # bins: 1, 11, 21, …
        
        return (
            df.groupby("doy_bin")[daily_ghi_var]
            .agg(
                median="median",
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75),
            )
            .reset_index()
            .rename(columns={"doy_bin": "doy"})
        )

    # Station 10-day stats
    stat_flesland = aggregate_10day(station_daily[station_daily["station"] == "Flesland"])
    stat_florida = aggregate_10day(station_daily[station_daily["station"] == "Florida"])

    # ---------------------------------------------------------
    # Prepare simulation daily GHI
    # ---------------------------------------------------------
    sim_model1["GHI_daily_kWh"] = sim_model1["GHI_daily_Wh"] / 1000
    sim_model2["GHI_daily_kWh"] = sim_model2["GHI_daily_Wh"] / 1000
    sim_model3["GHI_daily_kWh_Florida"] = sim_model3["Florida_GHI_daily_Wh_mean"] / 1000
    sim_model3["GHI_daily_kWh_Flesland"] = sim_model3["Flesland_GHI_daily_Wh_mean"] / 1000

    # Simulation 10-day stats
    sim_model1_stats  = aggregate_10day(sim_model1)
    sim_model2_stats = aggregate_10day(sim_model2)
    sim_model3_Florida_stats = aggregate_10day(sim_model3, daily_ghi_var="GHI_daily_kWh_Florida")
    sim_model3_Flesland_stats = aggregate_10day(sim_model3, daily_ghi_var="GHI_daily_kWh_Flesland")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))

    def add_line_with_errorbars(df, label, color, ls, ms):
        # Error size is asymmetric: (median - p25) downward, (p75 - median) upward
        yerr = [
            df["median"] - df["p25"],   # lower error
            df["p75"] - df["median"]    # upper error
        ]

        plt.errorbar(
            df["doy"], df["median"], yerr=yerr,
            markersize=6, marker=ms, linewidth=2, linestyle=ls,
            capsize=6, capthick=3, color=color, label=label
        )

    # Stations
    add_line_with_errorbars(stat_flesland, "Flesland", "tab:red", "-", "x")
    add_line_with_errorbars(stat_florida,  "Florida",  "tab:orange", "-", "x")

    # Simulations
    add_line_with_errorbars(sim_model1_stats,  "Model 1",  "tab:blue", "--", "o")
    add_line_with_errorbars(sim_model2_stats, "Model 2", "tab:green", "--", "o")
    add_line_with_errorbars(sim_model3_Flesland_stats, "Model 3: Flesland", "tab:brown", "--", "o")
    add_line_with_errorbars(sim_model3_Florida_stats, "Model 3: Florida", "tab:purple", "--", "o")


    # ---------------------------------------------------------
    # Add month grid lines
    # ---------------------------------------------------------
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    for doy in month_starts:
        plt.axvline(doy, color="grey", linestyle="--", alpha=0.35, linewidth=1)

    # Month labels
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ymax = plt.ylim()[1]
    for doy, label in zip(month_starts, month_labels):
        plt.text(doy + 1, ymax * 0.98, label, fontsize=12, color="dimgray", va="top")

    plt.xlabel("Day of Year (10-day bins)")
    plt.ylabel("Mean Daily Solar Irradiation (kWh/m²)")
    plt.title("10-Day Mean Solar Irradiation vs DOY — Stations vs Simulations")
    plt.grid(True, alpha=0.3)
    plt.legend()

    outpath = "output/daily_mean_ghi_vs_doy_10day_sim_vs_stations_median_with_model3.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Figure saved to {outpath}.")


def plot_monthly_boxplots_seasonal_monthly_variability(
    stations_data_path,
    sim_monthly_path,
    sim_annual_path,
    sim_spatial_path,
    outpath="output/seasonal_monthly_boxplots.png"
):

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_path, parse_dates=["datetime"])
    sim_monthly = pd.read_csv(sim_monthly_path)
    sim_annual = pd.read_csv(sim_annual_path)
    sim_spatial = pd.read_csv(sim_spatial_path)

    # ---------------------------------------------------------
    # Prepare station daily GHI data
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["month"] = stations["datetime"].dt.month
    stations["year"] = stations["datetime"].dt.year
    stations["GHI_daily_mean"] = stations.groupby(["station", "date"])["value"].transform("mean")
    stations["GHI_daily_kWh"] = stations["GHI_daily_mean"] * 24 / 1000
    
    station_monthly_means = stations.groupby(["station","year","month"])["GHI_daily_kWh"].mean().reset_index()

    # ---------------------------------------------------------
    # Prepare spatially uniform model daily data
    # ---------------------------------------------------------
    sim_monthly["GHI_daily_kWh"] = sim_monthly["GHI_daily_Wh"] / 1000
    sim_annual["GHI_daily_kWh"] = sim_annual["GHI_daily_Wh"] / 1000
    
    sim_monthly_means = sim_monthly.groupby(["year", "month"])["GHI_daily_kWh"].mean().reset_index()
    sim_annual_means = sim_annual.groupby(["year", "month"])["GHI_daily_kWh"].mean().reset_index()

    # ---------------------------------------------------------
    # Spatially resolved model: extract Florida + Flesland daily kWh
    # ---------------------------------------------------------
    sim_spatial["Florida_kWh"] = sim_spatial["Florida_GHI_daily_Wh_mean"] / 1000
    sim_spatial["Flesland_kWh"] = sim_spatial["Flesland_GHI_daily_Wh_mean"] / 1000

    # Only keep sky_type == "all-sky"
    sim_spatial_all = sim_spatial[sim_spatial["sky_type"] == "all-sky"]

    # ---------------------------------------------------------
    # Season definitions
    # ---------------------------------------------------------
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
    }

    colors = ["orange", "red", "green", "blue", "purple", "brown"]
    
    # ---------------------------------------------------------
    # Prepare figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    fig.subplots_adjust(hspace=0.35)

    # ---------------------------------------------------------
    # Loop through seasons
    # ---------------------------------------------------------
    for ax, (season_name, months) in zip(axes, seasons.items()):

        # Collect boxplot data:
        #   For each month in this season, create 6 datasets

        all_box_data = []
        x_labels = []

        for m in months:

            # 1. Station Florida
            fl_station = station_monthly_means[(station_monthly_means["station"] == "Florida") &
                                       (station_monthly_means["month"] == m)]["GHI_daily_kWh"]

            # 2. Station Flesland
            fs_station = station_monthly_means[(station_monthly_means["station"] == "Flesland") &
                                       (station_monthly_means["month"] == m)]["GHI_daily_kWh"]

            # 3. Spatially uniform model (monthly)
            sm = sim_monthly_means[sim_monthly_means["month"] == m]["GHI_daily_kWh"]

            # 4. Spatially uniform model (annual)
            sa = sim_annual_means[sim_annual_means["month"] == m]["GHI_daily_kWh"]

            # 5. Spatially resolved: Florida
            rs_fl = sim_spatial_all[sim_spatial_all["month"] == m]["Florida_kWh"]

            # 6. Spatially resolved: Flesland
            rs_fs = sim_spatial_all[sim_spatial_all["month"] == m]["Flesland_kWh"]

            all_box_data.append([fl_station, fs_station, sm, sa, rs_fl, rs_fs])
            x_labels.append(calendar.month_abbr[m])

        # Flatten into positions:
        # Each month contributes 6 boxplots
        flattened_data = []
        positions = []
        pos_counter = 0

        for i, month_group in enumerate(all_box_data):
            for dataset in month_group:
                flattened_data.append(dataset)
                positions.append(pos_counter)
                pos_counter += 1
            pos_counter += 1  # gap between months

        bp = ax.boxplot(
            flattened_data,
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.6
        )

        # -----------------------------------------------------
        # Color each box by model group
        # -----------------------------------------------------
        for i, box in enumerate(bp['boxes']):
            color = colors[i % 6]   # repeat color sequence every 6 boxplots
            box.set_facecolor(color)
            box.set_alpha(0.5)
            box.set_edgecolor(color)
            box.set_linewidth(1.5)

        for i, whisker in enumerate(bp['whiskers']):
            whisker.set_color(colors[(i // 2) % 6])

        for i, cap in enumerate(bp['caps']):
            cap.set_color(colors[(i // 2) % 6])

        for i, median in enumerate(bp['medians']):
            median.set_color("black")
            median.set_linewidth(1.4)

        # Create x tick labels at month centers
        month_centers = [(i * 7) + 2.5 for i in range(len(months))]
        ax.set_xticks(month_centers)
        ax.set_xticklabels(x_labels, fontsize=14)

        ax.set_title(season_name, fontsize=16)
        ax.set_ylabel("Mean Daily Irradiation (kWh/m²)", fontsize=14)
        ax.grid(True, alpha=0.3)

    # Legend
    legend_labels = [
        "Station Florida",
        "Station Flesland",
        "Model (2)",
        "Model (1)",
        "Model (3): Florida Pixel",
        "Model (3): Flesland Pixel",
    ]
    fig.legend(legend_labels, loc="upper left", bbox_to_anchor=[0.1, 0.95], fontsize=12)

    # Save figure
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Figure saved to {outpath}")


def plot_monthly_boxplots_seasonal_daily_variability(
    stations_data_path,
    sim_model1_path,
    sim_model2_path,
    sim_model3_path,
    outpath="output/seasonal_monthly_boxplots.png"
):

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_path, parse_dates=["datetime"])
    sim_model1 = pd.read_csv(sim_model1_path)
    sim_model2 = pd.read_csv(sim_model2_path)
    sim_model3 = pd.read_csv(sim_model3_path)
    
    # ---------------------------------------------------------
    # Prepare station daily GHI data
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["month"] = stations["datetime"].dt.month
    stations["year"] = stations["datetime"].dt.year
    stations["GHI_daily_mean"] = stations.groupby(["station", "date"])["value"].transform("mean")
    stations["GHI_daily_kWh"] = stations["GHI_daily_mean"] * 24 / 1000

    # ---------------------------------------------------------
    # Prepare model daily data
    # ---------------------------------------------------------
    sim_model2["GHI_daily_kWh"] = sim_model2["GHI_daily_Wh"] / 1000
    sim_model1["GHI_daily_kWh"] = sim_model1["GHI_daily_Wh"] / 1000
    sim_model3["GHI_daily_kWh_Florida"] = sim_model3["Florida_GHI_daily_Wh_mean"] / 1000
    sim_model3["GHI_daily_kWh_Flesland"] = sim_model3["Flesland_GHI_daily_Wh_mean"] / 1000


    # ---------------------------------------------------------
    # Season definitions
    # ---------------------------------------------------------
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
    }

    colors = ["orange", "red", "green", "blue", "purple", "brown"]
    
    # ---------------------------------------------------------
    # Prepare figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    fig.subplots_adjust(hspace=0.35)

    # ---------------------------------------------------------
    # Loop through seasons
    # ---------------------------------------------------------
    for ax, (season_name, months) in zip(axes, seasons.items()):

        # Collect boxplot data:
        #   For each month in this season, create 4 datasets

        all_box_data = []
        x_labels = []

        for m in months:

            # Station Florida
            fl_station = stations[(stations["station"] == "Florida") &
                                       (stations["month"] == m)]["GHI_daily_kWh"]

            # Station Flesland
            fs_station = stations[(stations["station"] == "Flesland") &
                                       (stations["month"] == m)]["GHI_daily_kWh"]

            # Spatially uniform model (annual k)
            sm1 = sim_model1[sim_model1["month"] == m]["GHI_daily_kWh"]
            
            # Spatially uniform model (monthly k)
            sm2 = sim_model2[sim_model2["month"] == m]["GHI_daily_kWh"]
            
            # Spatially heterogeneous model (monthly k)
            sm3flor = sim_model3[sim_model3["month"] == m]["GHI_daily_kWh_Florida"]
            sm3fles = sim_model3[sim_model3["month"] == m]["GHI_daily_kWh_Flesland"]

            all_box_data.append([fl_station, fs_station, sm1, sm2, sm3flor, sm3fles])
            x_labels.append(calendar.month_abbr[m])

        # Flatten into positions:
        # Each month contributes 6 boxplots
        flattened_data = []
        positions = []
        pos_counter = 0

        for i, month_group in enumerate(all_box_data):
            for dataset in month_group:
                flattened_data.append(dataset)
                positions.append(pos_counter)
                pos_counter += 1
            pos_counter += 1  # gap between months

        bp = ax.boxplot(
            flattened_data,
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.6
        )

        # -----------------------------------------------------
        # Color each box by model group
        # -----------------------------------------------------
        for i, box in enumerate(bp['boxes']):
            color = colors[i % 6]   # repeat color sequence every 4 boxplots
            box.set_facecolor(color)
            box.set_alpha(0.5)
            box.set_edgecolor(color)
            box.set_linewidth(1.5)

        for i, whisker in enumerate(bp['whiskers']):
            whisker.set_color(colors[(i // 2) % 6])

        for i, cap in enumerate(bp['caps']):
            cap.set_color(colors[(i // 2) % 6])

        for i, median in enumerate(bp['medians']):
            median.set_color("black")
            median.set_linewidth(1.4)

        # Create x tick labels at month centers
        month_centers = [(i * 7) + 2.5 for i in range(len(months))]
        ax.set_xticks(month_centers)
        ax.set_xticklabels(x_labels, fontsize=14)

        ax.set_title(season_name, fontsize=16)
        ax.set_ylabel("Mean Daily Irradiation (kWh/m²)", fontsize=14)
        ax.grid(True, alpha=0.3)

    # Legend
    legend_labels = [
        "Station Florida",
        "Station Flesland",
        "Model 1",
        "Model 2",
        "Model 3 - Florida", 
        "Model 3 - Flesland"
    ]
    fig.legend(legend_labels, loc="upper left", bbox_to_anchor=[0.1, 0.95], fontsize=12)

    # Save figure
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Figure saved to {outpath}")


def plot_ecdfs_daily_mean(stations_data_filepath, 
                          daily_sim_model1_filepath, 
                          daily_sim_model2_filepath,
                          daily_sim_model3_filepath,
                          outpath="output/ecdfs_daily_means_longterm_sim_vs_obs.png"): 
    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_filepath, parse_dates=["datetime"])
    sim_model1 = pd.read_csv(daily_sim_model1_filepath)
    sim_model2 = pd.read_csv(daily_sim_model2_filepath)
    sim_model3 = pd.read_csv(daily_sim_model3_filepath)

    # ---------------------------------------------------------
    # Prepare station daily GHI data
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["month"] = stations["datetime"].dt.month
    stations["year"] = stations["datetime"].dt.year
    stations["GHI_daily_mean"] = stations.groupby(["station", "date"])["value"].transform("mean")
    stations["GHI_daily_kWh"] = stations["GHI_daily_mean"] * 24 / 1000

    # ---------------------------------------------------------
    # Prepare model daily data
    # ---------------------------------------------------------
    sim_model1["GHI_daily_kWh"] = sim_model1["GHI_daily_Wh"] / 1000    
    sim_model2["GHI_daily_kWh"] = sim_model2["GHI_daily_Wh"] / 1000
    sim_model3["GHI_daily_kWh_Florida"] = sim_model3["Florida_GHI_daily_Wh_mean"] / 1000
    sim_model3["GHI_daily_kWh_Flesland"] = sim_model3["Flesland_GHI_daily_Wh_mean"] / 1000
    
     
    # Calculate cdfs
    flesland_all_obs = stations.loc[stations["station"]=="Flesland", "GHI_daily_kWh"]
    florida_all_obs = stations.loc[stations["station"]=="Florida", "GHI_daily_kWh"]
    model1_all_obs = sim_model1["GHI_daily_kWh"]
    model2_all_obs = sim_model2["GHI_daily_kWh"]
    model3_flesland_all_obs = sim_model3["GHI_daily_kWh_Flesland"]
    model3_florida_all_obs = sim_model3["GHI_daily_kWh_Florida"]
    ecdf_flesland = sm.distributions.ECDF(flesland_all_obs)
    ecdf_florida = sm.distributions.ECDF(florida_all_obs)
    ecdf_model1 = sm.distributions.ECDF(model1_all_obs)
    ecdf_model2 = sm.distributions.ECDF(model2_all_obs)
    ecdf_model3_flesland = sm.distributions.ECDF(model3_flesland_all_obs)
    ecdf_model3_florida = sm.distributions.ECDF(model3_florida_all_obs)
    
    # Plot 
    x = np.linspace(0, 12, 500)

    plt.plot(x, ecdf_flesland(x), label="Flesland", color="tab:red")
    plt.plot(x, ecdf_florida(x), label="Florida", color="tab:orange")
    plt.plot(x, ecdf_model1(x), label="Model 1", color="tab:blue")
    plt.plot(x, ecdf_model2(x), label="Model 2", color="tab:green")
    plt.plot(x, ecdf_model3_flesland(x), label="Model 3 - Flesland", color="tab:brown")
    plt.plot(x, ecdf_model3_florida(x), label="Model 3 - Florida", color="tab:purple")

    plt.xlabel("Daily Solar Irradiation [kWh/m²]")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
    plt.title("CDF Comparison - Daily Means")
    plt.tight_layout()
    
    # Save and print save message
    plt.savefig(outpath, dpi=300)
    print(f"Saved ecdf plot to {outpath}.")
    plt.close()
    

def plot_ecdfs_monthly_mean(stations_data_filepath, 
                          daily_sim_model1_filepath, 
                          daily_sim_model2_filepath,
                          monthly_sim_model_filepath,
                          outpath="output/ecdfs_monthly_means_longterm_sim_vs_obs.png"): 
    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_filepath, parse_dates=["datetime"])
    sim_model1 = pd.read_csv(daily_sim_model1_filepath)
    sim_model2 = pd.read_csv(daily_sim_model2_filepath)
    sim_model3 = pd.read_csv(monthly_sim_model_filepath)

    # ---------------------------------------------------------
    # Prepare station monthly mean GHI data
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["month"] = stations["datetime"].dt.month
    stations["year"] = stations["datetime"].dt.year
    stations["GHI_daily_mean"] = stations.groupby(["station", "date"])["value"].transform("mean")
    stations["GHI_daily_kWh"] = stations["GHI_daily_mean"] * 24 / 1000    
    station_monthly_means = stations.groupby(["station","year","month"])["GHI_daily_kWh"].mean().reset_index()
    flesland_monthly_means = station_monthly_means.loc[station_monthly_means["station"] == "Flesland", "GHI_daily_kWh"]
    florida_monthly_means = station_monthly_means.loc[station_monthly_means["station"] == "Florida", "GHI_daily_kWh"]

    # ---------------------------------------------------------
    # Prepare spatially uniform model monthly data
    # ---------------------------------------------------------
    sim_model2["GHI_daily_kWh"] = sim_model2["GHI_daily_Wh"] / 1000
    sim_model1["GHI_daily_kWh"] = sim_model1["GHI_daily_Wh"] / 1000   
    sim_model1_monthly_means = sim_model1.groupby(["year", "month"])["GHI_daily_kWh"].mean().reset_index()
    sim_model2_monthly_means = sim_model2.groupby(["year", "month"])["GHI_daily_kWh"].mean().reset_index()
 
    # ---------------------------------------------------------
    # Spatially resolved model: extract Florida + Flesland daily kWh
    # ---------------------------------------------------------
    sim_model3["Florida_kWh"] = sim_model3["Florida_GHI_daily_Wh_mean"] / 1000
    sim_model3["Flesland_kWh"] = sim_model3["Flesland_GHI_daily_Wh_mean"] / 1000

    # Only keep sky_type == "all-sky"
    sim_model3_monthly_means = sim_model3[sim_model3["sky_type"] == "all-sky"]
    sim_model3_flesland = sim_model3_monthly_means["Flesland_kWh"]
    sim_model3_florida = sim_model3_monthly_means["Florida_kWh"]
     
    # Calculate cdfs
    ecdf_flesland = sm.distributions.ECDF(flesland_monthly_means)
    ecdf_florida = sm.distributions.ECDF(florida_monthly_means)
    ecdf_model1 = sm.distributions.ECDF(sim_model1_monthly_means["GHI_daily_kWh"])
    ecdf_model2 = sm.distributions.ECDF(sim_model2_monthly_means["GHI_daily_kWh"])
    ecdf_model3_flesland = sm.distributions.ECDF(sim_model3_flesland)
    ecdf_model3_florida = sm.distributions.ECDF(sim_model3_florida)
    
    # Plot 
    x = np.linspace(0, 9, 400)

    plt.plot(x, ecdf_flesland(x), label="Flesland", color="tab:red")
    plt.plot(x, ecdf_florida(x), label="Florida", color="tab:orange")
    plt.plot(x, ecdf_model1(x), label="Model 1", color="tab:blue")
    plt.plot(x, ecdf_model2(x), label="Model 2", color="tab:green")
    plt.plot(x, ecdf_model3_flesland(x), label="Model 3: Flesland", color="tab:purple")
    plt.plot(x, ecdf_model3_florida(x), label="Model 3: Florida", color="tab:brown")

    plt.xlabel("Daily Mean Solar Irradiation [kWh/m²]")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
    plt.title("CDF Comparison - Monthly Means")
    plt.tight_layout()
    
    # Save and print save message
    plt.savefig(outpath, dpi=150)
    print(f"Saved ecdf plot to {outpath}.")
    plt.close()
  


def print_emd_daily_mean(
    stations_data_filepath, 
    daily_sim_model1_filepath, 
    daily_sim_model2_filepath,
    daily_sim_model3_filepath,
    outpath="output/emd_monthly_distributions.png"
):
    """
    Prints Earth Mover's Distance (Wasserstein) for daily mean GHI:
      1) Overall daily means
      2) Monthly daily means
      3) Plot of monthly EMD values for:
         - M1 vs Flesland
         - M1 vs Florida
         - M2 vs Flesland
         - M2 vs Florida
    """

    print("\n======================")
    print(" EMD for DAILY MEANS")
    print("======================\n")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_filepath, parse_dates=["datetime"])
    sim_model1 = pd.read_csv(daily_sim_model1_filepath)
    sim_model2 = pd.read_csv(daily_sim_model2_filepath)
    sim_model3 = pd.read_csv(daily_sim_model3_filepath)

    # Prepare stations daily means
    stations["date"] = stations["datetime"].dt.date
    stations["month"] = stations["datetime"].dt.month
    stations["GHI_daily_mean"] = stations.groupby(["station","date"])["value"].transform("mean")
    stations["GHI_daily_kWh"] = stations["GHI_daily_mean"] * 24 / 1000

    # Prepare model daily means
    sim_model1["GHI_daily_kWh"] = sim_model1["GHI_daily_Wh"] / 1000
    sim_model2["GHI_daily_kWh"] = sim_model2["GHI_daily_Wh"] / 1000
    sim_model3["GHI_daily_kWh_Florida"] = sim_model3["Florida_GHI_daily_Wh_mean"] / 1000
    sim_model3["GHI_daily_kWh_Flesland"] = sim_model3["Flesland_GHI_daily_Wh_mean"] / 1000
 
    stations_daily = stations.drop_duplicates(subset=["station","date"])

    # Split by station
    flesland = stations_daily.loc[stations_daily["station"]=="Flesland"]
    florida  = stations_daily.loc[stations_daily["station"]=="Florida"]

    # Convenience shortcuts
    m1 = sim_model1["GHI_daily_kWh"].values
    m2 = sim_model2["GHI_daily_kWh"].values
    m3_flesland = sim_model3["GHI_daily_kWh_Flesland"].values
    m3_florida = sim_model3["GHI_daily_kWh_Florida"].values

    # ---------------------------------------------------------
    # 1) OVERALL daily mean EMD
    # ---------------------------------------------------------
    print("---- Overall EMD (Daily Means) ----")
    for station_name, df_station in [("Flesland", flesland), ("Florida", florida)]:
        s = df_station["GHI_daily_kWh"].values

        emd_m1 = wasserstein_distance(s, m1)
        emd_m2 = wasserstein_distance(s, m2)
        if station_name == "Flesland":
            emd_m3 = wasserstein_distance(s,m3_flesland)
        elif station_name == "Florida":
            emd_m3 = wasserstein_distance(s,m3_florida)

        print(f"{station_name}:")
        print(f"   Model 1 vs Station: {emd_m1:.4f} kWh/m²")
        print(f"   Model 2 vs Station: {emd_m2:.4f} kWh/m²")
        print(f"   Model 3 vs Station: {emd_m3:.4f} kWh/m²")

    # ---------------------------------------------------------
    # 2) MONTHLY daily means EMD
    # ---------------------------------------------------------
    print("\n---- Monthly EMD (Daily Means) ----")

    # Store values for plotting
    months = list(range(1, 13))
    emd_m1_flesland = []
    emd_m1_florida  = []
    emd_m2_flesland = []
    emd_m2_florida  = []
    emd_m3_flesland = []
    emd_m3_florida = []

    for month in months:
        print(f"\nMonth {month:02d}:")
        m1_m = sim_model1.loc[sim_model1["month"] == month, "GHI_daily_kWh"].values
        m2_m = sim_model2.loc[sim_model2["month"] == month, "GHI_daily_kWh"].values
        m3_m_flesland = sim_model3.loc[sim_model3["month"] == month, "GHI_daily_kWh_Flesland"].values
        m3_m_florida = sim_model3.loc[sim_model3["month"] == month, "GHI_daily_kWh_Florida"].values

        # Flesland
        s_fles = flesland.loc[flesland["month"] == month, "GHI_daily_kWh"].values
        if len(s_fles) > 0:
            emd1 = wasserstein_distance(s_fles, m1_m)
            emd2 = wasserstein_distance(s_fles, m2_m)
            emd3 = wasserstein_distance(s_fles, m3_m_flesland)
            print(f"  Flesland: Model1={emd1:.4f} kWh/m², Model2={emd2:.4f} kWh/m², Model3={emd3:.4f} kWh/m²")
        else:
            emd1 = emd2 = emd3 = None
            print("  Flesland: No data")

        emd_m1_flesland.append(emd1)
        emd_m2_flesland.append(emd2)
        emd_m3_flesland.append(emd3)

        # Florida
        s_flor = florida.loc[florida["month"] == month, "GHI_daily_kWh"].values
        if len(s_flor) > 0:
            emd1 = wasserstein_distance(s_flor, m1_m)
            emd2 = wasserstein_distance(s_flor, m2_m)
            emd3 = wasserstein_distance(s_flor, m3_m_florida)
            print(f"  Florida: Model1={emd1:.4f} kWh/m², Model2={emd2:.4f} kWh/m², Model3={emd3:.4f} kWh/m²")
        else:
            emd1 = emd2 = emd3 = None
            print("  Florida: No data")

        emd_m1_florida.append(emd1)
        emd_m2_florida.append(emd2)
        emd_m3_florida.append(emd3)


    # ---------------------------------------------------------
    # 3) PLOT MONTHLY EMD VALUES
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Convert None to NaN for plotting gaps
    emd_m1_flesland = np.array([np.nan if v is None else v for v in emd_m1_flesland])
    emd_m1_florida  = np.array([np.nan if v is None else v for v in emd_m1_florida])
    emd_m2_flesland = np.array([np.nan if v is None else v for v in emd_m2_flesland])
    emd_m2_florida  = np.array([np.nan if v is None else v for v in emd_m2_florida])
    emd_m3_flesland = np.array([np.nan if v is None else v for v in emd_m3_flesland])
    emd_m3_florida  = np.array([np.nan if v is None else v for v in emd_m3_florida])

    # Model 1 (blue)
    plt.plot(months, emd_m1_flesland, color="blue", marker="o", markersize=10, linestyle="-", label="Model 1 vs Flesland")
    plt.plot(months, emd_m1_florida,  color="blue", marker="x", markersize=12, linestyle=":", label="Model 1 vs Florida")

    # Model 2 (green)
    plt.plot(months, emd_m2_flesland, color="green", marker="o", markersize=10, linestyle="-", label="Model 2 vs Flesland")
    plt.plot(months, emd_m2_florida,  color="green", marker="x", markersize=12, linestyle=":", label="Model 2 vs Florida")

    # Model 3 (purple and brown)
    plt.plot(months, emd_m3_flesland, color="brown", marker="o", markersize=10, linestyle="-", label="Model 3 vs Flesland")
    plt.plot(months, emd_m3_florida,  color="purple", marker="x", markersize=12, linestyle=":", label="Model 3 vs Florida")
       
    plt.xlabel("Month")
    plt.ylabel("EMD (kWh/m²)")
    plt.title("EMD for Monthly Distribution of Daily Mean Irradiation")
    plt.xticks(months)
    plt.grid(alpha=0.4)
    plt.legend(bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved figure to {outpath}.")


def print_emd_monthly_mean(
    stations_data_filepath, 
    daily_sim_model1_filepath, 
    daily_sim_model2_filepath,
    monthly_sim_model_filepath,
    outpath="output/emd_monthly_mean_vs_month.png"
):
    """
    Extends the monthly mean EMD analysis to:
      - Print overall monthly mean EMD (existing)
      - Print per-month EMD values (new)
      - Plot EMD vs. month for Model1, Model2, Model3 vs. each station (new)
    """

    print("\n=========================")
    print(" EMD for MONTHLY MEANS")
    print("=========================\n")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    stations = pd.read_csv(stations_data_filepath, parse_dates=["datetime"])
    sim_model1 = pd.read_csv(daily_sim_model1_filepath)
    sim_model2 = pd.read_csv(daily_sim_model2_filepath)
    sim_model3 = pd.read_csv(monthly_sim_model_filepath)

    # ---------------------------------------------------------
    # Prepare station monthly means
    # ---------------------------------------------------------
    stations["date"] = stations["datetime"].dt.date
    stations["month"] = stations["datetime"].dt.month
    stations["year"] = stations["datetime"].dt.year

    stations["GHI_daily_mean"] = stations.groupby(["station","date"])["value"].transform("mean")
    stations["GHI_daily_kWh"] = stations["GHI_daily_mean"] * 24 / 1000

    station_monthly = stations.groupby(
        ["station","year","month"]
    )["GHI_daily_kWh"].mean().reset_index()

    flesland_monthly = station_monthly.loc[
        station_monthly["station"]=="Flesland", "GHI_daily_kWh"
    ].values

    florida_monthly = station_monthly.loc[
        station_monthly["station"]=="Florida", "GHI_daily_kWh"
    ].values

    # ---------------------------------------------------------
    # Prepare model monthly means
    # ---------------------------------------------------------
    sim_model1["GHI_daily_kWh"] = sim_model1["GHI_daily_Wh"] / 1000
    sim_model2["GHI_daily_kWh"] = sim_model2["GHI_daily_Wh"] / 1000

    model1_monthly = sim_model1.groupby(["year","month"])["GHI_daily_kWh"].mean().values
    model2_monthly = sim_model2.groupby(["year","month"])["GHI_daily_kWh"].mean().values

    # Model 3 (already monthly)
    sim_model3["Flesland_kWh"] = sim_model3["Flesland_GHI_daily_Wh_mean"] / 1000
    sim_model3["Florida_kWh"]  = sim_model3["Florida_GHI_daily_Wh_mean"] / 1000
    sim_model3 = sim_model3[sim_model3["sky_type"] == "all-sky"]

    model3_flesland = sim_model3["Flesland_kWh"].values
    model3_florida  = sim_model3["Florida_kWh"].values

    # ---------------------------------------------------------
    # Compute Overall EMDs
    # ---------------------------------------------------------
    print("---- Overall EMD (Monthly Means) ----")

    for station_name, s_vals in [("Flesland", flesland_monthly),
                                 ("Florida", florida_monthly)]:

        emd_m1 = wasserstein_distance(s_vals, model1_monthly)
        emd_m2 = wasserstein_distance(s_vals, model2_monthly)
        emd_m3 = wasserstein_distance(
            s_vals,
            model3_flesland if station_name=="Flesland" else model3_florida
        )

        print(f"{station_name}:")
        print(f"   Model 1 vs Station: {emd_m1:.4f} kWh/m²")
        print(f"   Model 2 vs Station: {emd_m2:.4f} kWh/m²")
        print(f"   Model 3 vs Station: {emd_m3:.4f} kWh/m²")

    # ---------------------------------------------------------
    # Monthly EMD Values
    # ---------------------------------------------------------
    print("\n---- Monthly EMD (Monthly Means) ----")

    months = range(1, 13)

    # Arrays for plotting
    m1_fles = [];  m1_flor = []
    m2_fles = [];  m2_flor = []
    m3_fles = [];  m3_flor = []

    for month in months:
        print(f"\nMonth {month:02d}:")

        # observed station values for this month
        s_fles = station_monthly.loc[
            (station_monthly["station"]=="Flesland") &
            (station_monthly["month"]==month),
            "GHI_daily_kWh"
        ].values

        s_flor = station_monthly.loc[
            (station_monthly["station"]=="Florida") &
            (station_monthly["month"]==month),
            "GHI_daily_kWh"
        ].values

        # model values for this month
        m1_m = sim_model1.loc[sim_model1["month"]==month, "GHI_daily_kWh"].values
        m2_m = sim_model2.loc[sim_model2["month"]==month, "GHI_daily_kWh"].values

        m3_fles_m = sim_model3.loc[sim_model3["month"]==month, "Flesland_kWh"].values
        m3_flor_m = sim_model3.loc[sim_model3["month"]==month, "Florida_kWh"].values

        # --- Flesland ---
        if len(s_fles) > 0:
            e1 = wasserstein_distance(s_fles, m1_m)
            e2 = wasserstein_distance(s_fles, m2_m)
            e3 = wasserstein_distance(s_fles, m3_fles_m)
            print(f"  Flesland: M1={e1:.4f}, M2={e2:.4f}, M3={e3:.4f}")
        else:
            e1 = e2 = e3 = None
            print("  Flesland: No data")

        m1_fles.append(e1)
        m2_fles.append(e2)
        m3_fles.append(e3)

        # --- Florida ---
        if len(s_flor) > 0:
            e1 = wasserstein_distance(s_flor, m1_m)
            e2 = wasserstein_distance(s_flor, m2_m)
            e3 = wasserstein_distance(s_flor, m3_flor_m)
            print(f"  Florida: M1={e1:.4f}, M2={e2:.4f}, M3={e3:.4f}")
        else:
            e1 = e2 = e3 = None
            print("  Florida: No data")

        m1_flor.append(e1)
        m2_flor.append(e2)
        m3_flor.append(e3)

    # ---------------------------------------------------------
    # Plot EMD vs Month
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # convert None → NaN
    m1_fles = np.array([np.nan if v is None else v for v in m1_fles])
    m1_flor = np.array([np.nan if v is None else v for v in m1_flor])
    m2_fles = np.array([np.nan if v is None else v for v in m2_fles])
    m2_flor = np.array([np.nan if v is None else v for v in m2_flor])
    m3_fles = np.array([np.nan if v is None else v for v in m3_fles])
    m3_flor = np.array([np.nan if v is None else v for v in m3_flor])

    # Model 1 (blue)
    plt.plot(months, m1_fles, color="blue",  marker="o", markersize=10, linestyle="-", label="Model 1 vs Flesland")
    plt.plot(months, m1_flor, color="blue",  marker="d", markersize=12, linestyle=":", label="Model 1 vs Florida")

    # Model 2 (green)
    plt.plot(months, m2_fles, color="green", marker="o", markersize=10, linestyle="-", label="Model 2 vs Flesland")
    plt.plot(months, m2_flor, color="green", marker="d", markersize=12, linestyle=":", label="Model 2 vs Florida")

    # Model 3 (purple for Flesland, brown for Florida)
    plt.plot(months, m3_fles, color="purple", marker="o", markersize=10, linestyle="-", label="Model 3 vs Flesland")
    plt.plot(months, m3_flor, color="brown",  marker="d", markersize=12, linestyle=":", label="Model 3 vs Florida")

    plt.xlabel("Month")
    plt.ylabel("EMD (kWh/m²)")
    plt.title("EMD for Monthly Distribution of Monthly Mean of Daily Mean Irradiation")
    plt.xticks(list(months))
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)

    print(f"\nSaved monthly EMD plot to: {outpath}")
    

def sky_type_contribution_to_monthly_irradiation(results_model2_filepath, 
                                                 outpath="output/sky_type_contribution_monthly_irradiation_model2.png"):
    # ----------------------------------------------------
    # Load data
    # ----------------------------------------------------
    df = pd.read_csv(results_model2_filepath)
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)

    # Add daily GHI in kWh/m²
    df["GHI_kWh"] = df["GHI_daily_Wh"] / 1000

    # ----------------------------------------------------
    # Compute total irradiation per (year, month)
    # ----------------------------------------------------
    monthly_total = (
        df.groupby(["year", "month"])["GHI_kWh"]
        .sum()
        .rename("monthly_total")
        .reset_index()
    )

    df = df.merge(monthly_total, on=["year", "month"])

    # ----------------------------------------------------
    # Compute irradiation per sky type per (year, month)
    # ----------------------------------------------------
    monthly_sky = (
        df.groupby(["year", "month", "sky_type"])["GHI_kWh"]
        .sum()
        .rename("sky_irr_kWh")
        .reset_index()
    )

    # Merge to compute percent contribution
    monthly_sky = monthly_sky.merge(monthly_total, on=["year", "month"])
    monthly_sky["pct_contribution"] = 100 * monthly_sky["sky_irr_kWh"] / monthly_sky["monthly_total"]

    # ----------------------------------------------------
    # Aggregate across years to get mean ± sd for each month/sky_type
    # ----------------------------------------------------
    plot_df = (
        monthly_sky.groupby(["month", "sky_type"])["pct_contribution"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Ensure consistent sky type order
    order = ["clear", "mixed", "overcast"]
    plot_df["sky_type"] = pd.Categorical(plot_df["sky_type"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["month", "sky_type"])

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    months = np.arange(1, 13)
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]


    for i, sky in enumerate(order):
        subset = plot_df[plot_df["sky_type"] == sky]
        ax.bar(
            months + offsets[i],
            subset["mean"].values,
            width=bar_width,
            yerr=subset["std"].values,
            capsize=5,
            label=sky,
            color=SKY_TYPE_COLORS[sky],
            alpha=0.9,
        )

        # Add text for % days
        for m, mean_pct in zip(subset["month"], subset["mean"]):
            ax.text(
                m + offsets[i],
                subset.loc[subset["month"] == m, "mean"].values[0] + 
                subset.loc[subset["month"] == m, "std"].values[0] + 1.5,
                f"{mean_pct:.0f}%",
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(months)
    ax.set_xticklabels(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Percent contribution to monthly irradiation (%)")
    ax.set_title("Sky Type Contribution to Monthly Irradiation (Model 2)")
    ax.legend(title="Sky Type", bbox_to_anchor=(1.2, 0.5))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------
    plt.savefig(outpath, dpi=150)
    print(f"Figure saved to: {outpath}")



def calculate_energy_for_locations(monthly_irradiation_spatial_filepath, A=25, eta=0.25, pr=0.8):
    
    # -------------------------
    # Define locations
    # -------------------------
    locations = [
        ("Askøy",      60.450667, 5.130472),
        ("City Center",60.3864722,5.3283056),
        ("Airport",   60.274000, 5.229083),
        ("Litlesotra", 60.366861, 5.156306),
        ("Paradis",    60.341778, 5.346944),
        ("Haukeland",  60.3600278,5.4573333),
    ]

    # Containers for results
    monthly_energy = {name: [] for name,_,_ in locations}
    annual_energy = {}

    # -------------------------
    # Extract irradiation for each location
    # -------------------------
    for name, lat, lon in locations:

        months, irr_daily = extract_pixel_by_location(
            monthly_irradiation_spatial_filepath,
            lat, lon,
            var_name="all_sky_ghi",
            time_name="month"
        )

        print(f"\n====== {name} ======")
        print("Month | Mean daily irradiation (kWh/m²)")

        # Compute monthly energy
        for m, I_daily in enumerate(irr_daily, start=1):

            ndays = calendar.monthrange(2020, m)[1]  # year irrelevant, only month length used
            I_month = I_daily * ndays               # convert daily mean → monthly irradiation

            # PV energy for this month
            E_month = I_month * A * eta * pr

            monthly_energy[name].append(E_month)

            print(f"{m:2d}    | {I_daily:.3f}")

        # Annual sum
        annual_energy[name] = np.sum(monthly_energy[name])


    # -------------------------
    # Print the monthly and annual PV energy
    # -------------------------
    print("\n\n================ Energy Results ================\n")
    for name in monthly_energy.keys():
        print(f"\n--- {name} ---")
        for m, E_m in enumerate(monthly_energy[name], start=1):
            print(f"Month {m:2d}:  {E_m:.1f} kWh")
        print(f"Annual E: {annual_energy[name]:.1f} kWh")

    # -------------------------
    # Print annual difference relative to City Center
    # -------------------------
    city_center_E = annual_energy["City Center"]
    print("\n================ Annual PV Energy Difference Relative to City Center (%) ================")
    for name, E in annual_energy.items():
        diff_percent = (E - city_center_E) / city_center_E * 100
        print(f"{name:12s}: {diff_percent:+.2f}%")

    # -------------------------
    # Plotting
    # -------------------------
    fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(16,6), gridspec_kw={'width_ratios':[3,1]} ) 
    months = np.arange(1, 13) 
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] 
    line_colors = ["#FF7733", "#BC96E6", "#D81E5B", "#A6DD72", "#255F85", "#22DB7B"]
    
    # --- Left subplot: monthly barplot of differences from City Center ---

    city_ref = np.array(monthly_energy["City Center"])

    # Prepare bar width and offsets
    n_locations = len(locations)
    bar_width = 0.12
    offsets = np.linspace(-0.35, 0.35, n_locations)

    for idx, (name,_,_) in enumerate(locations):
        vals = np.array(monthly_energy[name])
        diff_percent = (vals - city_ref) / city_ref * 100
        
        ax1.bar(
            months + offsets[idx],
            diff_percent,
            width=bar_width,
            label=name,
            color=line_colors[idx],
            alpha=0.9
        )

    # Zero reference line (City Center)
    ax1.axhline(0, color="black", linewidth=1.4)

    # X-axis labels
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_labels)

    # Gridlines at month boundaries
    for m in months:
        ax1.axvline(m, color="gray", alpha=0.15, linewidth=1)

    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Relative Difference from City Center (%)")
    ax1.set_title("Monthly PV Energy Difference Relative to City Center")
    ax1.legend()


    # --- Right subplot: annual barplot with matching colors ---
    names = [name for name,_,_ in locations]
    annual_vals = [annual_energy[name] for name in names]

    bars = ax2.bar(names, annual_vals, color=line_colors)
    ax2.set_ylabel("Annual PV Energy (kWh)")
    ax2.set_title("Annual PV Energy")
    ax2.set_xticklabels(names, rotation=45, ha='right')
    
    # Add bar height labels on top
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,  # x position: center of bar
            height + 50,                      # y position: slightly above the bar
            f"{height:.0f}",                  # label text
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    outpath = "output/energy_monthly_annual_6locations.png"
    plt.savefig(outpath, dpi = 150)
    print(f"Saved figure to {outpath}.")
    return monthly_energy, annual_energy



if __name__ == "__main__": 
    stations_data_raw_path = "data/raw/frost_ghi_1H_Flesland_Florida_2015-2025.csv"
    daily_simulation_model2_path = "data/processed/longterm_sim_ghi_5000_k=monthly.csv"
    daily_simulation_model1_path = "data/processed/longterm_sim_ghi_5000_k=annual.csv"
    daily_sim_model_3_path = "data/processed/longterm_sim_Florida_Flesland_daily_pixels_5000_k=monthly.csv"
    stations_data_cleaned_path = "data/processed/frost_ghi_1H_Flesland_Florida_2015-2025_cleaned.csv"
    monthly_simulation_spatial_path = "data/processed/longterm_sim_Florida_Flesland_monthly_pixels_5000_k=monthly.csv"
    monthly_simulation_nc_path = "data/processed/longterm_ghi_spatially_resolved_monthly.nc"
    
    # Clean dataset
    #stations_data_df = flag_stations_data(stations_data_raw_path)
    #stations_data_df = stations_data_df.loc[stations_data_df["value_flag"] == "OK"]
    #stations_data_df = stations_data_df.drop(columns=["value_flag"])
    #print(stations_data_df.head())
    # Save 
    #stations_data_df.to_csv(stations_data_cleaned_path, index=False)
    
    # Plot daily ghi 
    #plot_doy_vs_daily_mean_ghi(stations_data_cleaned_path, daily_simulation_monthly_path, daily_simulation_annual_path)
    """plot_doy_vs_daily_mean_ghi_10day(stations_data_cleaned_path,
                                     daily_sim_model1_path=daily_simulation_model1_path, 
                                     daily_sim_model2_path=daily_simulation_model2_path, 
                                     daily_sim_model3_path=daily_sim_model_3_path)
    plot_monthly_boxplots_seasonal_daily_variability(stations_data_path=stations_data_cleaned_path,
                                   sim_model1_path=daily_simulation_model1_path,
                                   sim_model2_path=daily_simulation_model2_path,
                                   sim_model3_path=daily_sim_model_3_path,
                                   outpath="output/stations_vs_longterm_sim_monthly_boxplots_daily_variability.png")
    plot_monthly_boxplots_seasonal_monthly_variability(stations_data_path=stations_data_cleaned_path,
                                   sim_monthly_path=daily_simulation_monthly_path,
                                   sim_annual_path=daily_simulation_annual_path,
                                   sim_spatial_path=monthly_simulation_spatial_path,
                                   outpath="output/stations_vs_longterm_sim_monthly_boxplots_monthly_variability.png")
    plot_ecdfs_daily_mean(stations_data_filepath=stations_data_cleaned_path,
                          daily_sim_model1_filepath=daily_simulation_model1_path,
                          daily_sim_model2_filepath=daily_simulation_model2_path,
                          daily_sim_model3_filepath=daily_sim_model_3_path)
    plot_ecdfs_monthly_mean(stations_data_filepath=stations_data_cleaned_path, 
                            daily_sim_model1_filepath=daily_simulation_annual_path, 
                            daily_sim_model2_filepath=daily_simulation_monthly_path, 
                            monthly_sim_model_filepath=monthly_simulation_spatial_path)
    print_emd_daily_mean(stations_data_filepath=stations_data_cleaned_path,
                         daily_sim_model1_filepath=daily_simulation_model1_path,
                         daily_sim_model2_filepath=daily_simulation_model2_path,
                         daily_sim_model3_filepath = daily_sim_model_3_path)
    print_emd_monthly_mean(stations_data_filepath=stations_data_cleaned_path, 
                           daily_sim_model1_filepath=daily_simulation_annual_path,
                           daily_sim_model2_filepath=daily_simulation_monthly_path,
                           monthly_sim_model_filepath=monthly_simulation_spatial_path)"""
    #sky_type_contribution_to_monthly_irradiation(results_model2_filepath=daily_simulation_monthly_path)
    calculate_energy_for_locations(monthly_irradiation_spatial_filepath=monthly_simulation_nc_path)