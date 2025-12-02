import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from src.preprocessing.clean_stations_data import flag_observations
from src.plotting import STATION_COLORS, set_paper_style

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
                p10=lambda x: np.percentile(x, 10),
                p25=lambda x: np.percentile(x, 25),
                p50=lambda x: np.percentile(x, 50),  
                p75=lambda x: np.percentile(x, 75),
                p90=lambda x: np.percentile(x, 90),
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
        plt.fill_between(df["doy"], df["p10"], df["p90"], color=color, alpha=0.12)

    # Stations
    add_line_and_shade(stat_flesland, "Flesland", "tab:red")
    add_line_and_shade(stat_florida, "Florida", "tab:orange")

    # Simulations
    add_line_and_shade(sim_monthly_stats, "Model Monthly Mean k", "tab:green")
    add_line_and_shade(sim_annual_stats, "Model Annual Mean k", "tab:blue")

    plt.xlabel("Day of Year")
    plt.ylabel("Daily Mean Solar Energy (kWh/m²)")
    plt.title("Daily Mean Solar Energy vs Day of Year — Stations vs. Simulations")
    
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
    outpath = "output/daily_mean_ghi_vs_doy_longterm_sim_vs_stations.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Figure saved to {outpath}.")


if __name__ == "__main__": 
    stations_data_raw_path = "data/raw/frost_ghi_1H_Flesland_Florida_2015-2025.csv"
    daily_simulation_monthly_path = "data/processed/longterm_sim_ghi_5000_k=monthly.csv"
    daily_simulation_annual_path = "data/processed/longterm_sim_ghi_5000_k=annual.csv"
    stations_data_cleaned_path = "data/processed/frost_ghi_1H_Flesland_Florida_2015-2025_cleaned.csv"
    
    # Clean dataset
    #stations_data_df = flag_stations_data(stations_data_raw_path)
    #stations_data_df = stations_data_df.loc[stations_data_df["value_flag"] == "OK"]
    #stations_data_df = stations_data_df.drop(columns=["value_flag"])
    #print(stations_data_df.head())
    # Save 
    #stations_data_df.to_csv(stations_data_cleaned_path, index=False)
    
    # Plot daily ghi 
    plot_doy_vs_daily_mean_ghi(stations_data_cleaned_path, daily_simulation_monthly_path, daily_simulation_annual_path)