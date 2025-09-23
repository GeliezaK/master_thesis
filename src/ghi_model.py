import os 
import pvlib 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import datetime
import seaborn as sns
import random


BERGEN_LAT = 60.39
BERGEN_LON = 5.33
BERGEN_ALT = 52
CAF = {"clear": 1, "mixed": 0.7, "overcast": 0.3}

mixed_sky_threshold = 1
overcast_sky_threshold = 99
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

skytype_colors = ["skyblue", "sandybrown", "gray"]

def clean_dataset_seklima_ghi(path, outpath): 
    """Clean dataset seklima ghi ground measurements 10 min aggregates"""
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=["Station"])
    df.columns = ["name", "timestamp", "mean_ghi_10M"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M") # not timezone-aware, need to convert to oslo timezone

    # Keep only rows between 11:00 and 16:00 (inclusive)
    df = df[(df["timestamp"].dt.hour >= 11) & (df["timestamp"].dt.hour <= 16)]

    df.to_csv(outpath, sep=",", index=False)
    return df
    
def calculate_overcast_caf(s2_overcast_times, ghi_df): 
    """Pair s2 acquisition times with observations from seklima ghi_df. 
    Then, get the clear sky GHI and compute the CAF as CAF = GHI/GHI_clear"""
    
    ghi_df = ghi_df.copy()
    
    # timezone-aware, converted to utc for computations, only convert to Oslo for plotting
    ghi_df['timestamp'] = pd.to_datetime(ghi_df['timestamp'], utc=True) 
 
    # Convert s2_overcast_times (list of strings or datetime) to pandas datetime with timezone awareness
    s2_times = pd.to_datetime(s2_overcast_times, utc=True)

    # For each Sentinel timestamp, find nearest ghi_df timestamp using pd.merge_asof
    
    # Sort both series by timestamp to use merge_asof
    ghi_df = ghi_df.sort_values('timestamp').reset_index(drop=True)
    s2_times = pd.Series(s2_times).sort_values().reset_index(drop=True)
    
    # Create DataFrame from s2_times for merge
    s2_df = pd.DataFrame({'s2_time': s2_times})

    # Use merge_asof to find closest ghi timestamp before or equal to Sentinel time
    matched = pd.merge_asof(s2_df, ghi_df, left_on='s2_time', right_on='timestamp', direction='nearest', tolerance=pd.Timedelta('15min'))
    # tolerance can be adjusted (e.g. 15min max difference allowed)
    
    # Drop rows where no close match was found
    matched = matched.dropna(subset=['timestamp'])
    
    print("matched describe: \n", matched.describe())
    print("matched head: \n", matched.head())
    print("number of observations: ", len(matched.index))
      
    # Flesland location
    location = pvlib.location.Location(latitude=60.2935, longitude=5.2200, tz='Europe/Oslo', altitude=50)
    
    # Clear sky model requires times as a DatetimeIndex
    times = matched['timestamp']
    times = pd.DatetimeIndex(matched['timestamp'])
    
    # Print time differences 
    matched['time_diff'] = (matched['timestamp'] - matched['s2_time']).abs()
    print(matched["time_diff"].describe()) # on average 23 seconds difference between observations
    print("\nDistribution of daytimes in matched['s2_time']:")
    print(matched["s2_time"].dt.time.value_counts().sort_index())
     
    # Calculate clearsky GHI using Ineichen model
    clearsky = location.get_clearsky(times, model='ineichen')
    matched['ghi_clear'] = clearsky['ghi'].values
    
    # Calculate CAF = measured GHI / clear sky GHI
    matched['CAF'] = matched['mean_ghi_1M'] / matched['ghi_clear']
    
    matched.to_csv("data/caf_scores.csv")
    return matched
    

def visualize_overcast_caf_params(caf_df, save = False):
    """Plot and print overcast CAF parameter"""
    print("CAF overview: \n", caf_df["CAF"].describe())
    
    # Analyze where CAF out of range
    caf_1 = caf_df[caf_df["CAF"] > 1]
    print(f"Number of outliers: {len(caf_1.index)}")
    print("CAF > 1: \n", caf_1[["s2_time", "time_diff", "mean_ghi_1M", "ghi_clear", "CAF"]])
  
    caf_df = caf_df[caf_df["CAF"] <= 1]
    # Keep all timestamps in utc and only localize to Oslo for plotting
    caf_df['timestamp'] = pd.to_datetime(caf_df["timestamp"], utc=True) # timestamps are maintained correctly even if this is converted twice
    
    caf_df['month'] = caf_df['timestamp'].dt.month
    mean_caf_by_month = caf_df.groupby('month')['CAF'].mean()
    std_caf_by_month = caf_df.groupby('month')['CAF'].std()
    count_caf_by_month = caf_df.groupby('month')['CAF'].count()
    
    mean_of_std = std_caf_by_month.mean()
    std_of_mean = mean_caf_by_month.std()
    std_of_std = std_caf_by_month.std()
    mean_of_mean = mean_caf_by_month.mean()

    print(f"Mean of monthly CAF standard deviations: {mean_of_std:.3f}")
    print(f"Standard deviation of monthly CAF means: {std_of_mean:.3f}")
    print(f"Standard deviation of monthly CAF std: {std_of_std:.3f}")
    print(f"Mean of monthly CAF mean: {mean_of_mean:.3f}")
    
    print(f"{'Month':<10} {'Mean CAF':<12} {'Std':<10} {'Count obs':<10}")
    print("-" * 44)

    for month in range(1, 13):
        mean = mean_caf_by_month.get(month, np.nan)
        std = std_caf_by_month.get(month, np.nan)
        count = count_caf_by_month.get(month, 0)
        
        print(f"{month:<10} {mean:<12.2f} {std:<10.2f} {int(count):<10}")
        
      
    # Plot: Boxplot and Violinplot side by side
    fig = plt.figure(figsize=(14, 6))

    sns.boxplot(x='month', y='CAF', data=caf_df, palette='Set2')

    ax = plt.gca()  # Get current Axes
    ax.set_title('CAF Boxplot per Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('CAF')

    plt.tight_layout()
    plt.savefig("output/caf_by_month_boxplots.png") 
    
    # Save monthly CAF parameters to csv
    if save :
        mean_caf_by_month_df = mean_caf_by_month.reset_index()
        mean_caf_by_month_df.columns = ['Month', 'Mean_CAF']
        mean_caf_by_month_df.to_csv("data/mean_caf_by_month.csv", index=False)   

def plot_clear_sky_ghi(): 
    # Step 1: Define Bergen's location
    location = pvlib.location.Location(
        latitude=BERGEN_LAT,
        longitude=BERGEN_LON,
        tz="Europe/Oslo",
        altitude=BERGEN_ALT  # approximate altitude in meters
    )

    # Generate full-year datetime index (hourly)
    times = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h", tz=location.tz)
    
    # Compute clear-sky GHI using Ineichen model
    clearsky = location.get_clearsky(times, model='ineichen')  # returns GHI, DNI, DHI

    # Plot daily GHI totals
    daily_ghi = clearsky['ghi'].resample('D').sum()

    plt.figure(figsize=(12, 5))
    plt.plot(daily_ghi.index, daily_ghi.values, label='Daily GHI (clear-sky)', color='orange')
    plt.title("Daily Clear-Sky GHI in Bergen (2024)")
    plt.ylabel("Daily GHI (Wh/m²)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/clearsky_daily_ghi_bergen_annual.png")

def get_sky_type(cloud_cover, mixed_threshold, overcast_threshold): 
    """Assign sky type given total cloud cover and mixed and overcast thresholds"""
    if 0 <= cloud_cover < mixed_threshold:
        return "clear"
    elif mixed_threshold <= cloud_cover < overcast_threshold:
        return "mixed"
    elif overcast_threshold <= cloud_cover <= 100: 
        return "overcast"
    else:
        return "unknown"

def resolve_sky_type(group):
    # Group by unique day (using year and doy)
    mean_cloud_cover = group["cloud_cover"].mean()
    sky_type = get_sky_type(mean_cloud_cover, mixed_sky_threshold, overcast_sky_threshold)
    
    unique_datetimes = pd.to_datetime(group["datetime_start"], utc=True).unique()
    if len(unique_datetimes) > 1:
        diff = max(unique_datetimes) - min(unique_datetimes)
        if diff <= pd.Timedelta(seconds=30): 
            datetime_start = unique_datetimes.mean()
        else:
            datetime_start = pd.NaT # 24 missing values out of 1144, 16 of which in "overcast" sky type
    else:
        datetime_start = group["datetime_start"].iloc[0]
    
    return pd.Series({
        "cloud_cover": mean_cloud_cover,
        "month": group["month"].iloc[0],
        "season": group["season"].iloc[0],
        "sky_type": sky_type, 
        "datetime_start": datetime_start
    })
    
    
def get_daily_sky_conditions(df): 
    """Reshape dataframe to one that contains the date, cloud cover and sky type and one observation per day."""
    # Calculate daily probabilities from sentinel cloud data
    df_cloud_cover = df[["cloud_cover", "doy", "month", "season", "year", "datetime_start"]].copy()
    
    df_cloud_cover["datetime_start"] = pd.to_datetime(df_cloud_cover["datetime_start"], utc=True)
    
    # Define conditions and corresponding labels
    conditions = [
        df_cloud_cover["cloud_cover"] < mixed_sky_threshold,
        (df_cloud_cover["cloud_cover"] >= mixed_sky_threshold) & (df_cloud_cover["cloud_cover"] < overcast_sky_threshold),
        df_cloud_cover["cloud_cover"] >= overcast_sky_threshold
    ]
    sky_types = ["clear", "mixed", "overcast"]

    # Add sky_type column
    df_cloud_cover["sky_type"] = np.select(conditions, sky_types)
    
    # Aggregate cloud_cover as daily mean (optional), keep other info
    daily_df = df_cloud_cover.groupby(["year", "doy"]).apply(resolve_sky_type).reset_index()
    return daily_df


def create_monthly_sky_probabilities(daily_df):
    """Generate a table with monthly probabilities for clear, mixed and overcast skies. Plot probabilities and save to csv."""
    # Count number of days by month and sky_type
    counts = daily_df.groupby(["month", "sky_type"]).size().unstack(fill_value=0)

    # Total days per month
    counts["total_days"] = counts.sum(axis=1)

    # Compute probability (normalized values)
    probs = counts.div(counts["total_days"], axis=0)
    probs = probs.drop(columns="total_days")  # remove after division

    # Save lookup table
    probs.to_csv("data/monthly_sky_probabilities.csv")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Stacked bar for sky type fractions
    probs.plot(kind="bar", stacked=True, ax=ax1, colormap="Pastel1")
    ax1.set_ylabel("Sky Type Probability")
    ax1.set_title("Monthly Sky Type Probabilities (Clear / Mixed / Overcast)")
    ax1.set_xlabel("Month")
    ax1.set_xticks(range(12))
    ax1.set_xticklabels([month_labels[m-1] for m in range(1, 13)], rotation=0)

    # Overlay line plot for total observations per month
    ax2 = ax1.twinx()
    ax2.plot(counts.index - 1, counts["total_days"], color="black", marker="o", label="Total Observations")
    ax2.set_ylabel("Total Observations")
    ax2.set_ylim(0, 120)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("output/sky_types_monthly.png")
    return probs

def clear_sky_ghi_daily(doy):
    """For given day of year, get the total daily clear sky GHI in Bergen."""
    location = pvlib.location.Location(
        latitude=BERGEN_LAT,
        longitude=BERGEN_LON,
        tz="Europe/Oslo",
        altitude=BERGEN_ALT
    )

    # Generate datetime index for the given day of 2024
    date = datetime.datetime.strptime(f"2024 {doy:03}", "%Y %j").date()
    times = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:59", freq="15min", tz="Europe/Oslo")

    # Calculate clear sky irradiance
    clearsky = location.get_clearsky(times, model='ineichen')

    # Return total daily GHI (integrated over the day)
    ghi_daily = clearsky['ghi'].sum() * (15 * 60) / 3600  # Wh/m2 (15-min intervals converted to hours)
    return ghi_daily

def simulate_annual_ghi(monthly_sky_type_probs):
    """Simulate the annual GHI at surface level for Bergen based on sky probabilities."""
    ghi = 0
    ghi_sky_type = {'clear': 0, 'mixed': 0, 'overcast': 0}
    types_per_month = {month: {'clear': 0, 'mixed': 0, 'overcast': 0} for month in range(1, 13)}
    ghi_per_month = {month: 0 for month in range(1,13)}
    overcast_cafs = pd.read_csv("data/mean_caf_by_month.csv")
    
    overcast_caf_dict = dict(zip(overcast_cafs['Month'], overcast_cafs['Mean_CAF']))
    
    for doy in range(1, 366):
        # Get month from DOY
        date = datetime.datetime.strptime(f"2024 {doy:03}", "%Y %j")
        month = date.month

        # Sample sky type from monthly probabilities
        probs = monthly_sky_type_probs.loc[month]
        sky_type = random.choices(
            population=probs.index.tolist(),
            weights=probs.values.tolist(),
            k=1
        )[0]

        types_per_month[month][sky_type] += 1

        # Get daily clear sky GHI
        clearsky_ghi = clear_sky_ghi_daily(doy)

        # Attenuate by cloud factor
        # If overcast, use monthly overcast CAF values
        if sky_type == "overcast": 
            caf = overcast_caf_dict[month]
        else: 
            caf = CAF[sky_type]
            
        daily_ghi = clearsky_ghi * caf
        
        if doy % 30 == 0:
            print(f"doy {doy}, daily ghi {np.round(daily_ghi, 2)}")

        ghi += daily_ghi
        ghi_sky_type[sky_type] += daily_ghi
        ghi_per_month[month] += daily_ghi

    return ghi, ghi_sky_type, types_per_month, ghi_per_month

def print_single_run_results(ghi, ghi_sky_type, types_per_month, ghi_per_month): 
    """Print the total annual ghi, the ghi contribution of each sky type, and the sky type distribution"""
    
    # Print results
    print("\nSky type distribution per month (days):")
    for month in range(1, 13):
        total_days = sum(types_per_month[month].values())
        monthly_contrib = ghi_per_month[month]/1000     #kWh/m²
        monthly_contrib_pct = monthly_contrib/(ghi/1000) * 100
        print(f"Month {month:2}: {np.round(monthly_contrib, 1)} kWh/m² ({np.round(monthly_contrib_pct,1)} %) : ", end="")
        for sky_type in ['clear', 'mixed', 'overcast']:
            count = types_per_month[month][sky_type]
            percent = count / total_days * 100 if total_days > 0 else 0
            print(f"{sky_type}: {count} ({percent:.1f}%)  ", end="")
        print()
    
    print("Total annual GHI (Wh/m2):", round(ghi, 2))
    for k, v in ghi_sky_type.items():
        print(f"{k.capitalize()} contribution: {round(v, 2)} Wh/m2 ({round(v/ghi*100, 1)}%)")

def repeat_experiment(n_it, monthly_probs): 
    """Repeat the ghi simulation multiple (n_it) times and gather statistics"""
    
    ghi = []
    ghi_sky_type = []
    types_per_month = []
    ghi_per_month = []
    
    for i in range(n_it): 
        ghi_i, ghi_sky_type_i, types_per_month_i, ghi_per_month_i = simulate_annual_ghi(monthly_probs)
        ghi.append(ghi_i)
        ghi_sky_type.append(ghi_sky_type_i)
        types_per_month.append(types_per_month_i)
        ghi_per_month.append(ghi_per_month_i)
        print_single_run_results(ghi_i, ghi_sky_type_i, types_per_month_i, ghi_per_month_i)
        
    return ghi, ghi_sky_type, types_per_month, ghi_per_month

def plot_simulation_statistics(ghi, ghi_sky_type, types_per_month, ghi_per_month): 
    """Plot ghi contribution per month, overall ghi and GHI contribution per sky type."""
 
    sky_types = ["clear", "mixed", "overcast"]
    months = range(1, 13)

    # --- 1. Monthly sky type distribution with error bars ---
    monthly_counts = {sky: [] for sky in sky_types}

    for month in months:
        for sky in sky_types:
            counts = [run[month][sky] for run in types_per_month]
            monthly_counts[sky].append((np.mean(counts), np.std(counts)))

    # Separate into means and stds
    means = {sky: [monthly_counts[sky][i][0] for i in range(12)] for sky in sky_types}
    stds = {sky: [monthly_counts[sky][i][1] for i in range(12)] for sky in sky_types}

    x = np.arange(12)
    width = 0.25

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    ax1 = axs[0]
    ax2 = ax1.twinx()  # secondary y-axis for GHI


    for i, sky in enumerate(sky_types):
        ax1.bar(x + i * width, means[sky], width=width, yerr=stds[sky], label=sky.capitalize(), capsize=4, color=skytype_colors[i])

    # Line plot: monthly GHI contribution
    # Convert list of dicts to 2D array: rows = runs, cols = months
    monthly_values = np.array([[run[month] for month in range(1, 13)] for run in ghi_per_month])
    ghi_monthly_mean = np.mean(monthly_values, axis=0) / 1000  # to kWh/m²
    line = ax2.plot(x + width, ghi_monthly_mean, color='black', ls= "--", marker='o', label='Monthly GHI (kWh/m²)', linewidth=2, alpha=0.7)

    # Combine handles from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")


    # Axis settings
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(month_labels)
    ax1.set_ylabel("Number of Days")
    ax2.set_ylabel("Monthly GHI (kWh/m²)")
    ax1.set_title("Average Monthly Sky Type Distribution and Monthly GHI")
    ax1.grid(True, axis="y")
    

    # --- 2. GHI contribution per sky type (bar with error bars) ---
    ghi_by_type = {sky: [run[sky] for run in ghi_sky_type] for sky in sky_types}
    ghi_means = [np.mean(ghi_by_type[sky])/1000 for sky in sky_types] #transform to kWh/m2
    ghi_stds = [np.std(ghi_by_type[sky])/1000 for sky in sky_types]

    total_mean = np.mean(ghi)/1000
    total_std = np.std(ghi)/1000

    axs[1].bar(sky_types, ghi_means, yerr=ghi_stds, capsize=5, color=skytype_colors)
    axs[1].set_ylabel("GHI Contribution (kWh/m²)")
    caf_str = ', '.join([f"{k.capitalize()}: {v}" for k, v in CAF.items()])
    axs[1].set_title(f"Annual GHI Contributions per Sky Type\n"
                     f"Total GHI: {total_mean:.0f} ± {total_std:.0f} kWh/m² | CAF: {caf_str}")
    axs[1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("output/simulation_statistics.png")
    plt.close()
    

if __name__=="__main__":
    #plot_clear_sky_ghi()
    # Run once to generate dataframes and save:
    # Read cleaned 1M Flesland dataset
    # Clean seklima dataset with 10 min resolution for 2024
    #ghi_df = clean_dataset_seklima_ghi("data/seklima_ghi_flesland_2024.csv", "data/seklima_ghi_flesland_2024_cleaned.csv")
    # 1 -Minute resolution
    """ ghi_df = pd.read_csv("data/frost_ghi_1M_Flesland_filtered.csv")
    ghi_df = ghi_df.rename(columns={"local_time": "timestamp", "value": "mean_ghi_1M"})
     """
    # Add sky conditions to dataframe
    df = pd.read_csv("data/s2_cloud_cover_cleaned.csv")
    df_sky_types = get_daily_sky_conditions(df)
    """df_overcast = df_sky_types[
        (df_sky_types["sky_type"] == "overcast") & 
        (df_sky_types["datetime_start"].notna())
    ]    
    # Get list of acquisition times
    acq_times = df_overcast["datetime_start"].unique()
    
    # Extract time of day (hour and minute only)
    acq_daytimes = acq_times.time
    overcast_daytimes = pd.to_datetime(ghi_df["timestamp"], utc=True).dt.time

    # Assert range of acq_daytimes within range of overcast_daytimes
    acq_min = min(acq_daytimes)
    acq_max = max(acq_daytimes)
    overcast_min = min(overcast_daytimes)
    overcast_max = max(overcast_daytimes)

    assert overcast_min <= acq_min <= overcast_max, "s2 min acquisition time is out of Frost GHI time range"
    assert overcast_min <= acq_max <= overcast_max, "s2 max acquisition time is out of Frost GHI time range"
  
    print("Number of overcast satellite observations: ", len(acq_times))
    caf_df = calculate_overcast_caf(acq_times, ghi_df)
      
     
    caf_df = pd.read_csv("data/caf_scores.csv")
    visualize_overcast_caf_params(caf_df, save=True)
    """
     
    monthly_probs = create_monthly_sky_probabilities(df_sky_types)
    
    # Run several simulations 
    n_it = 30
    ghi, ghi_sky_types, types_per_month, ghi_per_month = repeat_experiment(n_it, monthly_probs)
    plot_simulation_statistics(ghi, ghi_sky_types, types_per_month, ghi_per_month)
    
    # How to model mixed days 
    # generate a high resolution cloud map based on pixel probabilities? 
    # Calculate shade, aso based on cloud height and cloud properties? 
    # calculate shady pixels as GHI = GHI_clear x CAF_monthly_mixed? For this I need to estimate the CAF on the pixel of 
    # Flesland station for each month, only if there are clouds and it is a mixed sky
    # Calculate clear pixels as GHI = GHI_clear
    # Sum for each pixel to get daily GHI 
    # Will be very intensive runtimes if I do this for 60% of days of the year
    # In a first draft of the model I could just take the average cloud map of the month and calculate cloud based on that
    # Then get a cloud probability distribution for each pixel and draw from that to generate new cloud map
    # Set a pixel to cloudy if all neighbouring pixels are cloudy, set a pixel to clear if all neighboring pixels are 
    # clear 
    # Check if images are useful 
    