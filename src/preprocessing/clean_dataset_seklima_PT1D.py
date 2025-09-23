import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

seklima_df = pd.read_csv("data/cloud_cover_snow_seklima_fullrange.csv", sep=";")
seklima_df["mean_cloud_cover"] = pd.to_numeric(seklima_df["mean_cloud_cover"], errors='coerce')
seklima_df["max_cloud_cover"] = pd.to_numeric(seklima_df["max_cloud_cover"], errors='coerce')
seklima_df["min_cloud_cover"] = pd.to_numeric(seklima_df["min_cloud_cover"], errors='coerce')
# Snow is encoded as follows 
# 1 = mostly snow free ground
# 2 = equal parts of snow covered and snow free ground
# 3 = mostly snow covered ground
# 4 = completely snow covered ground
# 0 or -1/. = no snow
seklima_df["snow_cover"] = seklima_df["snow_cover"].replace("-", "-1").astype(str)
print("------------------- Snow cover --------------------------")
print(seklima_df["snow_cover"].value_counts())

seklima_df["sunshine_duration"] = pd.to_numeric(seklima_df["sunshine_duration"], errors='coerce')
seklima_df = seklima_df[seklima_df["station"] != 'SN50450']
seklima_df["middle_cloud_cover"] = seklima_df[["max_cloud_cover", "min_cloud_cover"]].mean(axis=1)
seklima_df["name"] = seklima_df["name"].replace("Bergen - Florida", "Florida")

# Convert oktas to percent cloud cover
seklima_df["mean_cloud_cover_pct"] = (seklima_df["mean_cloud_cover"] / 8) * 100
seklima_df["middle_cloud_cover_pct"] = (seklima_df["middle_cloud_cover"] / 8) * 100
seklima_df["max_cloud_cover_pct"] = (seklima_df["max_cloud_cover"] / 8) * 100
seklima_df["min_cloud_cover_pct"] = (seklima_df["min_cloud_cover"] / 8) * 100

print(seklima_df[["mean_cloud_cover", "middle_cloud_cover", "max_cloud_cover", "min_cloud_cover", "snow_cover", "sunshine_duration"]].describe())
print(seklima_df.dtypes)
print("Number of missing values per column:")
print(seklima_df.isna().sum().sort_values(ascending=False)) # no missing data in "date"


# Convert "date" to datetime
seklima_df["date"] = pd.to_datetime(seklima_df["date"],format="%d.%m.%Y",)

# Extract components
seklima_df["year"] = seklima_df["date"].dt.year
seklima_df["doy"] = seklima_df["date"].dt.dayofyear
seklima_df["month"] = seklima_df["date"].dt.month

# Assign seasons
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

seklima_df["season"] = seklima_df["month"].apply(get_season)

print(seklima_df[["year", "doy"]].describe())
print(seklima_df.groupby("month").describe())
print(seklima_df.groupby("season").describe())

# Plot mean and middle cloud cover for Flesland and Florida
# Filter for Florida and Flesland
stations_to_plot = seklima_df["name"].unique()
colors = ["#CC2936", "#2E86AB"]
print("Unique station names: ", stations_to_plot)

# Group by station and month, then calculate means
monthly_means = seklima_df.groupby(["name", "month"])[["mean_cloud_cover_pct", "middle_cloud_cover_pct"]].mean().reset_index()

# Plot setup
plt.figure(figsize=(10, 6))

# Plot for each station
for i, station_name in enumerate(stations_to_plot):
    station_data = monthly_means[monthly_means["name"] == station_name]
    
    plt.plot(
        station_data["month"],
        station_data["mean_cloud_cover_pct"],
        label=f"{station_name} - Mean Cloud Cover",
        marker='o',
        color=colors[i]
    )
    plt.plot(
        station_data["month"],
        station_data["middle_cloud_cover_pct"],
        label=f"{station_name} - Middle Cloud Cover",
        marker='s',
        linestyle='--',
        alpha=0.8,
        color=colors[i]
    )

# Styling
plt.title("Monthly Average Cloud Cover (Mean vs Middle) — Florida and Flesland")
plt.xlabel("Month")
plt.ylim(0, 100)
plt.ylabel("Cloud Cover (%)")
plt.xticks(ticks=range(1, 13), labels=[
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/mean_middle_cloud_cover_Florida_vs_Flesland_pct.png")

# Plot snow 
# Define snow cover labels
snow_labels = {
    "-1": "No data",
    "0": "No snow",
    "1": "Mostly snow-free",
    "2": "Mixed snow",
    "3": "Mostly snow-covered",
    "4": "Completely snow-covered"
}

# Convert snow_cover to numeric and map labels
seklima_df["snow_cover_label"] = seklima_df["snow_cover"].map(snow_labels)
print("------------------------ Snow cover labels ---------------------")
print(seklima_df["snow_cover_label"].value_counts())

# Count occurrences of each snow cover label per month and station
snow_counts = (
    seklima_df.groupby(["name", "month", "snow_cover_label"])
    .size()
    .reset_index(name="count")
)

snow_categories_order = [
    "No data",
    "No snow",
    "Mostly snow-free",
    "Mixed snow",
    "Mostly snow-covered",
    "Completely snow-covered"
]

# Create a complete index for pivot (to avoid missing categories)
full_index = pd.MultiIndex.from_product(
    [stations_to_plot, range(1, 13), snow_categories_order],
    names=["name", "month", "snow_cover_label"]
)

# Reindex and pivot
snow_counts_full = snow_counts.set_index(["name", "month", "snow_cover_label"]).reindex(full_index, fill_value=0).reset_index()
pivot_df = snow_counts_full.pivot_table(
    index=["name", "month"],
    columns="snow_cover_label",
    values="count",
    fill_value=0
)

# Colors
colors = {
    "No data": "darkgray",
    "No snow": "#e0e0e0",
    "Mostly snow-free": "#a6cee3",
    "Mixed snow": "#1f78b4",
    "Mostly snow-covered": "#b2df8a",
    "Completely snow-covered": "#33a02c"
}

# Plot for Florida station only because Flesland has no snow data
station = "Florida"
station_data = pivot_df.loc[station]
station_data = station_data[snow_categories_order]  # consistent column order

station_data.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=[colors[c] for c in snow_categories_order]
)

plt.title(f"Monthly Snow Cover Distribution – {station}")
plt.xlabel("Month")
plt.ylabel("Number of Observations")
plt.xticks(
    ticks=range(12),
    labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    rotation=45
)
plt.legend(title="Snow Cover")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"output/monthly_snow_cover_{station}.png")

# Plot snow for each doy 
# Count occurrences of each snow cover label per DOY and station
snow_counts_doy = (
    seklima_df.groupby(["name", "doy", "snow_cover_label"])
    .size()
    .reset_index(name="count")
)

# Create multi index 
full_index_doy = pd.MultiIndex.from_product(
    [stations_to_plot, range(1, 366), snow_categories_order],
    names=["name", "doy", "snow_cover_label"]
)

# Reindex and pivot
snow_counts_doy_full = snow_counts_doy.set_index(["name", "doy", "snow_cover_label"]).reindex(full_index_doy, fill_value=0).reset_index()
pivot_df_doy = snow_counts_doy_full.pivot_table(
    index=["name", "doy"],
    columns="snow_cover_label",
    values="count",
    fill_value=0
)

# Plot for "Florida" station
station_data = pivot_df_doy.loc["Florida"]
station_data = station_data[snow_categories_order]  # consistent column order
    
# Plot all DOYs for "Florida" station in one stacked bar chart
ax = station_data.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    color=[colors[c] for c in snow_categories_order],
    title="Florida - Snow Cover Distribution per Day of Year (DOY)",
    ylabel="Number of Observations",
    xlabel="Day of Year (DOY)"
)

plt.legend(title="Snow Cover")
plt.xticks(
    ticks=range(0, 365, 30),  # Every 30th day (adjust as needed)
    labels=[f"DOY {i+1}" for i in range(0, 365, 30)],
    rotation=45
)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("output/doy_snow_cover_Florida.png")

# Plot missing values cloud cover 
grouped = seklima_df[seklima_df["name"] == "Flesland"].groupby(["year", "month"])
expected_per_month = grouped['mean_cloud_cover'].size()
missing_counts = grouped['mean_cloud_cover'].apply(lambda m : m.isna().sum())
missing_pct = (missing_counts/expected_per_month) * 100

missing_df = pd.DataFrame({
    'missing_count': missing_counts,
    'missing_percentage': missing_pct
}).reset_index()

# Plot
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Missing count
sns.barplot(data=missing_df, x='year', y='missing_count', hue='month', ax=ax[0], palette='viridis')
ax[0].set_title('Missing "mean_cloud_cover" Count per Month and Year')
ax[0].set_ylabel('Missing Count')
ax[0].legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

# Missing percentage
sns.barplot(data=missing_df, x='year', y='missing_percentage', hue='month', ax=ax[1], palette='viridis')
ax[1].set_title('Missing "mean_cloud_cover" Percentage per Month and Year')
ax[1].set_ylabel('Missing Percentage (%)')
ax[1].set_xlabel('Year')
ax[1].legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("output/seklima_mean_cloud_cover_NA_Flesland.png")

seklima_df["is_available"] = ~seklima_df["mean_cloud_cover"].isna()

# Plot by day of year 
missing_by_day = (
    seklima_df.groupby(["name", "doy"])["is_available"]
    .sum()
    .reset_index()
)

# Plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=missing_by_day, x="doy", y="is_available", hue="name")
plt.title('Available "mean_cloud_cover" Observations per Day-of-Year')
plt.xlabel('Day of Year')
plt.ylabel('Available Observations (max = 10)')
plt.legend(title='Station')
plt.grid(True)
plt.tight_layout()
plt.savefig("output/seklima_mean_cloud_cover_data_coverage_doy.png")


seklima_df.to_csv('data/cloud_cover_snow_seklima_fullrange_cleaned.csv', index=False)
