import pandas as pd
import matplotlib.pyplot as plt
import pytz 

# List of filenames
filenames = [
    "data/cloud_cover_2015-06-01_2020-05-05_Flesland_PT1H.csv",
    "data/cloud_cover_2015-06-01_2025-05-15_Florida_PT3H.csv",
    "data/cloud_cover_2020-05-05_2021-08-31_Flesland_PT10M.csv",
    "data/cloud_cover_2021-09-01_2023-06-30_Flesland_PT10M.csv",
    "data/cloud_cover_2023-07-01_2025-05-01_Flesland_PT10M.csv"
]

# Standard column names you want
column_rename_map = {
    "Name": "name",
    "Station": "station",
    "Time(norwegian mean time)": "datetime",
    "Cloud cover": "cloud_cover"
}

# Load, rename, merge all files together
dfs = []
for file in filenames:
    df = pd.read_csv(file, sep=";")
    df.rename(columns=column_rename_map, inplace=True)
    dfs.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Sort by datetime
oslo = pytz.timezone("Europe/Oslo")
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'],format="%d.%m.%Y %H:%M")
merged_df['datetime'] = merged_df['datetime'].dt.tz_localize(oslo, ambiguous=False, nonexistent="shift_forward") # Make timezone-aware
merged_df.sort_values(by='datetime', inplace=True)

# Rename florida station
merged_df["name"] = merged_df["name"].replace("Bergen - Florida", "Florida")


# Extract components
merged_df["year"] = merged_df["datetime"].dt.year
merged_df["doy"] = merged_df["datetime"].dt.dayofyear
merged_df["month"] = merged_df["datetime"].dt.month
merged_df["hour"] = merged_df["datetime"].dt.hour
merged_df["minute"] = merged_df["datetime"].dt.minute

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

merged_df["season"] = merged_df["month"].apply(get_season)

print(merged_df[merged_df["name"] == "Florida"]["hour"].value_counts())

# Converge to percent - remove all values that are not oktas
print("Number of missing cloud cover values: ", len(merged_df[(merged_df['cloud_cover'] < 0) | (merged_df['cloud_cover'] > 8)]))
print("Number of observations: ", len(merged_df))
merged_df = merged_df[(merged_df['cloud_cover'] >= 0) & (merged_df['cloud_cover'] <= 8)]
merged_df["cloud_cover_pct"] = (merged_df["cloud_cover"] / 8) * 100

assert ((merged_df["cloud_cover_pct"] >= 0)  & (merged_df["cloud_cover_pct"] <= 100)).all()

# Save to CSV
output_filename = "data/cloud_cover_2015-06-01_2025-05-01_Florida_Flesland.csv"
merged_df.to_csv(output_filename, index=False)

# Read s2 data frame 
s2 = pd.read_csv('data/s2_cloud_cover_cleaned.csv')
sub_s2 = s2[["year", "month", "season", "doy", "hour_start", "minute_start", "second_start", "datetime_start", "cloud_cover"]].copy()
# Convert this again because if pandas reads it, it is dtype "object"
sub_s2['datetime_start'] = pd.to_datetime(sub_s2['datetime_start'], utc=True)
sub_s2['datetime_start'] = sub_s2['datetime_start'].dt.tz_convert('Europe/Oslo')
print(sub_s2.head())

sub_stations = merged_df[["name", "year", "month", "season", "doy", "hour", "minute", "datetime", "cloud_cover_pct"]].copy()
print("Sentinel-2 dtype datetime: ", sub_s2["datetime_start"].dtype)
print("Sub-stations dtype datetime: ", sub_stations["datetime"].dtype) # they have to be the same dtype for merging

# Merge stations data to s2 by nearest timestamp
# Split stations by name
flesland_df = sub_stations[sub_stations['name'] == 'Flesland'][['datetime', 'cloud_cover_pct']].copy()
florida_df = sub_stations[sub_stations['name'] == 'Florida'][['datetime', 'cloud_cover_pct']].copy()

# Rename columns for clarity before merging
flesland_df.rename(columns={'datetime': 'Flesland_timestamp', 'cloud_cover_pct': 'Flesland_cloud_cover'}, inplace=True)
florida_df.rename(columns={'datetime': 'Florida_timestamp', 'cloud_cover_pct': 'Florida_cloud_cover'}, inplace=True)

# Sort all sub-dataframes as a necessary prerequisite
sub_s2 = sub_s2.sort_values('datetime_start')
flesland_df = flesland_df.sort_values('Flesland_timestamp')
florida_df = florida_df.sort_values('Florida_timestamp')

# Merge to find closest Flesland observation by time
merged_flesland = pd.merge_asof(sub_s2.sort_values('datetime_start'), 
                               flesland_df, 
                               left_on='datetime_start', 
                               right_on='Flesland_timestamp', 
                               direction='nearest')

# Merge to find closest Florida observation by time
merged_all = pd.merge_asof(merged_flesland.sort_values('datetime_start'), 
                           florida_df, 
                           left_on='datetime_start', 
                           right_on='Florida_timestamp', 
                           direction='nearest')

merged_all.rename(columns={'datetime_start': 's2_timestamp',
                          'cloud_cover': 's2_cloud_cover'}, inplace=True)

# Plot time distances between observations
# Compute time differences in minutes
merged_all['delta_florida'] = (merged_all['s2_timestamp'] - merged_all['Florida_timestamp']).abs().dt.total_seconds() / 60
merged_all['delta_flesland'] = (merged_all['s2_timestamp'] - merged_all['Flesland_timestamp']).abs().dt.total_seconds() / 60

print("Delta florida: ", merged_all['delta_florida'].describe())
print("Delta flesland: ", merged_all['delta_flesland'].describe())

# Examine outliers
outliers = merged_all[merged_all['delta_flesland'] > 30]
print(f"Total number of observations: {len(merged_all)}")
print(f"Number of Flesland outliers (>30 min): {len(outliers)}")
print(f"outliers: ", outliers.head())
print(f"outliers delta values: ", outliers["delta_flesland"].describe())

for index, row in outliers.iterrows():
    print(row['s2_timestamp'], row['Flesland_timestamp'], row["s2_cloud_cover"], row['Flesland_cloud_cover'])
    # Looks like there are no data in Flesland for a couple of time ranges and Sentinel-2 images
    # are then mapped onto observations of a couple days before or after that date. 
    # It is safe to exclude them because it's just 68 observations out of 4948 and the data are due
    # to measurement errors, so they do not contain valid information

# Remove outliers
merged_all = merged_all[merged_all["delta_flesland"] <= 30]

# Divide Flesland into two periods: before 2020-05-05 (resolution 1H) and after (resolution 10 minutes)
change_point = pd.Timestamp("2020-05-05 14:00", tz="Europe/Oslo")

merged_all['flesland_period'] = merged_all['Flesland_timestamp'].apply(
    lambda x: 'Before 2020-05-05' if x < change_point else 'After 2020-05-05'
)

plt.figure(figsize=(10, 6))

# Plot Florida distances
plt.hist(merged_all['delta_florida'], bins=50, alpha=0.6, color='orange', label='Florida')

# Plot Flesland hourly
plt.hist(merged_all[merged_all['flesland_period'] == 'Before 2020-05-05']['delta_flesland'], 
         bins=50, alpha=0.6, color='skyblue', label='Flesland (hourly)')

# Plot Flesland 10-min
plt.hist(merged_all[merged_all['flesland_period'] == 'After 2020-05-05']['delta_flesland'], 
         bins=50, alpha=0.6, color='blue', label='Flesland (10-min)')

plt.xlabel("Time difference to Sentinel-2 measurement (minutes)")
plt.ylabel("Frequency")
plt.title("Time differences between Sentinel-2 and ground station measurements")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/time_difference_between_s2_florida_flesland.png")

print(merged_all.head())
merged_all.to_csv("data/cloud_cover_2015-06-01_2025-05-01_s2_Flesland_Florida_paired.csv", index=False)