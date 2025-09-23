import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV
filename = "s2_cloud_cover_weighted_reducer.csv"
df = pd.read_csv('data/'+filename)
print("------------------------ Dataframe description ------------------------")
print("--------------------", filename, "--------------------")
print(df.describe())   # one row for each image 
print(df.dtypes)
print("------------------------ NA ------------------------")
print(df.isna().sum()) # only missing data in avg_cloud_prob

# Remove NA in avg_cloud_prob
print("------------------------ Without NA ------------------------")
df = df[~df['avg_cloud_prob'].isna()] # remove 3333 out of 8278
print(df.describe())

# Check where total_pixels == 0
print("------------------------ Total_pixels == 0 ------------------------")
print(df[df["total_pixels"] == 0].sum()) # no values

# Describe all relevant columns (cloud_cover, total_pixels, cloudy_pixels, avg_cloud_prob)
print(df[['cloud_cover', 'total_pixels', 'cloudy_pixels', 'avg_cloud_prob']].describe())

# Consistency checks
assert df['cloud_cover'].between(0, 100).all()
assert df['avg_cloud_prob'].between(0, 100).all()
assert (df['total_pixels'] > 0).all()

# Find rows where cloudy_pixels > total_pixels
epsilon = np.finfo(df["total_pixels"].dtype).eps
print(f"Machine epsilon for column total_pixels (dtype float64): {epsilon}")

# Only keep rows where difference is larger than machine precision relative to total_pixels
mask = (df['cloudy_pixels'] - df['total_pixels']) > (epsilon * df['total_pixels'])
invalid_rows = df[mask].copy()
print(f"Rows with cloudy_pixels > total_pixels (beyond machine precision): {len(invalid_rows)}")

df = df[~mask].copy()
assert not ((df['cloudy_pixels'] - df['total_pixels']) > (epsilon * df['total_pixels'])).any(), \
    "There are still rows with cloudy_pixels > total_pixels beyond machine precision."

# Highly repeated values 
print(df['avg_cloud_prob'].value_counts().head(10))

# calculate correlation between avg_cloud_prob and cloud_cover and plot
corr = df[['avg_cloud_prob', 'cloud_cover']].corr()
print("Correlation matrix:\n", corr)

sns.scatterplot(data=df, x='avg_cloud_prob', y='cloud_cover', alpha=0.5)
plt.title('avg_cloud_prob vs cloud_cover (r = 0.9965)')
plt.xlabel('Average Cloud Probability')
plt.ylabel('Cloud Cover (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig("output/corr_cloud_cover_avg_cloud_prob.png")

# check that cloud cover is always equals cloudy_pixels/total_pixels
df['computed_cloud_cover'] = (df['cloudy_pixels'] / df['total_pixels']) * 100
difference = (df['cloud_cover'] - df['computed_cloud_cover']).abs()
print("Max difference in cloud cover calculation:", difference.max())
# If you want to assert:
assert difference.max() < 1, "Cloud cover calculation mismatch!"

# Relabel season
season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
df['season'] = df['season'].map(season_map)
print(df['season'].value_counts())

# Group by season and list unique months
season_months = df.groupby('season')['month'].unique()

for season, months in season_months.items():
    print(f"{season}: {sorted(months)}")

# check doy, month, year ranges 
print("DOY range:", df['doy'].min(), "-", df['doy'].max())
print("Months:", sorted(df['month'].unique()))
print("Years:", sorted(df['year'].unique()))

# Add hours and milliseconds
# Convert milliseconds to datetime
df['datetime_start'] = pd.to_datetime(df['system:time_start'], unit='ms', utc=True)

# Convert UTC datetime to local Oslo time (includes daylight savings time DST)
df['datetime_start'] = df['datetime_start'].dt.tz_convert('Europe/Oslo')

# Extract start time components from Oslo datetime
df['hour_start'] = df['datetime_start'].dt.hour
df['minute_start'] = df['datetime_start'].dt.minute
df['second_start'] = df['datetime_start'].dt.second

# Drop the .geo column if it exists
df.drop(columns=['.geo'], inplace=True, errors='ignore')


# View result
print(df[["datetime_start", "hour_start", "minute_start", "second_start"]].head())
print(df[["datetime_start", "hour_start", "minute_start", "second_start"]].dtypes)


# Correlation between total_pixel and cloud_cover? 
corr = df[['total_pixels', 'cloud_cover']].corr()  # no correlation (r=0.0367)
print("Correlation matrix:\n", corr)

plt.figure()
sns.scatterplot(data=df, x='total_pixels', y='cloud_cover', alpha=0.5)
plt.title('total_pixels vs cloud_cover')
plt.xlabel('Total pixels covering Bergen ROI')
plt.ylabel('Cloud Cover (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig("output/corr_cloud_cover_total_pixels.png")

# Plot data coverage
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Month
df['month'].value_counts().sort_index().plot(kind='bar', ax=axs[0, 0])
axs[0, 0].set_title('Images per Month')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('Count')

# Season
df['season'].value_counts().plot(kind='bar', ax=axs[0, 1])
axs[0, 1].set_title('Images per Season')
axs[0, 1].set_xlabel('Season')
axs[0, 1].set_ylabel('Count')

# Day of Year
df['doy'].value_counts().sort_index().plot(kind='line', ax=axs[1, 0])
axs[1, 0].set_title('Images per Day of Year')
axs[1, 0].set_xlabel('Day of Year')
axs[1, 0].set_ylabel('Count')

# Year
df['year'].value_counts().sort_index().plot(kind='bar', ax=axs[1, 1])
axs[1, 1].set_title('Images per Year')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig("output/doy_monthly_seasonal_yearly_coverage.png")

# Save back to csv
df.to_csv('data/s2_cloud_cover_cleaned.csv', index=False)
