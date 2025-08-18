import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import os
import numpy as np

def load_aod_files(pattern: str, aod_type: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df['AOD'] = aod_type
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def load_csv_files(file_pattern: str) -> pd.DataFrame:
    """Load and combine all CSV files matching a glob pattern."""
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df

def plot_GHI_difference_by_variable(df: pd.DataFrame, variable: str, output_path: str = None):
    """
    Plot GHI difference (max - min) for each unique DOY and hour.
    Separate lines for each DOY, clear and cloudy both shown.
    """
    plt.figure(figsize=(12, 6))

    doys = sorted(df['DOY'].unique())
    cmap = cm.get_cmap('tab20', len(doys))  # assign a unique color per DOY

    for i, doy in enumerate(doys):
        df_doy = df[df['DOY'] == doy]
        hours = sorted(df_doy['Hour'].unique())
        clear_diffs = []
        cloudy_diffs = []

        for hour in hours:
            subset = df_doy[df_doy['Hour'] == hour]
            subset = subset.sort_values(variable)

            if subset[variable].nunique() < 2:
                # Not enough variation to compute a difference
                clear_diffs.append(np.nan)
                cloudy_diffs.append(np.nan)
                continue

            min_var = subset[variable].min()
            max_var = subset[variable].max()

            ghi_clear_min = subset[subset[variable] == min_var]['GHI_clear'].mean()
            ghi_clear_max = subset[subset[variable] == max_var]['GHI_clear'].mean()
            ghi_cloudy_min = subset[subset[variable] == min_var]['GHI_cloudy'].mean()
            ghi_cloudy_max = subset[subset[variable] == max_var]['GHI_cloudy'].mean()

            clear_diffs.append(ghi_clear_max - ghi_clear_min)
            cloudy_diffs.append(ghi_cloudy_max - ghi_cloudy_min)

        color = cmap(i)
        plt.plot(hours, clear_diffs, linestyle='--', marker='o', color=color, label=f'DOY {doy} - Clear')
        plt.plot(hours, cloudy_diffs, linestyle='-', marker='x', color=color, label=f'DOY {doy} - Cloudy')

    plt.xlabel('Hour of Day')
    plt.ylabel(f'ΔGHI ({variable} max - min) [W/m²]')
    plt.title(f'GHI Difference vs Hour for Each DOY ({variable})')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='small', ncol=1)
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    plt.show()



def plot_GHI_vs_variable(df: pd.DataFrame, variable: str, output_path: str = None):
    """Plot GHI (clear/cloudy) vs a selected variable for each hour."""
    plt.figure(figsize=(12, 6))

    hours = sorted(df['Hour'].unique())
    cmap = cm.get_cmap('tab10', len(hours))  # distinct color per hour

    for i, hour in enumerate(hours):
        color = cmap(i)
        subset = df[df['Hour'] == hour].sort_values(variable)
        plt.plot(subset[variable], subset['GHI_clear'], color=color,
                 label=f'Hour {hour} - Clear', linestyle='--', marker='o')
        plt.plot(subset[variable], subset['GHI_cloudy'], color=color,
                 label=f'Hour {hour} - Cloudy', linestyle='-', marker='x')

    plt.xlabel(variable.replace("_", " "))
    plt.ylabel('Global Horizontal Irradiance (GHI)')
    plt.title(f'GHI (Clear and Cloudy) vs {variable.replace("_", " ")} for Different Hours of Day')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    
    plt.show()
    
def plot_monthly_AOD(df): 
    """Plot monthly average and sd of Aerosol optical thickness at 550nm from modis"""
    # Group by month and calculate mean and std dev
    monthly_stats = df.groupby('month')['Optical_Depth_055'].agg(['mean', 'std'])
    
    print(f"Monthly avg and sd AOD: ", monthly_stats)

    # Plot
    plt.figure(figsize=(10, 5))
    
    # Plot individual data points as scatter
    plt.scatter(df['month'], df['Optical_Depth_055'], 
                color='lightblue', alpha=0.6, label='Individual AOD values', zorder=1)

    plt.errorbar(
        x=monthly_stats.index,
        y=monthly_stats['mean'],
        yerr=monthly_stats['std'],
        fmt='-o',
        ecolor='gray',
        capsize=5,
        label='AOD 550 nm ± SD'
    )

    plt.xticks(ticks=range(1, 13), labels=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ])
    plt.xlabel('Month')
    plt.ylabel('Aerosol Optical Depth (550 nm)')
    plt.title('Monthly Average AOD at 550 nm ± Standard Deviation (Bergen)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/aod_550nm_monthly.png")
    plt.show()
    
def plot_GHI_vs_AOD(df):
    """Plot clear and cloudy sky GHI vs AOD profile (none or custom monthly 550nm)"""
    # Sort by hour for plotting
    df = df.sort_values(by='Hour')

    # Plot
    plt.figure(figsize=(10, 6))
    
    aod_types = ['none','custom', 'default']
    
    cmap = cm.get_cmap('tab20', len(aod_types)*2)

    # Plot lines for each AOD type and GHI type
    for i, aod_type in enumerate(aod_types):
        subset = df[df['AOD'] == aod_type]
        plt.plot(subset['Hour'], subset['GHI_clear'], color=cmap(i+3), label=f'{aod_type.capitalize()} AOD - Clear', linestyle='--', marker='o')
        plt.plot(subset['Hour'], subset['GHI_cloudy'], color=cmap(i+3), label=f'{aod_type.capitalize()} AOD - Cloudy', linestyle='-', marker='x')

    # Formatting
    plt.xlabel('Hour of Day')
    plt.ylabel('Global Horizontal Irradiance (GHI) [W/m²]')
    plt.title('Effect of AOD on GHI (Clear and Cloudy) (DOY 166) for Albedo = 0.174')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/GHI_vs_AOD_doy166.png")
    plt.show()
    
def plot_GHI_vs_AOD_diff(df):
    """Plot GHI differences (none - custom/default) for clear and cloudy skies vs Hour of Day."""

    # Filter only for AOD types present
    available_aod_types = df['AOD'].unique()
    expected_aod_types = ['none', 'custom', 'default']
    present = [aod for aod in expected_aod_types if aod in available_aod_types]

    if 'none' not in present or len(present) < 2:
        print("Need at least 'none' and one other AOD type (custom/default) to compute differences.")
        return

    # Pivot data for each GHI type
    ghi_vars = ['GHI_clear', 'GHI_cloudy']
    diffs = []

    for ghi in ghi_vars:
        pivot = df.pivot_table(index='Hour', columns='AOD', values=ghi)

        for comp_aod in ['custom', 'default']:
            if comp_aod in pivot.columns:
                diff = pivot['none'] - pivot[comp_aod]
                diffs.append((ghi, comp_aod, diff))

    # Plot
    plt.figure(figsize=(10, 6))
    color_map = {'custom': 'tab:purple', 'default': 'tab:red'}
    linestyle_map = {'GHI_clear': '--', 'GHI_cloudy': '-'}

    for ghi, comp_aod, diff_series in diffs:
        plt.plot(diff_series.index, diff_series.values,
                 label=f"None - {comp_aod.capitalize()} AOD - {'Clear' if 'clear' in ghi else 'Cloudy'}",
                 linestyle=linestyle_map[ghi],
                 marker='o',
                 color=color_map[comp_aod])

    # Formatting
    plt.xlabel('Hour of Day')
    plt.ylabel('ΔGHI [W/m²]')
    plt.title('GHI Difference Due to AOD (None - Custom/Default) (DOY 166) for Albedo = 0.174')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/GHI_vs_AOD_diff_doy166.png")
    plt.show()


if __name__ == "__main__":
    # --- Plot Satellite AOD ---
    # Load the CSV file
    df = pd.read_csv("data/maiac_monthly_bergen.csv") # from Modis
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    # Remove rows with missing AOD values
    df = df[pd.to_numeric(df['Optical_Depth_055'], errors='coerce').notnull()]
    df['Optical_Depth_055'] = df['Optical_Depth_055'].astype(float)*0.001
    
    plot_monthly_AOD(df)
    
    """
    # --- Plot GHI vs AOD ---
    # File patterns
    pattern_custom = "output/LUT/casestudy_aod_doy*_hod*_alb*_custom_aod.csv"
    pattern_none = "output/LUT/casestudy_aod_doy*_hod*_alb*_none_aod.csv"
    pattern_default = "output/LUT/casestudy_aod_doy*_hod*_alb*_default_aod.csv"


    # Load and merge
    df_custom = load_aod_files(pattern_custom, "custom")
    df_none = load_aod_files(pattern_none, "none")
    df_default = load_aod_files(pattern_default, "default")
    df = pd.concat([df_custom, df_none, df_default], ignore_index=True)

    # Convert columns
    df['Albedo'] = pd.to_numeric(df['Albedo'], errors='coerce')
    df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

    # Filter for albedo 0.174
    df = df[df['Albedo'] == 0.174]
    df = df[df['DOY'] == 166]

    plot_GHI_vs_AOD_diff(df)
    
    # --- Plot GHI vs Albedo ---
    albedo_pattern = "output/LUT/casestudy_albedo_doy*_hod*_alb*.csv"
    df_albedo = load_csv_files(albedo_pattern)
    df_albedo['Albedo'] = pd.to_numeric(df_albedo['Albedo'])
    df_albedo['Hour'] = pd.to_numeric(df_albedo['Hour'])
    #plot_GHI_vs_variable(df_albedo, variable='Albedo', output_path="output/GHI_vs_albedo.png")
    plot_GHI_difference_by_variable(df_albedo, variable='Albedo', output_path="output/GHI_diff_vs_hour_albedo.png")


    # --- Plot GHI vs Altitude ---
    altitude_pattern = "output/LUT/casestudy_alt_doy*_hod*_alt*.csv"
    df_alt = load_csv_files(altitude_pattern)
    df_alt['Altitude_km'] = pd.to_numeric(df_alt['Altitude_km'])
    df_alt['Hour'] = pd.to_numeric(df_alt['Hour'])
    #plot_GHI_vs_variable(df_alt, variable='Altitude_km', output_path="output/GHI_vs_altitude.png")
    plot_GHI_difference_by_variable(df_alt, variable='Altitude_km', output_path="output/GHI_diff_vs_hour_altitude.png")
    """
    
