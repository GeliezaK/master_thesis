# =============================================================================
# This script provides functions to apply the quality control criteria by 
# Grini (2015), https://hdl.handle.net/11250/292854 to 1M and 1H mean GHI
# measurements from Florida and Flesland weather stations 
# =============================================================================

import pvlib 
from pvlib.location import Location
import numpy as np
from src.model.instantaneous_GHI_model import build_ghi_clear_lut
from src.model import CENTER_LAT, CENTER_LON



def get_toa_irradiance(lat, lon, datetime):
    """Get extraterrestrial (top of atmosphere) irradiance [W/m²] for given lat/lon/datetime."""
    site = Location(lat, lon, 'Europe/Oslo', 12, 'Bergen')
    solpos = site.get_solarposition(datetime)

    # Extra-terrestrial radiation normal to sun
    I0_normal = pvlib.irradiance.get_extra_radiation(datetime.dayofyear)

    # Project onto horizontal plane
    I0_horizontal = I0_normal * np.maximum(0, np.sin(np.radians(solpos['apparent_elevation'])))

    return I0_horizontal.values[0]

# ---------------------------------------------------------
# Helper: threshold for clear-sky exceeding test
# ---------------------------------------------------------
def cs_threshold(ics):
    """Return clear-sky threshold for a given array of clear sky irradiances."""
    # Apply threshold to remove GHI > f ∗ ICS + a, where
    # f = 2, a = 0 if ICS ≤ 100 W /m2
    # f = 1.05, a = 95 if ICS > 100 W /m
    ics = np.asarray(ics)

    # Conditions
    low = ics <= 100
    high = ~low

    # Allocate output
    threshholds = np.empty_like(ics, dtype=float)

    # Apply formulas
    threshholds[low] = 2 * ics[low]
    threshholds[high] = 1.05 * ics[high] + 95

    return threshholds


def flag_observations(df_in, obs_col, datetime_col="datetime"):
    """Create a flag column in df_in to flag data according to quality control criteria in Grini (2015).
    Prints the number of observations for each flag.
    Return dataframe df_in including flag column."""
    # Make subset copy
    df = df_in[[obs_col, datetime_col]].copy()
    
    # Get theoretical maximum irradiance   
    solpos = pvlib.solarposition.get_solarposition(
        df[datetime_col],
        CENTER_LAT,
        CENTER_LON
    )

    # Add helper column with TOA radiation (top-of-atmosphere/extraterrestrial)
    elev = np.radians(solpos["apparent_elevation"])
    I0_normal = pvlib.irradiance.get_extra_radiation(df[datetime_col].dt.dayofyear)
    df["TOA"] = I0_normal * np.maximum(0, np.sin(elev.to_numpy()))

    # -----------------------------------------
    # Compute time components
    # -----------------------------------------
    df["doy"] = df[datetime_col].dt.dayofyear
    df["hour"] = df[datetime_col].dt.hour + df[datetime_col].dt.minute/60

    # Round hour to nearest LUT hour
    df["hour_int"] = np.round(df["hour"])
       
    # -----------------------------------------
    # Lookup clear-sky GHI from LUT
    # -----------------------------------------
    GHI_CLEAR_LUT = build_ghi_clear_lut()
    doy_idx = df["doy"].astype(int) - 1
    hour_idx = df["hour_int"].astype(int)
    df["ICS"] = GHI_CLEAR_LUT[doy_idx, hour_idx]

    # -----------------------------------------
    # Initialize flags
    # -----------------------------------------
    df["flag"] = "OK"

    # Negative values -> N
    df.loc[df[obs_col] < 0, "flag"] = "N"

    # GHI > TOA -> T
    df.loc[df[obs_col] > df["TOA"], "flag"] = "T"

    # Clear-sky check -> CS
    mask_ok = df["flag"] == "OK"
    df.loc[mask_ok & (df[obs_col] > cs_threshold(df["ICS"])), "flag"] = "CS"
    
    # Print counts of each flag 
    print("TOA flagged:", (df["flag"] == "T").sum())
    print("CS flagged:", (df["flag"] == "CS").sum())
    print("N flagged:", (df["flag"] == "N").sum())
    
    # Print number of missing and total values for GHI columns
    n_missing = df[obs_col].isna().sum()
    n_total = len(df[obs_col])
    print(f"{obs_col}: missing = {n_missing}, total = {n_total}")
    
    n_valid = df.loc[df["flag"] == "OK", obs_col].notna().sum()    

    print(f"Valid obs for {obs_col}: {n_valid}")
    
    df_in[f"{obs_col}_flag"] = df["flag"]
    return df_in

