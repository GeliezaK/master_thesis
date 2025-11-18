import pandas as pd 
import numpy as np 

def simulate_annual_ghi(monthly_sky_type_prob_filepath, monthly_clear_sky_index_filepath,
                        n_years=100):
    """Given monthly sky type probabilities and monthly clear sky index for each sky type, 
    simulate n_years of the area-wide monthly and annual GHI. Returns a dataframe with one
    row per observation, including columns GHI (integral), sky type (clear/mixed/overcast),
    clear-sky index, month. """
    
    results = []
    day_range = range(1, 365*n_years)
    
    for day in day_range: 
        # Get day of year
        doy = day % 365
        
        # Get month
        month = None
        
        # Draw sky type 
        sky_type = None 
        
        # Draw clear-sky index according to month and sky type
        k = None 
        
        # Get hours between sunset and sunrise 
        hours = []
        for hour in hours:
            # Get clear-sky irradiance for this day and time from LUT

            # Multiply times k 
            GHI_hourly = None
            GHI_daily += GHI_hourly 
            
        # Append to results 
        year = day // 365 + 1
        results.append({
            "day_count": day,
            "doy": doy,
            "month": month,
            "year": year,
            "GHI_daily": GHI_daily,
            "k": k, 
            "sky_type": sky_type
        })
            

    
    return pd.DataFrame(results)
    


if __name__ == "__main__": 
    monthly_sky_type_prob_filepath = "data/processed/monthly_sky_probabilities.csv"
    monthly_clear_sky_index_filepath = "data/processed/monthly_clear_sky_index.csv"
    simulate_annual_ghi(monthly_sky_type_prob_filepath, monthly_clear_sky_index_filepath)