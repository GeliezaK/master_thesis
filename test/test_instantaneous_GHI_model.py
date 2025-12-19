import unittest
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pathlib import Path
import sys
from datetime import datetime
from netCDF4 import Dataset
import os
import shutil
#sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from src.model.cloud_shadow import get_solar_angle, get_cloud_shadow_displacement
from src.model.instantaneous_GHI_model import get_closest_lut_entry
from src.model.shortwave_correction_factor import calculate_sw_dir_cor
from src.model import BBOX

class TestSurfaceGHIModel(unittest.TestCase):
    def setUp(self):
        # Create temporary test directory
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a small synthetic DEM (100x100)
        nrows, ncols = 10, 10

        # Create a 1D parabolic profile for rows (south → north)
        y = np.arange(nrows)
        y_profile = -((y - (nrows-1)/2) ** 2) + ((nrows-1)/2) ** 2
        y_profile = y_profile / y_profile.max()  # normalize 0–1

        # Same for columns (west → east)
        x = np.arange(ncols)
        x_profile = -((x - (ncols-1)/2) ** 2) + ((ncols-1)/2) ** 2
        x_profile = x_profile / x_profile.max()  # normalize 0–1

        # Combine row and column profiles to get a 2D hill
        self.dem_test = np.outer(y_profile, x_profile) * 1000.0  # scale to 0–1000 m

        # GeoTransform: top-left corner at (0,100), pixel size = 10x10
        transform = rasterio.transform.from_origin(
            BBOX["west"], BBOX["north"], 
            (BBOX["east"] - BBOX["west"]) / ncols, 
            (BBOX["north"] - BBOX["south"]) / nrows
        )

        self.dsm_filepath = os.path.join(self.test_dir, "test_dem.tif")
        with rasterio.open(
            self.dsm_filepath,
            "w",
            driver="GTiff",
            height=self.dem_test.shape[0],
            width=self.dem_test.shape[1],
            count=1,
            dtype=self.dem_test.dtype,
            crs="EPSG:4326",
            transform=transform
        ) as dst:
            dst.write(self.dem_test, 1)

        # Use a few test timestamps
        self.times = [
            datetime(2024, 6, 21, 12, 0, 0),
            datetime(2024, 3, 15, 9, 0, 0),
        ]
        
        # Setup for test lut 
        # Build a small synthetic LUT
        self.lut = pd.DataFrame([
            {"doy": 172, "hour": 12, "albedo": 0.2, "altitude_km": 0.5,
             "direct_clear": 800, "diffuse_clear": 100,
             "direct_cloudy": 400, "diffuse_cloudy": 300,
             "cloud_top_km": 5.0, "cot": 10.0, "cloud_phase": "water"},
            {"doy": 166, "hour": 12, "albedo": 0.2, "altitude_km": 0.5,
             "direct_clear": 550, "diffuse_clear": 30,
             "direct_cloudy": 0.1, "diffuse_cloudy": 350,
             "cloud_top_km": 3.0, "cot": 100.0, "cloud_phase": "water"},
            {"doy": 74, "hour": 9, "albedo": 0.2, "altitude_km": 0.5,
             "direct_clear": 600, "diffuse_clear": 150,
             "direct_cloudy": 200, "diffuse_cloudy": 400,
             "cloud_top_km": 8.0, "cot": 20.0, "cloud_phase": "ice"}
        ])

        self.unique_values = {
            "doy": self.lut["doy"].unique(),
            "hour": self.lut["hour"].unique(),
            "albedo": self.lut["albedo"].unique(),
            "altitude_km": self.lut["altitude_km"].unique(),
            "cloud_top_km": self.lut["cloud_top_km"].unique(),
            "cot": self.lut["cot"].unique(),
            "cloud_phase": self.lut["cloud_phase"].unique(),
        }
        

    def tearDown(self):
        # Remove created files and test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
                       
    def test_get_closest_lut_entry(self):
        """Test that correct values from lut are returned."""
        # Check that correct clear-sky values are returned when no cloud params given
        res = get_closest_lut_entry(
            self.lut, self.unique_values,
            doy=172, hour=12, albedo=0.2, altitude_km=0.5
        )
        self.assertEqual(res["direct_clear"], 800)
        self.assertEqual(res["diffuse_clear"], 100)
        self.assertIsNone(res["direct_cloudy"])
        self.assertIsNone(res["diffuse_cloudy"])  
        
        #Check that correct cloudy values are returned when cloud parameters are given.
        res = get_closest_lut_entry(
            self.lut, self.unique_values,
            doy=170, hour=11, albedo=0.17, altitude_km=0.5,
            cloud_top_km=5.0, cot=10.0, cloud_phase="water"
        )
        self.assertEqual(res["direct_clear"], 800)
        self.assertEqual(res["diffuse_clear"], 100)
        self.assertEqual(res["direct_cloudy"], 400)
        self.assertEqual(res["diffuse_cloudy"], 300)  
        
        # If no matching LUT entry exists, all values should be None.
        res = get_closest_lut_entry(
            self.lut, self.unique_values,
            doy=360, hour=2, albedo=0.9, altitude_km=0.5
        )
        self.assertTrue(all(v is None for v in res.values()))  
            

    def test_get_solar_angle_within_expected_range(self):
        """Compare get_solar_angle outputs to expected values within ±5 degrees."""
        timestamps = ["2015-09-03_10-00-00", "2025-09-03_15-00-00", "2025-06-21_12-00-00",
                    "2024-12-01_11-00-00", "2024-03-15_07-30-00", "2024-05-01_04-15-30"]
        # Convert to datetime64
        dt_array = np.array([
            np.datetime64(datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S"))
            for ts in timestamps
        ])
        altitudes_exp = [34.11, 25.0, 52.91, 7.62, 10.73, 3.85]
        zenith_exp = [90 - alt for alt in altitudes_exp]
        azimuth_exp = [150.19, 237.62, 187.41, 173.48, 113.52, 65.16]

        for dt, z_exp, a_exp in zip(dt_array, zenith_exp, azimuth_exp):
            zenith, azimuth = get_solar_angle(dt, lat=60.39, lon=5.33)
            self.assertIsInstance(zenith, float)
            self.assertIsInstance(azimuth, float)
            # Compare to expected within ±1 degrees
            self.assertAlmostEqual(zenith, z_exp, delta=1.0,
                                msg=f"Zenith mismatch for {dt}")
            self.assertAlmostEqual(azimuth, a_exp, delta=1.0,
                                msg=f"Azimuth mismatch for {dt}")
    
    def test_get_cloud_shadow_displacement(self):
        test_cases = [
            {
                "solar_zenith": 60,
                "solar_azimuth": 202,
                "sat_zenith": 7,
                "sat_azimuth": 108,
                "cbh_m": 1000,
                "expected_dx": -532.06,
                "expected_dy": 1643.87, 
                "cloud_pos": (49, 49),
                "expected_pos": (33, 44),  # manually calculated
            },
            {
                "solar_zenith": 50,
                "solar_azimuth": 174,
                "sat_zenith": 5,
                "sat_azimuth": 150,
                "cbh_m": 500,
                "expected_dx": 84.16,
                "expected_dy": 630.50, 
                "cloud_pos": (49, 49),
                "expected_pos": (43, 50),  # manually calculated
            },
            {
                "solar_zenith": 60,
                "solar_azimuth": 202,
                "cbh_m" : 1000, 
                "sat_zenith": 0,
                "sat_azimuth": 0,
                "expected_dx": -648.84, 
                "expected_dy": 1605.93,
                "cloud_pos": (49, 49),
                "expected_pos": (33, 43),  # manually calculated
            }
        ]

        for i, case in enumerate(test_cases, start=1):
            dx_pix, dy_pix = get_cloud_shadow_displacement(
                solar_zenith=case["solar_zenith"],
                solar_azimuth=case["solar_azimuth"],
                cbh_m=case["cbh_m"],
                sat_zenith=case["sat_zenith"],
                sat_azimuth=case["sat_azimuth"],
                cloud_top_height=None
            )
            self.assertAlmostEqual(dx_pix, case["expected_dx"], places=1)
            self.assertAlmostEqual(dy_pix, case["expected_dy"], places=1)

            
        
if __name__ == "__main__":
    unittest.main()
