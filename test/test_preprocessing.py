import unittest
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
from datetime import datetime, timezone
import netCDF4 as nc
from cftime import DatetimeGregorian
import pandas as pd
from src.preprocessing.preprocess_netcdf_data import add_claas3_variable_to_cloud_cover_table, get_claas3_filepath, compute_roi_stats
from src.preprocessing.merge_station_obs_with_sim import add_simulated_florida_flesland_ghi_by_date, extract_pixel_by_location


class TestGetClaas3Filepath(unittest.TestCase):

    def setUp(self):
        """Create temporary CLAAS-3 like folder structure with sample .nc files"""
        self.base_dir = Path("test_data_claas3")
        self.year = "2021"
        self.month = "07"
        self.day = "15"
        self.file_prefix = "CPPin"

        # Build nested directory structure
        self.target_dir = self.base_dir / self.year / self.month / self.day
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Create two test files with simple NetCDF structure
        self.file1 = self.target_dir / f"{self.file_prefix}{self.year}{self.month}{self.day}1030_example.nc"
        self.file2 = self.target_dir / f"{self.file_prefix}{self.year}{self.month}{self.day}1045_example.nc"

        self.data = np.array([
            [1, 2, np.nan],
            [-5, 3, np.inf],
            [0, 10, -1]
        ], dtype=float)

        for f in [self.file1, self.file2]:
            with nc.Dataset(f, "w", format="NETCDF4") as ds:
                ds.createDimension("time", 1)
                ds.createDimension("x", self.data.shape[0])
                ds.createDimension("y", self.data.shape[1])
                var = ds.createVariable("test", "f4", ("time", "x", "y"))
                var[0, :, :] = self.data
                
        # Load file 1 via xarray
        self.ds = xr.open_dataset(self.file1)
        self.var = self.ds["test"]
        
        # Aux file with lat/lon grid
        self.aux_path = self.base_dir / "aux.nc"
        lat = np.array([
            [60.3, 60.4, 60.5],
            [60.3, 60.5, 70],
            [60, 60.5, 70],
        ])
        lon = np.array([
            [5.2, 5.3, 5.5],
            [5, 5.3, 5.5],
            [5, 5.7, 6],
        ])
        with nc.Dataset(self.aux_path, "w", format="NETCDF4") as ds:
            ds.createDimension("x", lat.shape[0])
            ds.createDimension("y", lat.shape[1])
            ds.createDimension("georef_offset_corrected", 1)
            var_lat = ds.createVariable("lat", "f4", ("georef_offset_corrected", "x", "y"))
            var_lon = ds.createVariable("lon", "f4", ("georef_offset_corrected", "x", "y"))
            var_lat[0, :, :] = lat
            var_lon[0, :, :] = lon

        # Cloud cover CSV with 3 rows
        # Two matching the CLAAS files, one too late (warn case)
        ts1 = pd.Timestamp(f"{self.year}-{self.month}-{self.day} 10:58:00", tz="UTC")
        ts2 = pd.Timestamp(f"{self.year}-{self.month}-{self.day} 10:46:00", tz="UTC")
        ts3 = pd.Timestamp(f"2024-08-01 11:10:00", tz="UTC")

        self.cloud_cover_path = self.base_dir / "cloud_cover.csv"
        df = pd.DataFrame({
            "system:time_start_large": [
                int(ts1.value / 1e6),
                int(ts2.value / 1e6),
                int(ts3.value / 1e6),
            ],
            "date": [ts1.date(), ts2.date(), ts3.date()],
            "cloud_cover_large": [0.5, 0.6, 0.7],
            "cloud_cover_small": [0.4, 0.5, 0.6],
        })
        df.to_csv(self.cloud_cover_path, index=False)
 

    def tearDown(self):
        """Remove all created test data"""
        self.ds.close()
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
            
    def test_extract_pixel_by_location(self):
        """Test that the correct pixel is extracted for Florida and Flesland stations"""

        # Create a small test NetCDF with lat/lon and GHI_total
        model_nc_path = self.base_dir / "model_ghi.nc"
        lats = np.array([60.0, 60.3, 60.5], dtype=np.float32)
        lons = np.array([5.0, 5.3, 5.5], dtype=np.float32)

        with nc.Dataset(model_nc_path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("lat", len(lats))
            ds.createDimension("lon", len(lons))

            var_lat = ds.createVariable("lat", "f4", ("lat",))
            var_lon = ds.createVariable("lon", "f4", ("lon",))
            var_time = ds.createVariable("time", "f8", ("time",))
            var_time.units = "hours since 2021-07-15 00:00:00"
            var_time.calendar = "gregorian"

            var_lat[:] = lats
            var_lon[:] = lons
            var_time[:] = [10, 11]  # 10h and 11h UTC

            ghi = ds.createVariable("GHI_total", "f4", ("time", "lat", "lon"))
            # Shape: (time=2, lat=3, lon=3)
            ghi[0, :, :] = np.array([
                [100, 200, 300],
                [400, 500, 600],
                [700, 800, 900],
            ])
            ghi[1, :, :] = np.array([
                [150, 250, 350],
                [450, 550, 650],
                [750, 850, 950],
            ])

        # Florida and Flesland coordinates
        case_1_lat, case_1_lon = 59.5, 5.6 # outside of grid 
        case_2_lat, case_2_lon = 60.4, 5.25 # inside grid
        flesland_lat, flesland_lon = 60.292792, 5.222689

        # Extract pixels
        times, case_1_vals = extract_pixel_by_location(model_nc_path, case_1_lat, case_1_lon)
        times, case_2_vals = extract_pixel_by_location(model_nc_path, case_2_lat, case_2_lon)
        _, flesland_vals = extract_pixel_by_location(model_nc_path, flesland_lat, flesland_lon)

        # Case 1 should map to nearest pixels [lat=0, lon=2] --> [300,350]
        np.testing.assert_array_equal(case_1_vals, [300, 350])

        # Case 2 should map to nearest pixels [lat=2, lon=1] --> [800,850]
        np.testing.assert_array_equal(case_2_vals, [800, 850])

        # Flesland also maps to nearest [lat=1, lon=1] --> [500, 550]
        np.testing.assert_array_equal(flesland_vals, [500, 550])

        # Times should be 2 entries (10h, 11h UTC)
        self.assertEqual(len(times), 2)
        
    def test_add_simulated_florida_flesland_ghi_by_date(self):
        """Test merging simulated Florida/Flesland GHI by date with duplicates and missing dates."""

        # Create a df with simulated GHI values
        ts_obs = pd.to_datetime([f"{self.year}-{self.month}-{self.day} 10:58:00",
                                f"{self.year}-{self.month}-{self.day} 10:46:00",
                                "2021-07-16 11:00:00"], utc=True)
        
        # Convert to Gregorian because that's how they are saved in ghi.nc file
        ts_cftime = [DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in ts_obs]
        
        df_sim = pd.DataFrame({
            "time": ts_cftime + ts_cftime,  # duplicate rows
            "Florida_ghi_sim": [100, 200, 300, 100, 200, 300],
            "Flesland_ghi_sim": [10, 20, 30, 10, 20, 30]
        })
        

        # Output path
        out_csv = self.base_dir / "merged_obs.csv"

        # Call function to merge
        merged = add_simulated_florida_flesland_ghi_by_date(self.cloud_cover_path, df_sim, out_csv)

        # Check that merged has new columns
        self.assertIn("Florida_ghi_sim", merged.columns)
        self.assertIn("Flesland_ghi_sim", merged.columns)

        # Load original obs dates
        obs_dates = pd.read_csv(self.cloud_cover_path)["date"].apply(pd.to_datetime).dt.date

        # Check that for missing date (2024-08-01) values are NaN
        missing_idx = merged["date"] == pd.to_datetime("2024-08-01").date()
        self.assertTrue(merged.loc[missing_idx, "Florida_ghi_sim"].isna().all())
        self.assertTrue(merged.loc[missing_idx, "Flesland_ghi_sim"].isna().all())

        # Check that for the other two dates, correct values are merged (should match first occurrence in df_sim)
        for date, florida_val, flesland_val in zip(obs_dates[:-1], [100, 100], [10, 10]):
            row = merged[merged["date"] == date]
            self.assertEqual(row["Florida_ghi_sim"].values[0], florida_val)
            self.assertEqual(row["Flesland_ghi_sim"].values[0], flesland_val)

        # Check that CSV file was created
        self.assertTrue(out_csv.exists())


    def test_closest_file_selection(self):
        # Define candidate times in minutes after midnight
        possible_times = [630, 645, 660]  # 10:30, 10:45, 11:00

        # Case 1: dt closer to 10:30 file
        dt1 = datetime(2021, 7, 15, 10, 32, tzinfo=timezone.utc)
        result1 = get_claas3_filepath(self.base_dir, dt1, self.file_prefix, possible_times)
        self.assertIsNotNone(result1)
        self.assertTrue(str(result1).endswith("1030_example.nc"))

        # Case 2: dt closer to 10:45 file
        dt2 = datetime(2021, 7, 15, 10, 44, tzinfo=timezone.utc)
        result2 = get_claas3_filepath(self.base_dir, dt2, self.file_prefix, possible_times)
        self.assertIsNotNone(result2)
        self.assertTrue(str(result2).endswith("1045_example.nc"))

        # Case 3: exact match
        dt3 = datetime(2021, 7, 15, 10, 30, tzinfo=timezone.utc)
        result3 = get_claas3_filepath(self.base_dir, dt3, self.file_prefix, possible_times)
        self.assertTrue(str(result3).endswith("1030_example.nc"))
        
        # Case 4: no file found (different date)
        dt4 = datetime(2021, 7, 16, 10, 30, tzinfo=timezone.utc)  # Next day, no folder created
        result4 = get_claas3_filepath(self.base_dir, dt4, self.file_prefix, possible_times)
        self.assertIsNone(result4)
    
    def test_stats_without_mask(self):
        """Test compute_roi_stats with full array"""
        result = compute_roi_stats(self.var, variable_name="test", suffix="_test")

        # Only positive finite values: [1, 2, 3, 10]
        expected = {
            "test_min_test": 1.0,
            "test_max_test": 10.0,
            "test_mean_test": (1+2+3+10)/4,  # 4.0
            "test_median_test": (2+3)/2,     # 2.5
            "test_std_test": np.std([1,2,3,10])
        }

        for key, val in expected.items():
            self.assertAlmostEqual(result[key], val, places=6)
            

    def test_stats_with_mask(self):
        """Test compute_roi_stats with mask selecting subset"""
        # Mask only selects values at (0,0) and (1,1): [1, 3]
        mask_array = np.array([
            [True, False, False],
            [False, True, False],
            [False, False, False]
        ])

        # Expand mask to include time dimension (length 1)
        mask_array = mask_array[np.newaxis, :, :]  # shape (1, 3, 3)

        mask = xr.DataArray(
            mask_array,
            dims=self.var.dims,           # ("time", "x", "y")
            coords=self.var.coords
        )

        result = compute_roi_stats(self.var, mask=mask, variable_name="test", suffix="_mask")

        expected = {
            "test_min_mask": 1.0,
            "test_max_mask": 3.0,
            "test_mean_mask": 2.0,
            "test_median_mask": 2.0,
            "test_std_mask": np.std([1, 3])
        }

        for key, val in expected.items():
            self.assertAlmostEqual(result[key], val, places=6)
            
            
    def test_add_claas3_variable_to_cloud_cover_table(self):
        # Run function
        add_claas3_variable_to_cloud_cover_table(
            claas_folderpath=self.base_dir,
            aux_path=self.aux_path,
            cloud_cover_path=str(self.cloud_cover_path),
            variable_name="test",
            file_prefix=self.file_prefix
        )

        out_path = str(self.cloud_cover_path).replace(".csv", "_with_test.csv")
        self.assertTrue(Path(out_path).exists(), "Output CSV was not created")

        df_out = pd.read_csv(out_path)

        # Expected columns exist
        expected_cols = [
            f"test_{stat}{suf}"
            for suf in ["_large", "_small"]
            for stat in ["min", "max", "mean", "median", "std"]
        ]
        for col in expected_cols:
            self.assertIn(col, df_out.columns)

        # Check stats row 2 (closest to 10:30 file)
        row2 = df_out.iloc[1]
        # Large ROI: [1,2,3,10]
        self.assertAlmostEqual(row2["test_min_large"], 1.0, places=6)
        self.assertAlmostEqual(row2["test_max_large"], 10.0, places=6)
        self.assertAlmostEqual(row2["test_mean_large"], 4.0, places=6)
        self.assertAlmostEqual(row2["test_median_large"], 2.5, places=6)

        # Small ROI: depends on BBOX, mask selects lat/lon in BBOX
        # In our aux grid, that corresponds to center block (values [1,2,3, nan])
        self.assertAlmostEqual(row2["test_min_small"], 1.0, places=6)
        self.assertAlmostEqual(row2["test_max_small"], 3.0, places=6)
        self.assertAlmostEqual(row2["test_mean_small"], 2.0, places=6)
        self.assertAlmostEqual(row2["test_median_small"], 2.0, places=6)
    
        # Row 1 should be all NaN because no file matches
        row1 = df_out.iloc[0]
        for col in expected_cols:
            self.assertTrue(np.isnan(row1[col]), f"{col} should be NaN for row3")

        # Row 3 should be all NaN because no file matches
        row3 = df_out.iloc[2]
        for col in expected_cols:
            self.assertTrue(np.isnan(row3[col]), f"{col} should be NaN for row3")
            



if __name__ == "__main__":
    unittest.main()