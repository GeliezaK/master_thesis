import unittest
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import rioxarray

from pathlib import Path
import sys
from datetime import datetime
from netCDF4 import Dataset
import os
import shutil
#sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from src.model.surface_GHI_model import load_image, get_solar_angle, get_cloud_shadow_displacement
from src.model.surface_GHI_model import calculate_sw_dir_cor, project_cloud_shadow, BBOX
from src.model.surface_GHI_model import get_closest_lut_entry

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
            {"DOY": 172, "Hour": 12, "Albedo": 0.2, "Altitude_km": 0.5,
             "Direct_clear": 800, "Diffuse_clear": 100,
             "Direct_cloudy": 400, "Diffuse_cloudy": 300,
             "CloudTop_km": 5.0, "Tau550": 10.0, "CloudType": "water"},
            {"DOY": 166, "Hour": 12, "Albedo": 0.2, "Altitude_km": 0.5,
             "Direct_clear": 550, "Diffuse_clear": 30,
             "Direct_cloudy": 0.1, "Diffuse_cloudy": 350,
             "CloudTop_km": 3.0, "Tau550": 100.0, "CloudType": "water"},
            {"DOY": 74, "Hour": 9, "Albedo": 0.2, "Altitude_km": 0.5,
             "Direct_clear": 600, "Diffuse_clear": 150,
             "Direct_cloudy": 200, "Diffuse_cloudy": 400,
             "CloudTop_km": 8.0, "Tau550": 20.0, "CloudType": "ice"}
        ])

        self.unique_values = {
            "DOY": self.lut["DOY"].unique(),
            "Hour": self.lut["Hour"].unique(),
            "Albedo": self.lut["Albedo"].unique(),
            "Altitude_km": self.lut["Altitude_km"].unique(),
            "CloudTop_km": self.lut["CloudTop_km"].unique(),
            "Tau550": self.lut["Tau550"].unique(),
            "CloudType": self.lut["CloudType"].unique(),
        }
        

    def tearDown(self):
        # Remove created files and test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_resample_alignment(self):
        """Test resampling of sw_dir_cor (NetCDF) to cloud mask (GeoTIFF) grid."""

        # --- Load NetCDF ---
        nc_file = Path("data/processed/sw_cor/sw_cor_bergen.nc")
        ds = xr.open_dataset(nc_file)
        # take first timestep
        sw_dir_cor = ds["sw_dir_cor"].isel(time=0)

        # --- Load GeoTIFF ---
        tif_file = Path("data/raw/S2_cloud_mask_large/S2_cloud_mask_large_2015-08-08_11-00-36-2015-08-08_11-01-17.tif")
        with rasterio.open(tif_file) as src:
            shadow_mask = src.read(1)
            shadow_meta = src.profile
            shadow_bounds = src.bounds
            shadow_transform = src.transform

        # --- Resample ---
        sw_dir_cor_resampled = resample(sw_dir_cor, shadow_meta)

        # --- Check shape ---
        self.assertEqual(
            sw_dir_cor_resampled.shape,
            shadow_mask.shape,
            msg="Shapes after resampling should match exactly"
        )

        # --- Check extent ---
        # rioxarray stores bounds in .rio.bounds()
        nc_bounds = sw_dir_cor_resampled.rio.bounds()
        self.assertAlmostEqual(nc_bounds[0], shadow_bounds.left, places=6, msg="West bound mismatch")
        self.assertAlmostEqual(nc_bounds[1], shadow_bounds.bottom, places=6, msg="South bound mismatch")
        self.assertAlmostEqual(nc_bounds[2], shadow_bounds.right, places=6, msg="East bound mismatch")
        self.assertAlmostEqual(nc_bounds[3], shadow_bounds.top, places=6, msg="North bound mismatch")

        # --- Check center ---
        shadow_center = ((shadow_bounds.left + shadow_bounds.right) / 2,
                         (shadow_bounds.bottom + shadow_bounds.top) / 2)
        nc_center = ((nc_bounds[0] + nc_bounds[2]) / 2,
                     (nc_bounds[1] + nc_bounds[3]) / 2)
        self.assertAlmostEqual(shadow_center[0], nc_center[0], places=6, msg="Center longitude mismatch")
        self.assertAlmostEqual(shadow_center[1], nc_center[1], places=6, msg="Center latitude mismatch")

        # --- Optional: check max lat/lon values ---
        self.assertAlmostEqual(sw_dir_cor_resampled.rio.bounds()[2], shadow_bounds.right, places=6)
        self.assertAlmostEqual(sw_dir_cor_resampled.rio.bounds()[3], shadow_bounds.top, places=6)
            
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
            doy=160, hour=2, albedo=0.9, altitude_km=10.0
        )
        self.assertTrue(all(v is None for v in res.values()))  
            
            
    def test_calculate_sw_dir_cor(self):
        out_file = calculate_sw_dir_cor(
            dsm_filepath=self.dsm_filepath,
            times=self.times,
            out_dir=self.test_dir
        ) #TODO: fix datatype error 

        # Check that output file exists
        self.assertTrue(os.path.exists(out_file), f"Output file not found: {out_file}")

        # Open NetCDF and check values
        # Open NetCDF and check contents
        with Dataset(out_file, "r") as nc:
            # Check dimensions exist
            self.assertIn("time", nc.dimensions)
            self.assertIn("sun_x", nc.dimensions)
            self.assertIn("sun_y", nc.dimensions)
            self.assertIn("sun_z", nc.dimensions)

            # Check correct sizes
            self.assertEqual(len(nc.dimensions["time"]), len(self.times))
            self.assertEqual(len(nc.dimensions["sun_x"]), 1)
            self.assertEqual(len(nc.dimensions["sun_y"]), 1)
            self.assertEqual(len(nc.dimensions["sun_z"]), 1)

            # Check data values
            data = nc.variables["sw_dir_cor"][:]
            self.assertTrue(np.all(data >= 0), "All values must be >= 0")
            self.assertTrue(np.all(data <= 6), "All values must be <= 6")
            

    def test_load_image_returns_array_and_profile(self):
        # Save synthetic DEM to a temp file
        import tempfile
        import rasterio
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            with rasterio.open(
                tmp.name, 'w',
                driver='GTiff',
                height=self.dem_test.shape[0],
                width=self.dem_test.shape[1],
                count=1,
                dtype=self.dem_test.dtype
            ) as dst:
                dst.write(self.dem_test, 1)
            
            dem, profile = load_image(tmp.name)
            self.assertEqual(dem.shape, self.dem_test.shape)
            self.assertIn('transform', profile)

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
                "pixel_size": 100,
                "expected_dx": -5.3,
                "expected_dy": 16.4, 
                "cloud_pos": (49, 49),
                "expected_pos": (33, 44),  # manually calculated
            },
            {
                "solar_zenith": 50,
                "solar_azimuth": 174,
                "sat_zenith": 5,
                "sat_azimuth": 150,
                "cbh_m": 500,
                "pixel_size": 100,
                "expected_dx": 0.84,
                "expected_dy": 6.3, 
                "cloud_pos": (49, 49),
                "expected_pos": (43, 50),  # manually calculated
            },
            {
                "solar_zenith": 60,
                "solar_azimuth": 202,
                "sat_zenith": 0,
                "sat_azimuth": 0,
                "cbh_m": 1000,
                "pixel_size": 100,
                "expected_dx": -6.5, 
                "expected_dy": 16.1,
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
                pixel_size=case["pixel_size"],
                cloud_top_height=None
            )
            self.assertAlmostEqual(dx_pix, case["expected_dx"], places=1)
            self.assertAlmostEqual(dy_pix, case["expected_dy"], places=1)

        
    def test_project_cloud_shadow(self):
        # Create fake raster with one cloud pixel
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[50, 50] = 1

        transform = rasterio.Affine(10,0, 0, 0, -10, 1000)  # arbitrary grid
        crs = "EPSG:4326"

        # Save to temporary GeoTIFF
        filepath = "test_cloud.tif"
        with rasterio.open(
            filepath, "w", driver="GTiff",
            height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype,
            crs=crs, transform=transform
        ) as dst:
            dst.write(arr, 1)

        # Displace by 5 px right, 3 px down
        dx_pix, dy_pix = 15, 30
        bbox = {"west": 200, "south": 100, "east": 900, "north": 1000}

        roi_mask = project_cloud_shadow(filepath, dy_pix, dx_pix, bbox)

        # Expected mask: cloud pixel at (65, 80)
        expected = np.zeros_like(arr)
        expected[50-dy_pix, 50+dx_pix] = 1
        ymax = 1000  
        pixel_size = 10

        idx_min = int(bbox["west"]/pixel_size)
        idx_max = int(bbox["east"]/pixel_size)

        idy_min = int((ymax - bbox["north"]) / pixel_size)
        idy_max = int((ymax - bbox["south"]) / pixel_size)

        print(f"xrange:{idx_min}:{idx_max}, yrange:{idy_min}:{idy_max}")
        expected = expected[idy_min:idy_max, idx_min:idx_max]

        try:
            np.testing.assert_array_equal(roi_mask, expected)
            print("✅ Test passed")
        except AssertionError:
            exp_idx = np.argwhere(expected == 1)
            got_idx = np.argwhere(roi_mask == 1)

            print("❌ Test failed")
            print("Expected pixel(s) at:", exp_idx)
            print("Got pixel(s) at:     ", got_idx)
            
            raise  # re-raise so pytest still fails

        os.remove(filepath)
            
        
if __name__ == "__main__":
    unittest.main()
