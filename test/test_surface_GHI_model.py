import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from surface_GHI_model import load_image, get_hillshade_band, get_solar_angle, get_cloud_shadow_displacement


class TestSurfaceGHIModel(unittest.TestCase):
    
    def setUp(self):
        # Create a small synthetic DEM (10x10) for testing
        self.dem_test = np.linspace(0, 100, 100).reshape((10, 10))
        self.profile_test = {
            'transform': (10, 0, 0, 0, -10, 0),  # pixel size 10x10
            'crs': None,
            'width': 10,
            'height': 10
        }

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
        altitudes_exp = [34.11, 25.0, 52.91, 7.62, 10.73, 3.85]
        zenith_exp = [90 - alt for alt in altitudes_exp]
        azimuth_exp = [150.19, 237.62, 187.41, 173.48, 113.52, 65.16]

        for ts_str, z_exp, a_exp in zip(timestamps, zenith_exp, azimuth_exp):
            zenith, azimuth = get_solar_angle(ts_str, lat=60.39, lon=5.33)
            self.assertIsInstance(zenith, float)
            self.assertIsInstance(azimuth, float)
            # Compare to expected within ±1 degrees
            self.assertAlmostEqual(zenith, z_exp, delta=1.0,
                                msg=f"Zenith mismatch for {ts_str}")
            self.assertAlmostEqual(azimuth, a_exp, delta=1.0,
                                msg=f"Azimuth mismatch for {ts_str}")


    def test_get_hillshade_band_output(self):
        zenith = 45
        azimuth = 180
        hs = get_hillshade_band(self.dem_test, zenith, azimuth, self.profile_test)
        self.assertEqual(hs.shape, self.dem_test.shape)
        self.assertTrue(np.all(hs >= 0) and np.all(hs <= 1))
        
    def test_project_cloud_shadow(self):
        test_cases = [
            {
                "solar_zenith": 60,
                "solar_azimuth": 202,
                "sat_zenith": 7,
                "sat_azimuth": 108,
                "cbh_m": 1000,
                "pixel_size": 100,
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
                "cloud_pos": (49, 49),
                "expected_pos": (33, 43),  # manually calculated
            }
        ]

        for i, case in enumerate(test_cases, start=1):
            print(f"\n--- Running shadow projection test case {i} ---")
            cloud_mask = np.zeros((100, 100), dtype=np.uint8)
            cloud_mask[case["cloud_pos"]] = 1

            shadow_exp = np.zeros((100, 100), dtype=np.uint8)
            shadow_exp[case["expected_pos"]] = 1

            shadow_res = get_cloud_shadow_displacement(
                cloud_mask,
                case["solar_zenith"],
                case["solar_azimuth"],
                case["cbh_m"],
                case["sat_zenith"],
                case["sat_azimuth"],
                pixel_size=case["pixel_size"],
                cloud_top_height=case["cbh_m"],
            )

            try:
                np.testing.assert_array_equal(shadow_res, shadow_exp)
                print(f"Test case {i} PASSED")
            except AssertionError:
                diff = shadow_res - shadow_exp
                diff_pixels = np.argwhere(diff != 0)
                print(f"Test case {i} FAILED")
                print("Expected pixel at:", np.argwhere(shadow_exp == 1))
                print("Got pixel at:     ", np.argwhere(shadow_res == 1))
                print("Difference pixels:", diff_pixels)
                raise         
        
if __name__ == "__main__":
    unittest.main()
