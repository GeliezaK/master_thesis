import unittest
import numpy as np
from time import time
import os
from pyhdf.SD import SD, SDC  # HDF4 support

from src.preprocessing import *


class TestDownloadData(unittest.TestCase):
    def setUp(self):
        """Open the test file and extract datasets before running tests."""
        self.test_file_path = "../data/test_files/testfile_MOD35_L2.A2000055.1145.061.2017202184222.hdf"
        self.output_dir = "../data/test_files/output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Open test HDF file
        self.granule = SD(self.test_file_path, SDC.READ)

    def tearDown(self):
        """Close the test file and clean up output files."""
        self.granule.end()

        # Remove all files in output_dir
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def test_is_within_bbox(self):
        bbox = {"north": 70, "south": 60, "east": 6, "west": 5}
        latitudes1 = np.array([69.5, 71])
        longitudes1 = np.array([5.9, 7.5])
        self.assertTrue(is_within_bbox(latitudes1, longitudes1, bbox))

        latitudes2 = np.array([61, 65])
        longitudes2 = np.array([7, 8])
        self.assertFalse(is_within_bbox(latitudes2, longitudes2, bbox))

        latitudes3 = np.array([-10])
        longitudes3 = np.array([5.5])
        self.assertFalse(is_within_bbox(latitudes3, longitudes3, bbox))

        latitudes4 = np.array([62.5, 63.5, 72, 75])
        longitudes4 = np.array([4.4, 4.5, 5.5, 5.9])
        self.assertFalse(is_within_bbox(latitudes4, longitudes4, bbox))

        latitudes5 = np.array([60])
        longitudes5 = np.array([6])
        self.assertTrue(is_within_bbox(latitudes5, longitudes5, bbox))

        latitudes6 = np.array([60.5, 78])
        longitudes6 = np.array([5.5, 5.9])
        self.assertTrue(is_within_bbox(latitudes6, longitudes6, bbox))

        latitudes = np.array(self.granule.select("Latitude")[:])
        longitudes = np.array(self.granule.select("Longitude")[:])
        self.assertTrue(is_within_bbox(latitudes, longitudes, BBOX))

    def test_extract_roi_indices(self):
        print(BBOX)
        lats1 = np.array([[1,2,3], [4,5,6], [7,8,9]])
        lons1 = np.array([[5, 5, 5.5], [6, 6.5, 6], [5.5, 6.5, 6]])
        bbox = {"north": 6, "south": 2, "east": 6, "west": 4}
        exp_ind_x = np.array([0,0,1,1])
        exp_ind_y = np.array([1,2,0,2])
        out_ind_x, out_ind_y = extract_roi_indices(lats1, lons1, bbox)
        self.assertTrue((exp_ind_x == out_ind_x).all())
        self.assertTrue((exp_ind_y == out_ind_y).all())

        latitudes = np.array(self.granule.select("Latitude")[:])
        longitudes = np.array(self.granule.select("Longitude")[:])
        out_ind_x_1, out_ind_y_1 = extract_roi_indices(latitudes, longitudes, BBOX)
        print(f"Indices of Bergen ROI in test file: \n"
              f"out ind x : {out_ind_x_1}, (len: {len(out_ind_x_1)}) \n"
              f"out ind y : {out_ind_y_1} (len : {len(out_ind_y_1)})")



    def test_expand_indices_to_1km(self):
        ind_x_5km = np.array([0,0,1,1])
        ind_y_5km = np.array([1,2,0,2])
        orig_n = 3
        target_n = 15
        exp_y_1km = np.array([5,6,7,8,9,10,11,12,13,14,0,1,2,3,4,10,11,12,13,14])
        exp_x_1km = np.array([0,1,2,3,4,0,1,2,3,4,5,6,7,8,9,5,6,7,8,9])
        assert len(exp_x_1km) == len(exp_y_1km)
        out_x_1km = expand_indices_to_1km(ind_x_5km, orig_n, target_n)
        out_y_1km = expand_indices_to_1km(ind_y_5km, orig_n, target_n)
        self.assertTrue((exp_x_1km == out_x_1km).all())
        self.assertTrue((exp_y_1km == out_y_1km).all())

    def test_interpolate_to_1km_resolution(self):
        ind_x_5km = np.array([0, 0, 1, 1])
        ind_y_5km = np.array([1, 2, 0, 2])
        orig_shape = (3,3)
        target_shape = (15,16)
        exp_y_1km = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
        exp_x_1km = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        out_x_1km, out_y_1km = interpolate_to_1km_resolution(ind_x_5km, ind_y_5km, orig_shape, target_shape)
        self.assertTrue((exp_x_1km == out_x_1km).all())
        self.assertTrue((exp_y_1km == out_y_1km).all())


    def test_save_filtered_hdf(self):
        reduced_dataset = {"latitude": np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.int16),
                           "longitude": np.array([[5,5,5], [6,6,6], [7,7,7]], dtype=np.int16)}
        filename = "test_save_filtered_hdf.hdf"
        attr_dict = {}
        save_filtered_hdf(filename, reduced_dataset, attr_dict, self.output_dir)
        exp_filepath = self.output_dir + "/filtered_" + filename

        # Open HDF file
        granule = SD(exp_filepath, SDC.READ)
        latitude = np.array(granule.select("latitude")[:])
        longitude = np.array(granule.select("longitude")[:])
        self.assertTrue((latitude == reduced_dataset["latitude"]).all())
        self.assertTrue((longitude == reduced_dataset["longitude"]).all())



    def test_filter_roi(self):
        start = time()
        filter_roi(os.path.dirname(self.test_file_path), self.output_dir)
        end = time()
        duration = end-start
        print(f"Filtering 4 granules covering 1 day took {np.round(duration, 5)} seconds.")
        exp_total_duration = duration * 3.5 * 365 * 5
        print(f"Expected duration for 5 years is {np.round(exp_total_duration/360)} hours. ")
        exp_filepath = self.output_dir + "/filtered_" + os.path.basename(self.test_file_path)
        orig_file_size = os.path.getsize(self.test_file_path)
        reduced_file_size = os.path.getsize(exp_filepath)
        self.assertTrue(reduced_file_size < orig_file_size)
        print(f"Filtering reduced the file size from {orig_file_size} to {reduced_file_size} "
              f"({np.round(reduced_file_size/orig_file_size,5)}%)!")

        # Open HDF file
        reduced_granule = SD(exp_filepath, SDC.READ)
        reduced_latitudes = np.array(reduced_granule.select("Latitude")[:])
        reduced_longitudes = np.array(reduced_granule.select(("Longitude")[:]))
        # Assert no datapoints outside of roi
        self.assertTrue(self.all_in_roi(reduced_latitudes, reduced_longitudes, BBOX))
        # Assert contains each dataset from original granule
        self.assertTrue(self.granule.datasets().keys() == reduced_granule.datasets().keys())


    def test_how_many_in_roi(self):
        """Print how many datapoints are in ROI bbox."""
        latitudes = np.array(self.granule.select("Latitude")[:])
        longitudes = np.array(self.granule.select("Longitude")[:])
        count = 0
        for lat, lon in zip(latitudes.flat, longitudes.flat):
            if BBOX["south"] <= lat <= BBOX["north"] and BBOX["west"] <= lon <= BBOX["east"]:
                count += 1
        n_datapoints = np.size(latitudes,0) * np.size(latitudes,1)
        print(f"There are {count} out of {n_datapoints} datapoints within the Bergen ROI ({np.round(count/n_datapoints,3)} %)." )

    def all_in_roi(self, latitudes, longitudes, bbox):
        """Check that all points in latitudes/longitudes are within bbox."""
        for lat, lon in zip(latitudes.flat, longitudes.flat):
            if not (bbox["south"] <= lat <= bbox["north"] and bbox["west"] <= lon <= bbox["east"]):
                return False
        return True




