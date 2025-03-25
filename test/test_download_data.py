import unittest
import numpy as np
from src.download_data import is_within_bbox

class TestDownloadData(unittest.TestCase):
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

