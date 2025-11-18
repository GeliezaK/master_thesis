import numpy as np
import unittest
from netCDF4 import Dataset, date2num
from datetime import datetime
import tempfile
import os
from src.model.upscale_cloud_masks import count_misclassifications   # adjust import


def create_test_nc(path, lat, lon, masks, times):
    """
    Helper to create small synthetic NetCDF shadow-mask files.
    lat, lon = 1D arrays
    masks = list of 2D arrays (one per timestep)
    times = list of python datetime objects
    """

    nc = Dataset(path, "w")
    nc.createDimension("time", None)
    nc.createDimension("lat", len(lat))
    nc.createDimension("lon", len(lon))

    vlat = nc.createVariable("lat", "f4", ("lat",))
    vlat[:] = lat
    vlon = nc.createVariable("lon", "f4", ("lon",))
    vlon[:] = lon

    vtime = nc.createVariable("time", "f8", ("time",))
    vtime.units = "hours since 2015-01-01 00:00:00"
    vtime.calendar = "gregorian"

    vmask = nc.createVariable("shadow_mask", "i1", ("time", "lat", "lon"))

    for i, (m, t) in enumerate(zip(masks, times)):
        vtime[i] = date2num(t, units=vtime.units)
        vmask[i, :, :] = m

    nc.close()


class TestMisclassificationNearestNeighbor(unittest.TestCase):

    def test_misclassification_nn_mapping(self):

        # Fine grid (4×4)
        lat_f = np.linspace(60, 60.03, 4)
        lon_f = np.linspace(5, 5.03, 4)

        # Coarse grid (3×3) — not divisible
        lat_c = np.linspace(60, 60.03, 3)
        lon_c = np.linspace(5, 5.03, 3)

        # Two timestamps
        times = [
            datetime(2016, 1, 1, 10),
            datetime(2016, 1, 1, 11),
        ]

        fine_masks = [
            np.zeros((4, 4), dtype=np.int8),
            np.ones((4, 4), dtype=np.int8),
        ]

        coarse_masks = [
            np.ones((3, 3), dtype=np.int8),   # all mismatched
            np.ones((3, 3), dtype=np.int8),   # perfect match
        ]

        with tempfile.TemporaryDirectory() as td:
            fine_path = os.path.join(td, "fine.nc")
            coarse_path = os.path.join(td, "coarse.nc")

            create_test_nc(fine_path, lat_f, lon_f, fine_masks, times)
            create_test_nc(coarse_path, lat_c, lon_c, coarse_masks, times)

            df = count_misclassifications(fine_path, coarse_path, "test_resolution")

            # Two timestamps expected
            self.assertEqual(len(df), 2)

            # First timestep: mismatched count = 16 (4×4)
            self.assertEqual(df.iloc[0].misclassified_count, 16)
            self.assertAlmostEqual(df.iloc[0].misclassified_percentage, 100.0)

            # Second timestep: perfect match
            self.assertEqual(df.iloc[1].misclassified_count, 0)
            self.assertAlmostEqual(df.iloc[1].misclassified_percentage, 0.0)


if __name__ == "__main__":
    unittest.main()