from pyhdf.SD import SD, SDC  # HDF4 support
import os
import numpy as np
import math


def deg2rad(deg):
    return deg * math.pi / 180

# Define the center of the bounding box (Bergen, Norway)
CENTER_LAT = 60.39
CENTER_LON = 5.33

# Approximate degree adjustments for 100km x 100km box
DEG_LAT_TO_KM = 111.412  # 1 degree latitude at 60° converted to km (https://en.wikipedia.org/wiki/Latitude)
DEG_LON_TO_KM = 111.317 * math.cos(deg2rad(CENTER_LAT))  # 1 degree longitude converted to km
LAT_OFFSET = 50 / DEG_LAT_TO_KM  # ~50km north/south, ~ 0.45
LON_OFFSET = 50 / DEG_LON_TO_KM  # ~50km east/west (varies with latitude, approximation), ~ 0.9

# Define the bounding box
BBOX = {
    "north": CENTER_LAT + LAT_OFFSET,
    "south": CENTER_LAT - LAT_OFFSET,
    "west": CENTER_LON - LON_OFFSET,
    "east": CENTER_LON + LON_OFFSET
}


# interpolation: if Bergen area at left or right edge of dataset (i.e. column 269 or 0 included), pad to the left or
# right of the 1km resolution dataset, else discard the edges left and right

def is_within_bbox(latitudes, longitudes, bbox):
    """Check if any latitude/longitude points fall within the bounding box."""
    for lat, lon in zip(latitudes.flat, longitudes.flat):
        if bbox["south"] <= lat <= bbox["north"] and bbox["west"] <= lon <= bbox["east"]:
            return True
    return False

def extract_roi_indices(latitudes, longitudes, bbox):
    """Returns indices in lat and lon np arrays where geolocation is within bbox, assuming lat and lon have equal
    shape."""
    assert np.shape(latitudes) == np.shape(longitudes)
    mask = ((bbox["south"] <= latitudes) & (latitudes <= bbox["north"]) &
            (bbox["west"] <= longitudes) & (longitudes <= bbox["east"]))
    ind_x, ind_y = mask.nonzero()
    return ind_x, ind_y


def interpolate_to_1km_resolution(ind_x_5km, ind_y_5km, orig_shape, target_shape):
    """Convert indices from 5kmx5km resolution to target resolution (1km x 1km). Simply cut off last indices."""
    orig_n, orig_m = orig_shape
    target_n, target_m = target_shape


    # Rows
    remainder_x = target_n % orig_n
    ind_x_1km = expand_indices_to_1km(ind_x_5km, orig_n, target_n - remainder_x)

    # Columns
    remainder_y = target_m % orig_m
    ind_y_1km = expand_indices_to_1km(ind_y_5km, orig_m, target_m - remainder_y)

    return ind_x_1km, ind_y_1km


def expand_indices_to_1km(ind_5km, orig_n, target_n):
    """Expand 5km resolution indices to 1km resolution indices."""
    assert target_n % orig_n == 0, "Target size is not multiple of original size."
    divisor = target_n // orig_n
    ind_1km = np.hstack([np.arange(i * divisor, (i + 1) * divisor) for i in ind_5km])
    return ind_1km


def save_filtered_hdf(filename, dataset_dict, attr_dict, output_dir):
    """Save reduced datasets in a new HDF file with the correct data types."""
    new_filename = os.path.join(output_dir, "filtered_" + filename)
    new_hdf = SD(new_filename, SDC.WRITE | SDC.CREATE)

    for dataset_name, data in dataset_dict.items():
        # Map NumPy dtypes to HDF formats
        if data.dtype == np.float32 :
            hdf_dtype = SDC.FLOAT32
        elif data.dtype == np.float64:
            hdf_dtype = SDC.FLOAT64
        elif data.dtype == np.int8:
            hdf_dtype = SDC.INT8
        elif data.dtype == np.int16:
            hdf_dtype = SDC.INT16
        else:
            raise ValueError(f"The type of the dataset {dataset_name}, {data.dtype}, was not recognized.")

        # Create dataset with the correct type
        sds = new_hdf.create(dataset_name, hdf_dtype, data.shape)
        sds[:] = data

        # Copy attributes
        #if dataset_name in attr_dict:
        #    for attr_name, attr_value in attr_dict[dataset_name]:
        #        print(f"Attr name: {attr_name}, attr_value: {attr_value}")
        #        attr_type = # Todo: get type of attribute and set in this function
        #        sds.attr(attr_name).set(attr_type, attr_value)

        sds.endaccess()

    new_hdf.end()
    print(f"Saved filtered HDF: {new_filename}")


def filter_roi(path, outdir):
    """Loop over all files in path and delete all files that do not contain the Bergen region of interest. """

    # Loop over all .hdf files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".hdf"):
            filepath = os.path.join(path, filename)

            # Open HDF file
            granule = SD(filepath, SDC.READ)

            # Extract Latitude and Longitude
            latitudes = np.array(granule.select("Latitude")[:])
            longitudes = np.array(granule.select("Longitude")[:])

            if not is_within_bbox(latitudes, longitudes, BBOX):
                # Close file
                granule.end()
                os.remove(filepath)  # Delete file if outside Bergen
                print(f"Deleted: {filepath} (Outside Bergen area)")
                continue

            ind_x_5km, ind_y_5km = extract_roi_indices(latitudes, longitudes, BBOX)
            n_indices = len(ind_x_5km)
            n_lat_rows = np.size(latitudes, 0)
            n_lat_cols = np.size(latitudes, 1)
            n_datapoints = n_lat_rows * n_lat_cols
            roi_percentage = n_indices/n_datapoints
            print(f"The Bergen roi is covered in {n_indices} datapoints ( {np.round(roi_percentage,5)} % "
                  f"of {n_datapoints} datapoints in total).")

            cloud_mask = np.array(granule.select("Cloud_Mask")[:])[0,:,:] # Get 2D array
            ind_x_1km, ind_y_1km = interpolate_to_1km_resolution(ind_x_5km, ind_y_5km, np.shape(latitudes),np.shape(cloud_mask))

            # Loop over all datasets and extract ROI
            reduced_datasets = {}
            attr_dict = {}

            for dataset_name in granule.datasets():
                dataset = np.array(granule.select(dataset_name)[:])
                attr_dict[dataset_name] = granule.select(dataset_name).attributes().items()

                if np.shape(dataset) == np.shape(latitudes):
                    # 5 km resolution, 2D
                    reduced_datasets[dataset_name] = dataset[ind_x_5km, ind_y_5km] # todo: check if it's a problem that cell along-swath and cell-across swath are not preserved
                elif dataset_name == "Quality_Assurance" or dataset_name == "Cloud_Mask_SPI":
                    # 3D with spatial Cell-Along-Swath in dimension 1, Cell-Accross-Swath in dimension 2
                    n_rows_dataset = np.shape(dataset)[0]
                    n_cols_dataset = np.shape(dataset)[1]
                    # Assert is 1 km resolution - may have a few more datapoints than exactly 5*n_lat_rows/cols
                    assert n_lat_rows * 5 <= n_rows_dataset < n_lat_rows * 6 and n_lat_cols * 5 <= n_cols_dataset < n_lat_cols * 6
                    reduced_datasets[dataset_name] = dataset[ind_x_1km, ind_y_1km, :]
                elif dataset_name == "Cloud_Mask":
                    # 3D with Cell-Along-Swath in dimension 2, Cell-Accross-Swath in dimension 3
                    n_rows_dataset = np.shape(dataset)[1]
                    n_cols_dataset = np.shape(dataset)[2]
                    # Assert is 1 km resolution - may have a few more datapoints than exactly 5*n_lat_rows/cols
                    assert n_lat_rows * 5 <= n_rows_dataset < n_lat_rows * 6 and n_lat_cols * 5 <= n_cols_dataset < n_lat_cols * 6
                    reduced_datasets[dataset_name] = dataset[:, ind_x_1km, ind_y_1km]
                else:
                    print(f"Dataset {dataset_name} has unknown shape: {np.shape(dataset)}")
                    reduced_datasets[dataset_name] = dataset


            # Save reduced hdf file in same file format
            save_filtered_hdf(filename, reduced_datasets, attr_dict, outdir)

            # Close file
            granule.end()


if __name__ == '__main__':
    filepath = "../data/MOD35_L2/2000/056"

