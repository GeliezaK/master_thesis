import csv
import json
import math
import os
from pyhdf.SD import SD, SDC  # HDF4 support
import sys
from datetime import datetime
from io import StringIO
import numpy as np
import requests

desc = "This script recursively downloads all files from MOD35_L2 and MYD35_L2 for a 100x100km area around Bergen, " \
       "Norway, and stores them in a specified path. "


def deg2rad(deg):
    return deg * math.pi / 180


# User-Agent string
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n', '').replace('\r', '')
DOWNLOAD_URL_AQUA = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD35_L2/"  # since 2002/185
DOWNLOAD_URL_TERRA = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2/2001"  # since 2000/055

# Your NASA API Token (Replace with your actual token)
with open("C:/Users/gelie/Home/ComputationalScience/MA/master_thesis/20250320_api_token.txt", "r") as file:
    API_TOKEN = file.read().strip()  # Read and remove any trailing newlines or spaces

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

# Define start and end dates (ALL available records)
START_DATE_TERRA = "2000/055"  # MODIS starts in 2000/055 (2000-02-24)
START_DATE_AQUA = "2002/185"  # AQUA starts in 2002/185 (2002-07-04)
day_of_year = datetime.now().timetuple().tm_yday
END_DATE = "2025/" + str(day_of_year)


def write_unique_urls(file_path, err_files): # TODO: unit test this
    try:
        # Read existing URLs from the file if it exists
        try:
            with open(file_path, "r") as f:
                existing_urls = set(line.strip() for line in f)
        except FileNotFoundError:
            existing_urls = set()

        # Filter new URLs that are not already in the file
        new_urls = [url for url in err_files if url not in existing_urls]

        # Append new URLs to the file
        if new_urls:
            with open(file_path, "a") as f:
                for url in new_urls:
                    f.write(url + "\n")

        print(f"Added {len(new_urls)} new URLs to {file_path}")

    except Exception as e:
        print(f"Error: {e}")



# Download a single file
def get_url(url, err_files, token=None, out=None):
    """Fetches a URL using the requests library and optional Bearer token. Downloads data into local file if out is
    specified. Otherwise, returns metadata.
    :param err_files: """
    headers = {'User-Agent': USERAGENT}

    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        if out:
            # write data to filepath
            with open(out, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            # Return metadata (filenames)
            return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        err_files.append(url)
        # sys.exit(1)


def sync(src, dest, token, err_files):
    """Synchronizes the source URL with the destination directory.
    :param err_files:
    """
    """Synchronizes the source URL with the destination directory."""
    # Get a list of available granules/ subfolders
    try:
        metadata = get_url(f'{src}.csv', err_files, token)
        files = {'content': [row for row in csv.DictReader(StringIO(metadata), skipinitialspace=True)]}
    except requests.RequestException:
        metadata = get_url(f'{src}.json', err_files, token)
        files = json.loads(metadata)

    for file in files['content']:
        filesize = int(file['size'])
        path = os.path.join(dest, file['name'])
        url = src + '/' + file['name']

        # Filesize == 0 is used to indicate directory
        if filesize == 0:
            os.makedirs(path, exist_ok=True)
            sync(url, path, token, err_files)  # recursively iterate through child directory
        else:
            # If data not already exists locally
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f'Downloading: {path}')
                get_url(url, err_files, token, path)
                # Keep file only if inside ROI
                delete_outside_roi(path)
            else:
                print(f'Skipping: {path}')

def is_within_bbox(latitudes, longitudes, bbox): #TODO write unit test for this
    """Check if any latitude/longitude points fall within the bounding box."""
    within_box = False

    for i in np.nditer([latitudes, longitudes]):
        # Iterate through each element and check if it is within bbox
        lat = i[0]
        lon = i[1]
        if bbox["south"] <= lat <= bbox["north"] and bbox["west"] <= lon <= bbox["east"]:
            within_box = True
            break

    return within_box


def delete_outside_roi(filepath):
    """Delete single file if it does not intersect with Bergen region of interest.
    Only keep data points that are within roi."""

    inside_roi = False

    try:
        # Open HDF file
        granule = SD(filepath, SDC.READ)

        # Extract Latitude and Longitude
        latitudes = np.array(granule.select("Latitude")[:])
        longitudes = np.array(granule.select("Longitude")[:])

        # Close file
        granule.end()

        if is_within_bbox(latitudes, longitudes, BBOX):
            inside_roi = True
            # TODO: cut out only datapoints that are inside roi
        else:
            os.remove(filepath)  # Delete file if outside Bergen
            print(f"Deleted: {filepath} (Outside Bergen area)")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")  # TODO: certain files may be corrupted

    return inside_roi



def filter_roi(path):
    """Loop over all files in path and delete all files that do not contain the Bergen region of interest. """
    matching_files = []

    # Loop over all .hdf files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".hdf"):
            filepath = os.path.join(path, filename)
            match = delete_outside_roi(filepath)
            if match :
                matching_files.append(filepath)
    return matching_files


def main(argv):
    source = DOWNLOAD_URL_TERRA
    destination = "data/MOD35_L2/2001"
    token = API_TOKEN
    err_files = []


    # TODO : reduce amount of data further, e.g. cut only lat/lon points that are within Bergen area, remove unnecessary
    # data


    os.makedirs(destination, exist_ok=True)
    sync(source, destination, token, err_files)
    print("Number of errorneous downloads: ", len(err_files))
    write_unique_urls("../errorneous_urls.txt", err_files)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(1)
