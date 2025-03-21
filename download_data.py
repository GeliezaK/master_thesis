import csv
import json
import math
import os
import sys
from datetime import datetime
from io import StringIO
from time import time
import numpy as np
import requests

desc = "This script recursively downloads all files from MOD35_L2 and MYD35_L2 for a 100x100km area around Bergen, " \
       "Norway, and stores them in a specified path. "


def deg2rad(deg):
    return deg * math.pi / 180


# User-Agent string
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n', '').replace('\r', '')
DOWNLOAD_URL_AQUA = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD35_L2/"  # since 2002/185
DOWNLOAD_URL_TERRA = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2/2000/056/"  # since 2000/055

# Your NASA API Token (Replace with your actual token)
with open("20250320_api_token.txt", "r") as file:
    API_TOKEN = file.read().strip()  # Read and remove any trailing newlines or spaces

# Define the center of the bounding box (Bergen, Norway)
CENTER_LAT = 60.39
CENTER_LON = 5.33

# Approximate degree adjustments for 100km x 100km box
DEG_LAT_TO_KM = 111.412  # 1 degree latitude at 60° converted to km (https://en.wikipedia.org/wiki/Latitude)
DEG_LON_TO_KM = 111.317 * math.cos(deg2rad(CENTER_LAT))  # 1 degree longitude converted to km
LAT_OFFSET = 50 / DEG_LAT_TO_KM  # ~50km north/south, ~ 0.45
LON_OFFSET = 50 / DEG_LON_TO_KM  # ~50km east/west (varies with latitude, approximation), ~ 0.9

print("km per longitude degree: ", DEG_LON_TO_KM, ", Lat offset: ", LAT_OFFSET, ", long offset: ", LON_OFFSET)

# Define the bounding box
BBOX = {
    "north": CENTER_LAT + LAT_OFFSET,
    "south": CENTER_LAT - LAT_OFFSET,
    "west": CENTER_LON - LON_OFFSET,
    "east": CENTER_LON + LON_OFFSET
}

# Define start and end dates (ALL available records)
START_DATE_TERRA = "2000-02-24"  # MODIS starts in 2000/055
START_DATE_AQUA = "2002-07-04"  # AQUA starts in 2002/185
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Directory to save data
SAVE_DIR = "data/MODIS_CloudMask_full_range"
os.makedirs(SAVE_DIR, exist_ok=True)

# Extract only granules containing Bergen area from downloaded patch, discard all other files

# Loop over files (days)


# Download a single file
def get_url(url, token=None, out=None):
    """Fetches a URL using the requests library and optional Bearer token. Downloads data into local file if out is
    specified. Otherwise, returns metadata. """
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
        # sys.exit(1)


def sync(src, dest, token):
    """Synchronizes the source URL with the destination directory."""
    """Synchronizes the source URL with the destination directory."""
    # Get a list of available granules/ subfolders
    try:
        metadata = get_url(f'{src}.csv', token)
        files = {'content': [row for row in csv.DictReader(StringIO(metadata), skipinitialspace=True)]}
    except requests.RequestException:
        metadata = get_url(f'{src}.json', token)
        files = json.loads(metadata)

    for file in files['content']:
        filesize = int(file['size'])
        path = os.path.join(dest, file['name'])
        url = src + '/' + file['name']

        # Filesize == 0 is used to indicate directory
        if filesize == 0:
            os.makedirs(path, exist_ok=True)
            sync(url, path, token)  # recursively iterate through child directory
        else:
            # If data not already exists locally
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f'Downloading: {path}')
                get_url(url, token, path)
            else:
                print(f'Skipping: {path}')


def main(argv):
    source = DOWNLOAD_URL_TERRA
    destination = "data/MOD35_L2/2000/056"
    token = API_TOKEN

    # TODO: count errors, store files with access error in list?
    # TODO: read token via file, do not upload file with token to GitHub, push code to github
    # TODO: read data for one day, check if Bergen area fully covered (~1-2 files)
    # TODO: maybe read remaining granules if Bergen area not fully covered
    # TODO: loop over all days, all years. Keep list of dates that already have coverage, do not download those again.
    os.makedirs(destination, exist_ok=True)
    sync(source, destination, token)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(1)
