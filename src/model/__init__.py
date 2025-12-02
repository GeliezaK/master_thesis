import math
import pandas as pd
import numpy as np

# Define the center of the bounding box (Bergen, Norway)
CENTER_LAT = 60.39
CENTER_LON = 5.33

# Approximate degree adjustments for 100km x 100km box
DEG_LAT_TO_KM = 111.412  # 1 degree latitude at 60° converted to km (https://en.wikipedia.org/wiki/Latitude)
DEG_LON_TO_KM = 111.317 * math.cos(np.deg2rad(CENTER_LAT))  # 1 degree longitude converted to km
LAT_OFFSET = 12.5 / DEG_LAT_TO_KM  # ~10km north/south
LON_OFFSET = 12.5 / DEG_LON_TO_KM  # ~10km east/west (varies with latitude, approximation)

# Define the bounding box
BBOX = {
    "north": CENTER_LAT + LAT_OFFSET, # 60.50219617276416
    "south": CENTER_LAT - LAT_OFFSET, # 60.27780382723584
    "west": CENTER_LON - LON_OFFSET,  #  5.10273148294384
    "east": CENTER_LON + LON_OFFSET   #  5.55726851705616
} 

MIXED_THRESHOLD = 1
OVERCAST_THRESHOLD = 99

FLORIDA_LAT, FLORIDA_LON = 60.38375436372568, 5.331906586858453
FLESLAND_LAT, FLESLAND_LON = 60.28911716265775, 5.227437237523992

COARSE_RESOLUTIONS = [100, 500, 1000, 5000, 25000]
