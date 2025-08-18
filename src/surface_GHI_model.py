# Model surface GHI 
# Inputs:   - filepath to DSM/DEM file 
#           - folderpath to LUT 
#           - filepath to file with input params to LUT
#           -

LUT_folderpath = "output/LUT"
DSM_filepath = "data/bergen_dem.tif"
LUT_params_filepath = "output/LUT/key_params.txt"


def read_LUT_key_params_to_list(LUT_params_filepath, variable): 
    """Read params in LUT filepath to local global variables. """
    # Open file 
    # Read the variables that are available 
    # return local list or dict 
    if variable == "doy": 
        return [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    elif variable == "hod": 
        return {15: [8, 9, 10, 11, 12, 13, 14, 15], 
                46: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                74: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
                105: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 
                135: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
                166: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 
                196: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 
                227: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
                258: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
                288: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
                319: [8, 9, 10, 11, 12, 13, 14, 15], 
                349: [9, 10, 11, 12, 13, 14]}
    elif variable == "albedo": 
        return [0.081, 0.129, 0.174, 0.224, 0.354] 
    elif variable == "tau": 
        return [1.0, 3.41, 5.50, 7.68, 10.18, 13.67, 19.34, 27.79, 42.03, 73.23, 125.42, 250.0]
    elif variable == "cloud_base_height": 
        return [0.08, 0.167, 0.285, 0.571, 0.915, 1.286, 1.753, 2.370, 3.171, 4.165, 5.451, 6.543, 8.498]
    elif variable == "cloud_type": 
        ['ice', 'water']
    else : 
        return []


DOY = read_LUT_key_params_to_list(LUT_params_filepath, "doy")
HOD = read_LUT_key_params_to_list(LUT_params_filepath, "hod")
ALBEDO = read_LUT_key_params_to_list(LUT_params_filepath, "albedo")
TAU = read_LUT_key_params_to_list(LUT_params_filepath, "tau")
CLOUD_BASE_HEIGHT = read_LUT_key_params_to_list(LUT_params_filepath, "cloud_base_height")
CLOUD_TYPE = read_LUT_key_params_to_list(LUT_params_filepath, "cloud_type")
    

def read_GHI_from_LUT(doy, hod, albedo, cloud_base_height, tau, cloud_type): 
    """Given params (Doy, Hour of day, albedo, cloud base height, cloud optical thickness (=tau), cloud_type (ice/water))
    get corresponding closest entry in LUT"""
    # Get closest parameter config that is available 
    get_LUT_key(doy, hod, albedo, cloud_base_height, tau, cloud_type)
    # Open corresponding LUT file
    # if nonexistend, throw error (here it should be asserted that all param configs give valid files)
    # turn to pd dataframe
    # Read from file GHI_clear, diffuse, direct, as well as CAF_clear, diffuse, direct
    # If returns none, throw error 
    # return ghi values
    
def get_LUT_key(doy, hod, albedo, cloud_base_height, tau, cloud_type): 
    """Match parameters with closest available config that has entry in the table"""
    # Get closest doy from list 
    # Get closest hod from list 
    # get clostest albedo 
    # get closest cloud_base_height
    # get closest tau
    # get closest cloud type 
    
def read_DSM(DSM_filepath): 
    """Read and return DSM file"""
    pass 

# TODO: get shaded pixels from DEM 
def read_satellite_image(satellite_filepath): 
    """Read s2 image"""
    pass 

def read_cloud_properties(): 
    """From other satellites (s5p): read cloud optical thickness, cloud base height, albedo, cloud type (s2harmonized), 
    """
    pass

def simulate_GHI_pixelwise(): 
    """for each pixel, calculate the GHI """
    # for each pixel in satellite image : 
    # Get cloud mask (cloud 0 or 1)
    # Get cloud properties (albedo, cloud_base_height, cloud_optical thickness, cloud_type)
    # calculate if shady 
    # if so, if clear:  GHI = GHI_diffuse_clear (set direct component to 0)
    # if so, if cloudy: GHI = GHI_diffuse_clear * CAF_diffuse
    # else (not shady): 
    # if clear : GHI = GHI_clear
    # if cloudy : GHI = GHI_clear * CAF_GHI
    # return: output is the map of GHI for each pixel 
