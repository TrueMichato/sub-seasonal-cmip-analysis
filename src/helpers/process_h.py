import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy
import tqdm
import matplotlib.colors as mcolors
# import pandas as pd
##############################################################################################################################
    ########################################## for initial exploration part ##########################################
##############################################################################################################################
def scale_chirps(target_dim: tuple, chirps_precip: np.ndarray) -> tuple:
    "This function scales the CHIRPS data to the target dimension, currently not in use"
    scale_factors = target_dim[0] / chirps_precip_data.shape[1], target_dim[1] / chirps_precip_data.shape[2]
    chirps_precip = scipy.ndimage.zoom(np.mean(chirps_precip_data, axis=0), scale_factors, order=1)  # order=1 for bilinear
    return chirps_precip

def precip_processing(data: np.ndarray, target_dim: tuple, unit_conversion: float, bounds_lat: list, bounds_lon: list, interpolate: bool, force_grid: tuple) -> np.ndarray:
    # Unit Conversion
    precipitation = data.variables['pr'][:]
    avg_pr_middle_east = np.mean(precipitation, axis=0)

    mm_avg_lat_lon = avg_pr_middle_east * unit_conversion

    res_mat = mm_avg_lat_lon
    if interpolate:
        m, n = res_mat.shape if not force_grid else force_grid

        lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], m)
        lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], n)
#         print(f"{lat_grid=}\n{lon_grid=}")

        # Interpolation
        if m > target_dim[0] or n > target_dim[1]:
            # Downsample the matrix using bilinear interpolation
            scale_factors = target_dim[0] / m, target_dim[1] / n
            res_mat = scipy.ndimage.zoom(original_matrix, scale_factors, order=1)  # order=1 for bilinear


        else:
            # Upsample the matrix
            interpolator = scipy.interpolate.RegularGridInterpolator((lat_grid, lon_grid), mm_avg_lat_lon, method='slinear')

            new_lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], target_dim[0])
            new_lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], target_dim[1])

            new_grid_points = np.meshgrid(new_lat_grid, new_lon_grid, indexing='ij')

            new_points = np.stack([new_grid_points[0].ravel(), new_grid_points[1].ravel()], axis=-1)

            res_mat = interpolator(new_points).reshape(target_dim)
    return np.array(res_mat)
    


def collect_all_datasets(dir_path: str, data_type: str, target_dim: tuple = None, unit_conversion: float= None, bounds_lat: list= None, bounds_lon: list= None, interpolate: bool = False, force_grid: tuple = False) -> dict:
    """
    Collects and processes all NetCDF files from the given directory.
    Interpolates each dataset to a predefined dimension and stores them in a 3D array.
    
    Args:
        precipitation_path (str): Path to the directory containing NetCDF files.
        target_dim (tuple): Dimensions of the target matrix.
        unit_conversion (float): Unit conversion factor.
        bounds_lat (list): Latitude bounds.
        bounds_lon (list): Longitude bounds.
    
    Returns:
        np.ndarray: 3D array of interpolated datasets.
    """
    output = dict()

    for file in tqdm.tqdm(os.listdir(dir_path)):
        filename = os.fsdecode(file)
        path = f"{dir_path}/{filename}"
        data = nc.Dataset(path)
        print(f"{filename}")
        
        if data_type == 'precip':
            output[filename.split("_")[2]] = precip_processing(data, target_dim, unit_conversion, bounds_lat, bounds_lon, interpolate, force_grid)
        else:
            output[filename.split("_")[2]] = data
    return output


##############################################################################################################################
    ##################################### for paper reproduction part ##########################################
##############################################################################################################################
def calc_dmi_precp_corr_loop(dmi: np.ndarray, precip: np.ndarray) -> np.ndarray:
    # Initialize an array to store the correlation coefficients for each grid point
    correlation_map = np.zeros((precip.shape[1], precip.shape[2]))

    # Compute the correlation for each grid point
    for i in range(precip.shape[1]):
        for j in range(precip.shape[2]):
            # Extract the time series for the current grid point
            precip_series = precip[:, i, j]

            # Compute the correlation coefficient between DMI and the current grid point's time series
            if np.any([item != item for item in precip_series]):
                correlation_map[i, j] = np.nan
            elif np.std(precip_series) == 0:
            # Set the correlation to NaN for constant series
                correlation_map[i, j] = np.nan
            else:
                correlation_map[i, j] = scipy.stats.pearsonr(dmi, precip_series)[0]

    return correlation_map


def calc_dmi_precp_corr_vec(dmi: np.ndarray, precip: np.ndarray) -> np.ndarray:
    B_reshaped = dmi[:, np.newaxis, np.newaxis] 

    # Calculate the correlation
    correlations = np.sum((precip - precip.mean(axis=0, keepdims=True)) * (B_reshaped - dmi.mean(axis=0, keepdims=True)), axis=0) / \
                   (dmi.shape[0] * precip.std(axis=0, keepdims=True) * dmi.std(axis=0, keepdims=True))
    return np.squeeze(correlations)


def calc_corr_t_test(r: float, n: int = 30):
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    df = n - 2 # Degrees of freedom

    # Calculate p-values from the t-statistics
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df))
    return p_values
    










