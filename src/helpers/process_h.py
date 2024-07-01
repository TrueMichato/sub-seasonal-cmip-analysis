import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import interpolate
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


def collect_all_datasets(precipitation_dir: str, target_dim: tuple, unit_conversion: float, bounds_lat: list, bounds_lon: list) -> dict:
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
    interpolated_matrices = dict()

    for file in tqdm.tqdm(os.listdir(precipitation_dir)):
        filename = os.fsdecode(file)
        path = f"{precipitation_dir}/{filename}"
        data = nc.Dataset(path)
        print(f"{filename}")
        
        precipitation = data.variables['pr'][:]
        avg_pr_middle_east = np.mean(precipitation, axis=0)
        
        mm_avg_lat_lon = avg_pr_middle_east * unit_conversion
        
        original_matrix = mm_avg_lat_lon
        m, n = original_matrix.shape
        
        lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], m)
        lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], n)
        
        if m > target_dim[0] or n > target_dim[1]:
            # Downsample the matrix using bilinear interpolation
            scale_factors = target_dim[0] / m, target_dim[1] / n
            interpolated_matrix = scipy.ndimage.zoom(original_matrix, scale_factors, order=1)  # order=1 for bilinear

            
        else:
            # Upsample the matrix
            interpolator = interpolate.RegularGridInterpolator((lat_grid, lon_grid), original_matrix, method='slinear')

            new_lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], target_dim[0])
            new_lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], target_dim[1])

            new_grid_points = np.meshgrid(new_lat_grid, new_lon_grid, indexing='ij')

            new_points = np.stack([new_grid_points[0].ravel(), new_grid_points[1].ravel()], axis=-1)

            interpolated_matrix = interpolator(new_points).reshape(target_dim)
        
        interpolated_matrices[filename] = np.array(interpolated_matrix)

    return interpolated_matrices


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
            if any(item == np.nan for item in precip_series):
                correlation_map[i, j] = np.nan
            elif np.std(precip_series) == 0:
            # Set the correlation to NaN for constant series
                correlation_map[i, j] = np.nan
            else:
                correlation_map[i, j] = scipy.stats.pearsonr(dmi, precip_series)[0]

    return correlation_map



def calc_dmi_precp_corr_vec(dmi: np.ndarray, precip: np.ndarray) -> np.ndarray:
    # Reshape precipitation data from (30, 600, 600) to (30, 360000)
    # Each column represents a grid point's time series

#     # Step 1: Detrend the DMI to ensure no linear trend affects correlation
#     dmi_detrended = dmi - np.mean(dmi)

#     # Step 2: Detrend the precipitation data for each grid point
#     precip_detrended = precip - np.mean(precip, axis=0)

#     # Step 3: Normalize the detrended data
#     dmi_normalized = dmi_detrended / np.std(dmi_detrended)

#     # Step 4: Normalize the precipitation data along the time axis
#     precip_normalized = precip_detrended / np.std(precip_detrended, axis=0)

#     # Step 5: Compute the correlation coefficient across all grid points in one operation
#     # Dot product between normalized DMI and normalized precipitation for each grid point
#     correlation_map = np.tensordot(dmi_normalized, precip_normalized, axes=1) / (len(dmi) - 1)
    reshaped_precip = precip.reshape(dmi.shape[0], -1)

    # Calculate the mean and standard deviation of the reshaped chirps
    mean_precip = np.mean(reshaped_precip, axis=0)
    # Check for constant time series
    nan_mask = np.any(np.isnan(reshaped_precip), axis=0)
    std_precip = np.nanstd(reshaped_precip, axis=0)
    constant_series_mask = (std_precip == 0)

    # Combine masks to get final invalid mask
    invalid_mask = nan_mask | constant_series_mask

    # Compute the normalized (zero mean, unit variance) time series
    normalized_precip = (reshaped_precip - mean_precip) / std_precip

    # Calculate the mean and standard deviation of `dmi`
    mean_dmi = np.mean(dmi)
    std_dmi = np.std(dmi)

    # Normalize `dmi`
    normalized_dmi = (dmi - mean_dmi) / std_dmi

    # Compute the correlation for non-constant time series
    correlation_flat = np.dot(normalized_precip.T, normalized_dmi) / (dmi.shape[0] - 1)

    # Replace correlations for constant series with NaN
    correlation_flat[invalid_mask] = np.nan

    # Reshape back to (400, 600)
    correlations = correlation_flat.reshape(precip.shape[1], precip.shape[2])
    return correlations



def calc_corr_t_test(r: float, n: int = 30):
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    df = n - 2 # Degrees of freedom

    # Calculate p-values from the t-statistics
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df))
    return p_values
    










