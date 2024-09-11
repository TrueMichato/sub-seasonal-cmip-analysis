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


VAR_NAME = {'precip': 'pr',
           'slp': 'psl',
           'gph500': 'zg'}
##############################################################################################################################
    ########################################## for initial exploration part ##########################################
##############################################################################################################################
# def scale_chirps(target_dim: tuple, chirps_precip: np.ndarray) -> tuple:
#     """This function scales the CHIRPS data to the target dimension, currently not in use"""
#     scale_factors = target_dim[0] / chirps_precip_data.shape[1], target_dim[1] / chirps_precip_data.shape[2]
#     chirps_precip = scipy.ndimage.zoom(np.mean(chirps_precip_data, axis=0), scale_factors, order=1)  # order=1 for bilinear
#     return chirps_precip

def interpolate_data(mat: np.ndarray, target_dim: tuple, bounds_lat: list, bounds_lon: list, options: dict) -> np.ndarray:
    res_mat = mat
    force_grid = options.get("force grid", False)
    if force_grid:
        m, n = force_grid
    else:
        m, n = res_mat.shape if mat.ndim == 2 else res_mat[0].shape

    lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], m)
    lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], n)
    if mat.ndim == 2:  # Interpolation for 2D matrix
        # Interpolation
        if m > target_dim[0] or n > target_dim[1]:
            # Downsample the matrix using bilinear interpolation
            scale_factors = target_dim[0] / m, target_dim[1] / n
            res_mat = scipy.ndimage.zoom(res_mat, scale_factors, order=1)  # order=1 for bilinear


        else:
            # Upsample the matrix
            interpolator = scipy.interpolate.RegularGridInterpolator((lat_grid, lon_grid), res_mat, method='slinear')

            new_lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], target_dim[0])
            new_lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], target_dim[1])

            new_grid_points = np.meshgrid(new_lat_grid, new_lon_grid, indexing='ij')

            new_points = np.stack([new_grid_points[0].ravel(), new_grid_points[1].ravel()], axis=-1)

            res_mat = interpolator(new_points).reshape(target_dim)
        
    elif mat.ndim == 3:  # Interpolation for 3D matrix
        interpolated_slices = []
        for slice_2d in res_mat:
            if slice_2d.shape != (m, n):
                raise ValueError(f"Shape mismatch: slice shape {slice_2d.shape} vs grid shape ({m}, {n})")
            
            if m > target_dim[0] or n > target_dim[1]:
                scale_factors = target_dim[0] / m, target_dim[1] / n
                interpolated_slice = scipy.ndimage.zoom(slice_2d, scale_factors, order=1)
            else:
                interpolator = scipy.interpolate.RegularGridInterpolator((lat_grid, lon_grid), slice_2d, method='slinear')
                new_lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], target_dim[0])
                new_lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], target_dim[1])
                new_grid_points = np.meshgrid(new_lat_grid, new_lon_grid, indexing='ij')
                new_points = np.stack([new_grid_points[0].ravel(), new_grid_points[1].ravel()], axis=-1)
                interpolated_slice = interpolator(new_points).reshape(target_dim)
            
            interpolated_slices.append(interpolated_slice)
        
        res_mat = np.array(interpolated_slices)
    return np.array(res_mat)

def precip_processing(data: np.ndarray, target_dim: tuple, unit_conversion: float, bounds_lat: list, bounds_lon: list, options: dict) -> np.ndarray:
    """
    Processes precipitation data by applying unit conversion and optional interpolation and averaging.

    Args:
        data (np.ndarray): The input dataset containing precipitation data.
        target_dim (tuple): Target dimensions for the output matrix.
        unit_conversion (float): Factor for converting units of precipitation.
        bounds_lat (list): Latitude bounds for the data.
        bounds_lon (list): Longitude bounds for the data.
        options (dict): Processing options with keys 'interpolate', 'force grid', and 'average'.

    Returns:
        np.ndarray: Processed precipitation data, possibly interpolated to the target dimensions.
    """
    # Unit Conversion
    mat = data.variables['pr'][:]
    if options.get("average"):
        mat = np.mean(mat, axis=0)

    converted = mat * unit_conversion

    res_mat = converted
    print(res_mat.shape)
    if options.get("interpolate"):
        force_grid = options.get("force grid", False)
        if force_grid:
            m, n = force_grid
        else:
            m, n = res_mat.shape if mat.ndim == 2 else res_mat[0].shape

        lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], m)
        lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], n)
        if mat.ndim == 2:  # Interpolation for 2D matrix
            # Interpolation
            if m > target_dim[0] or n > target_dim[1]:
                # Downsample the matrix using bilinear interpolation
                scale_factors = target_dim[0] / m, target_dim[1] / n
                res_mat = scipy.ndimage.zoom(res_mat, scale_factors, order=1)  # order=1 for bilinear


            else:
                # Upsample the matrix
                interpolator = scipy.interpolate.RegularGridInterpolator((lat_grid, lon_grid), res_mat, method='slinear')

                new_lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], target_dim[0])
                new_lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], target_dim[1])

                new_grid_points = np.meshgrid(new_lat_grid, new_lon_grid, indexing='ij')

                new_points = np.stack([new_grid_points[0].ravel(), new_grid_points[1].ravel()], axis=-1)

                res_mat = interpolator(new_points).reshape(target_dim)
         
        elif mat.ndim == 3:  # Interpolation for 3D matrix
            interpolated_slices = []
            for slice_2d in res_mat:
                if slice_2d.shape != (m, n):
                    raise ValueError(f"Shape mismatch: slice shape {slice_2d.shape} vs grid shape ({m}, {n})")
                
                if m > target_dim[0] or n > target_dim[1]:
                    scale_factors = target_dim[0] / m, target_dim[1] / n
                    interpolated_slice = scipy.ndimage.zoom(slice_2d, scale_factors, order=1)
                else:
                    interpolator = scipy.interpolate.RegularGridInterpolator((lat_grid, lon_grid), slice_2d, method='slinear')
                    new_lat_grid = np.linspace(bounds_lat[0], bounds_lat[1], target_dim[0])
                    new_lon_grid = np.linspace(bounds_lon[0], bounds_lon[1], target_dim[1])
                    new_grid_points = np.meshgrid(new_lat_grid, new_lon_grid, indexing='ij')
                    new_points = np.stack([new_grid_points[0].ravel(), new_grid_points[1].ravel()], axis=-1)
                    interpolated_slice = interpolator(new_points).reshape(target_dim)
                
                interpolated_slices.append(interpolated_slice)
            
            res_mat = np.array(interpolated_slices)

    res_2 = interpolate_data(converted, target_dim, bounds_lat, bounds_lon, options)
    assert np.array_equal(np.array(res_mat), res_2)
    return np.array(res_mat)
    


def collect_all_datasets(dir_path: str, data_type: str, target_dim: tuple = None, unit_conversion: float= None, bounds_lat: list= None, bounds_lon: list= None, options: dict=None) -> dict:
    """
    Collects and processes all NetCDF files from the specified directory.

    Args:
        dir_path (str): Path to the directory containing NetCDF files.
        data_type (str): Type of data to process ('precip' for precipitation).
        target_dim (tuple, optional): Target dimensions for interpolation.
        unit_conversion (float, optional): Factor for unit conversion.
        bounds_lat (list, optional): Latitude bounds for the data.
        bounds_lon (list, optional): Longitude bounds for the data.
        options (dict, optional): Processing options with keys 'interpolate', 'force grid', and 'average'.

    Returns:
        dict: A dictionary containing processed datasets or raw data.
    """
    output = dict()
    
    if not options:
        options = {"interpolate": False,
                   "force grid": False,
                   "average": False}

    for file in tqdm.tqdm(os.listdir(dir_path)):
        filename = os.fsdecode(file)
        path = f"{dir_path}/{filename}"
        data = nc.Dataset(path)
        print(f"{filename}")
        
        if data_type == 'precip':
            name_lat = 'lat' if 'lat' in data.variables.keys() else 'latitude'
            name_lon = 'lon' if 'lon' in data.variables.keys() else 'longitude'
            output[filename.split("_")[2]] = (precip_processing(data, target_dim, unit_conversion, bounds_lat, bounds_lon, options), data.variables[name_lat][:], data.variables[name_lon][:])
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
    


def calc_dmi(wtio: dict, seio: dict, calc_type: str='baseline') -> dict:
    dmi = dict()
    for key in tqdm.tqdm(wtio.keys()):
        wtio_model = wtio[key].variables['tos'][:]
        seio_model = seio[key].variables['tos'][:]
        if calc_type == 'baseline':
            wtio_reshape = wtio_model.reshape(30, -1)
            seio_reshape = seio_model.reshape(30, -1)
            wtio_mean = np.nanmean(wtio_reshape, axis=1)
            seio_mean = np.nanmean(seio_reshape, axis=1)
            dmi[key] = (wtio_mean - seio_mean) - np.nanmean(wtio_mean - seio_mean)
        elif calc_type == 'anomaly grid - mean last':
            wtio_climatology = np.nanmean(wtio_model, axis=0)
            seio_climatology = np.nanmean(seio_model, axis=0)
            wtio_anomaly = wtio_model - wtio_climatology
            seio_anomaly = seio_model - seio_climatology
            dmi[key] = np.nanmean((wtio_anomaly - seio_anomaly).reshape(30, -1), axis=1)
        elif calc_type == 'anomaly grid - mean first':
            wtio_climatology = np.nanmean(wtio_model, axis=0)
            seio_climatology = np.nanmean(seio_model, axis=0)
            wtio_anomaly = (wtio_model - wtio_climatology).reshape(30, -1)
            seio_anomaly = (seio_model - seio_climatology).reshape(30, -1)
            dmi[key] = np.nanmean(wtio_anomaly, axis=1) - np.nanmean(seio_anomaly, axis=1)
            
        # caluate sst anomaly for each grid, then either subtruct grids then means means then subtruct means
    return dmi

def signed_diff(baseline: np.ndarray, test: np.ndarray) -> np.ndarray:
    # Signed difference
    signed_diff = baseline - test

    # Sign change indicator
    sign_change = np.sign(baseline) != np.sign(test)

    # Combine signed difference with sign change information
    # Here, we multiply the signed difference by 2 if there is a sign change to emphasize it
    modified_diff = signed_diff + (sign_change * signed_diff)
    return modified_diff

def my_sign(x):
    signed = np.sign(x)
    signed[signed == 0] = 1
    return signed
 
def create_corr_diff_mats(baseline, corrs):
    mats = [(np.abs(baseline - arr), my_sign(baseline) != my_sign(arr)) for arr in list(corrs.values())]
    combined = []
    for mat in mats:
        same_sign_diff = np.where(~mat[1], mat[0], np.nan)
        different_sign_diff = np.where(mat[1], mat[0], np.nan)
        combined.append((same_sign_diff, different_sign_diff))
    titles = list(corrs.keys())
    return combined, titles

def calc_iod_precp_effect(iod_pos_neg, precip):
    pos = np.mean(precip[iod_pos_neg], axis=0)
    neg = np.mean(precip[~iod_pos_neg], axis=0)
    return pos - neg

def slp_process(slp, target_dim: tuple, bounds_lat: list, bounds_lon: list, options: dict) -> dict: 
    slp_p = dict()
    for key in slp.keys():
        if 'psl' in slp[key].variables.keys():
            res = slp[key]['psl'][:]
            if options.get("interpolate"):
                res = interpolate_data(res, target_dim, bounds_lat, bounds_lon, options)
            slp_p[key] = res
        else:
            res = np.squeeze(slp[key]['zg'][:])
            if options.get("interpolate"):
                res = interpolate_data(res, target_dim, bounds_lat, bounds_lon, options)
            slp_p[key] = res
    return slp_p

def gph_process(gph, target_dim: tuple, bounds_lat: list, bounds_lon: list, options: dict) -> dict:
    gph_p = dict()
    for key in gph.keys():
        res = np.squeeze(gph[key]['zg'][:], axis=1)
        if options.get("interpolate"):
            res = interpolate_data(res, target_dim, bounds_lat, bounds_lon, options)
        gph_p[key] = res
    return gph_p

def calc_iod_slp_effect(iod_pos_neg, slp):
    pos = np.mean(slp[iod_pos_neg], axis=0)
    neg = np.mean(slp[~iod_pos_neg], axis=0)
    return pos - neg

def calc_iod_gph_effect(iod_pos_neg, gph):
    pos = np.mean(gph[iod_pos_neg], axis=0)
    neg = np.mean(gph[~iod_pos_neg], axis=0)
    return pos - neg

def compute_SAL(Rmod, Robs, domain):
    # Domain information
    N = domain.size
    d = np.linalg.norm(np.array(domain.shape) - 1)  # Largest distance between boundary points in domain

    # Amplitude component (A)
    D_mod = np.mean(Rmod[domain==1])
    D_obs = np.mean(Robs[domain==1])
    A = (D_mod - D_obs) / (0.5 * (D_mod + D_obs))
    A = np.clip(A, -2, 2)

    # Location component (L)
    x_mod = np.array(np.unravel_index(np.argmax(Rmod), Rmod.shape)).mean(axis=0)  # Center of mass for Rmod
    x_obs = np.array(np.unravel_index(np.argmax(Robs), Robs.shape)).mean(axis=0)  # Center of mass for Robs
    L1 = np.linalg.norm(x_mod - x_obs) / d  # Normalized distance between centers of mass

    objects_mod = identify_objects(Rmod)
    objects_obs = identify_objects(Robs)

    r_mod = compute_weighted_distance(Rmod, objects_mod)
    r_obs = compute_weighted_distance(Robs, objects_obs)
    L2 = 2 * np.abs(r_mod - r_obs) / d  # Scaled and normalized

    L = L1 + L2  # Total location component

    # Structure component (S)
    V_mod = compute_scaled_volume(Rmod, objects_mod)
    V_obs = compute_scaled_volume(Robs, objects_obs)
    S = (V_mod - V_obs) / (0.5 * (V_mod + V_obs))

    return S, A, L

def identify_objects(R, threshold_factor=1/15):
    # Here we make no assumptions about the existence of objects in the data - every grid point can belong to an object
    max_R = np.max(R)
    threshold = threshold_factor * max_R
    objects = (R >= threshold)
    return objects

def compute_weighted_distance(R, objects):
    # Get the indices of all grid points where "objects" is True
    object_indices = np.array(np.nonzero(objects))  # shape will be (2, num_objects)
    
    # Compute total precipitation for normalization
    total_precip = np.sum(R[objects])
    
    # Calculate the weighted center of mass for the object
    weighted_positions = np.sum(object_indices * R[objects], axis=1) / total_precip
    
    # Compute the weighted distance of all object grid points to the center of mass
    distances = np.linalg.norm(object_indices.T - weighted_positions, axis=1)
    weighted_distances = np.sum(distances * R[objects]) / total_precip
    
    return weighted_distances


def compute_scaled_volume(R, objects):
    max_R = np.max(R[objects])
    scaled_volume = np.sum(R[objects]) / max_R  # Scaled precipitation volume
    if np.isnan(scaled_volume):
        print(f"{max_R=}, {np.sum(R[objects])=}, {scaled_volume=}")
    return scaled_volume