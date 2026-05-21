import netCDF4 as nc
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy
import tqdm
import matplotlib.colors as mcolors
from statsmodels.stats.multitest import multipletests
import pysteps

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
    np.ma.set_fill_value(res_mat, 0)
    res_mat = res_mat.filled()
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

# def detrend(func):
#     def detrend_data(dmi: np.ndarray, precip: np.ndarray):
#         dmi_detrended = scipy.signal.detrend(dmi, axis=0)
#         precip_detrended = scipy.signal.detrend(precip, axis=0)
#         print("Detrended!")
#         return func(dmi_detrended, precip_detrended)
#     return detrend_data


def detrend(func):
    def detrend_data(dmi: np.ndarray, precip: np.ndarray):
        def masked_detrend(data_ma):
            detrended = np.ma.copy(data_ma)
            if not np.ma.is_masked(data_ma):
                return scipy.signal.detrend(data_ma, axis=0)
            valid_spatial_mask = ~data_ma.mask.any(axis=0)
            valid_data = data_ma[:, valid_spatial_mask]
            detrended_valid = scipy.signal.detrend(valid_data, axis=0)
            detrended[:, valid_spatial_mask] = detrended_valid
            return detrended
        dmi_detrended = scipy.signal.detrend(dmi, axis=0)
        precip_detrended = masked_detrend(precip)
        
        print("Detrended with Mask preserved!")
        return func(dmi_detrended, precip_detrended)
    return detrend_data

@detrend
def calc_dmi_precp_corr_vec(dmi: np.ndarray, precip: np.ma.MaskedArray) -> np.ndarray:
    # Ensure dmi is aligned with the shape for broadcasting
    # dmi shape: (time,) -> (time, 1, 1)
    dmi_reshaped = dmi[:, np.newaxis, np.newaxis] 

    # Use masked-aware operations
    # np.ma.mean and np.ma.std automatically ignore masked values
    precip_mean = precip.mean(axis=0, keepdims=True)
    dmi_mean = dmi.mean() # dmi is usually 1D time series
    
    precip_std = precip.std(axis=0, keepdims=True)
    dmi_std = dmi.std()

    # The numerator: Covariance-like sum
    # Masked arrays will propagate the mask during subtraction and multiplication
    numerator = np.ma.sum((precip - precip_mean) * (dmi_reshaped - dmi_mean), axis=0)
    
    # The denominator: 
    # count() returns the number of non-masked elements for each pixel
    n_counts = precip.count(axis=0) 
    denominator = n_counts * precip_std * dmi_std

    correlations = numerator / denominator
    return np.squeeze(correlations.filled(np.nan))


# def calc_dmi_precp_corr_vec(dmi: np.ndarray, precip: np.ndarray) -> np.ndarray:
#     B_reshaped = dmi[:, np.newaxis, np.newaxis] 

#     # Calculate the correlation
#     correlations = np.sum((precip - precip.mean(axis=0, keepdims=True)) * (B_reshaped - dmi.mean(axis=0, keepdims=True)), axis=0) / \
#                    (dmi.shape[0] * precip.std(axis=0, keepdims=True) * dmi.std(axis=0, keepdims=True))
#     return np.squeeze(correlations)


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

def calc_iod_precp_effect(iod_pos_neg, precip, multip=True):
    return test_means_diff(precip, iod_pos_neg, multip)
#     pos = np.mean(precip[iod_pos_neg], axis=0)
#     neg = np.mean(precip[~iod_pos_neg], axis=0)
#     return pos - neg

def calc_iod_slp_effect(iod_pos_neg, slp, multip=True):
    return test_means_diff(slp, iod_pos_neg, multip)
#     pos = np.mean(slp[iod_pos_neg], axis=0)
#     neg = np.mean(slp[~iod_pos_neg], axis=0)
#     return pos - neg

def calc_iod_gph_effect(iod_pos_neg, gph, multip=True):
    return test_means_diff(gph, iod_pos_neg, multip)
#     pos = np.mean(gph[iod_pos_neg], axis=0)
#     neg = np.mean(gph[~iod_pos_neg], axis=0)
#     return pos - neg

def correct_pvals(pvals, method, alpha):
    return multipletests(pvals, alpha=alpha, method=method)[1]

def apply_multiple_correction(data, pvals, method, alpha):
    corrected_pvals = correct_pvals(pvals, method, alpha)
    return np.where(corrected_pvals.reshape(data.shape) > alpha, 0, data)

def test_means_diff_simple(data1, data2, alpha=0.05, apply_multip=True, pre_avg=False):
    if not pre_avg:
        data1 = np.mean(data1, axis=0)
        data2 = np.mean(data2, axis=0)
    numerator = data1 - data2
    var_a = np.var(data1, axis=0) / data1.shape[0]
    var_b = np.var(data2, axis=0) / data2.shape[0]
    denominator = np.sqrt(var_a + var_b)
    statistic = numerator / denominator
    freedom = ((var_a + var_b) ** 2) / ((((var_a / data1.shape[0]) ** 2) / (data1.shape[0] - 1)) + (((var_b / data2.shape[0]) ** 2) / (data2.shape[0] - 1)))
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(statistic), freedom))
    if apply_multip:
        return apply_multiple_correction(numerator, p_values.flatten(), 'fdr_bh', alpha)
    else:
        return np.where(p_values.reshape(data1.shape) > alpha, 0, numerator)
    
    
def test_means_diff(data, indices, alpha=0.05, apply_multip=True, sample_correlation=False):
    """
    Performs a t-test to compare the means of two groups in the data.
    Assumes that the variance of both groups is known.
    Args:
        data (np.ndarray): The input dataset containing the data to be tested.
        indices (np.ndarray): A boolean array indicating the group membership for the t-test.
        alpha (float): Significance level for the test.
        apply_multip (bool): Whether to apply multiple testing correction.
    Returns:
        np.ndarray: The t-statistic for the test, or the corrected values if apply_multip is True.
    """
    if not sample_correlation:
        numerator = (np.mean(data[indices], axis=0) - np.mean(data[~indices], axis=0))
        var_a = np.var(data[indices], axis=0) / data[indices].shape[0]
        var_b = np.var(data[~indices], axis=0) / data[~indices].shape[0]
        denominator = np.sqrt(var_a + var_b)
        statistic = numerator / denominator
    #     freedom = data[indices].shape[0] + data[~indices].shape[0]  - 2
        freedom = ((var_a + var_b) ** 2) / ((((var_a /data[indices].shape[0]) **2) / (data[indices].shape[0] - 1)) + (((var_b / data[~indices].shape[0]) **2) / (data[~indices].shape[0] - 1)) )
        p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(statistic), freedom))
        if apply_multip:
            return apply_multiple_correction(numerator, p_values.flatten(), 'fdr_bh', alpha)
        else:
            return np.where(p_values.reshape(data.shape) > alpha, 0, numerator)
    else:
        numerator = np.mean(data[indices] - data[~indices], axis=0)
        var = np.var(data[indices], axis=0) + np.var(data[~indices], axis=0) - 2 * np.cov(data[indices], data[~indices], rowvar=False)
        denominator = np.sqrt(var / (data[indices].shape[0] + data[~indices].shape[0]))
        statistic = numerator / denominator
        freedom = data[indices].shape[0] + data[~indices].shape[0] - 2
        p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(statistic), freedom))
        if apply_multip:
            return apply_multiple_correction(numerator, p_values.flatten(), 'fdr_bh', alpha)
        else:
            return np.where(p_values.reshape(data.shape) > alpha, 0, numerator)

def slp_process(slp, target_dim: tuple, bounds_lat: list, bounds_lon: list, options: dict) -> dict: 
    slp_p = dict()
    for key in slp.keys():
        if 'psl' in slp[key].variables.keys():
            res = slp[key]['psl'][:]
            if options.get("interpolate"):
                res = interpolate_data(res, target_dim, bounds_lat, bounds_lon, options)
            slp_p[key] = res / 100 # change units from pascal to hectopascal
        else:
            res = np.squeeze(slp[key]['zg'][:])
            if options.get("interpolate"):
                res = interpolate_data(res, target_dim, bounds_lat, bounds_lon, options)
            slp_p[key] = res / 100 # change units from pascal to hectopascal
    return slp_p

def gph_process(gph, target_dim: tuple, bounds_lat: list, bounds_lon: list, options: dict) -> dict:
    gph_p = dict()
    for key in gph.keys():
        res = np.squeeze(gph[key]['zg'][:], axis=1)
        if options.get("interpolate"):
            res = interpolate_data(res, target_dim, bounds_lat, bounds_lon, options)
        gph_p[key] = res
    return gph_p


# def compute_SAL(Rmod, Robs, domain):
#     # Domain information
#     N = domain.size
#     d = np.linalg.norm(np.array(domain.shape) - 1)  # Largest distance between boundary points in domain

#     # Amplitude component (A)
#     D_mod = np.mean(Rmod[domain==1])
#     D_obs = np.mean(Robs[domain==1])
#     A = (D_mod - D_obs) / (0.5 * (D_mod + D_obs))
#     A = np.clip(A, -2, 2)

#     # Location component (L)
#     x_mod = np.array(np.unravel_index(np.argmax(Rmod), Rmod.shape)).mean(axis=0)  # Center of mass for Rmod
#     x_obs = np.array(np.unravel_index(np.argmax(Robs), Robs.shape)).mean(axis=0)  # Center of mass for Robs
#     L1 = np.linalg.norm(x_mod - x_obs) / d  # Normalized distance between centers of mass

#     objects_mod = identify_objects(Rmod)
#     objects_obs = identify_objects(Robs)

#     r_mod = compute_weighted_distance(Rmod, objects_mod)
#     r_obs = compute_weighted_distance(Robs, objects_obs)
#     L2 = 2 * np.abs(r_mod - r_obs) / d  # Scaled and normalized

#     L = L1 + L2  # Total location component

#     # Structure component (S)
#     V_mod = compute_scaled_volume(Rmod, objects_mod)
#     V_obs = compute_scaled_volume(Robs, objects_obs)
#     S = (V_mod - V_obs) / (0.5 * (V_mod + V_obs))

#     return S, A, L

# def identify_objects(R, threshold_factor=1/15):
#     # Here we make no assumptions about the existence of objects in the data - every grid point can belong to an object
#     max_R = np.max(R)
#     threshold = threshold_factor * max_R
#     objects = (R >= threshold)
#     return objects if not np.ma.is_masked(objects) else objects.data

# def compute_weighted_distance(R, objects):
#     # Get the indices of all grid points where "objects" is True
#     object_indices = np.array(np.nonzero(objects))  # shape will be (2, num_objects)
    
#     # Compute total precipitation for normalization
#     total_precip = np.sum(R[objects])
    
#     # Calculate the weighted center of mass for the object
#     weighted_positions = np.sum(object_indices * R[objects], axis=1) / total_precip
    
#     # Compute the weighted distance of all object grid points to the center of mass
#     distances = np.linalg.norm(object_indices.T - weighted_positions, axis=1)
#     weighted_distances = np.sum(distances * R[objects]) / total_precip
    
#     return weighted_distances


# def compute_scaled_volume(R, objects):
#     max_R = np.max(R[objects])
#     scaled_volume = np.sum(R[objects]) / max_R  # Scaled precipitation volume
#     if np.isnan(scaled_volume):
#         print(f"{max_R=}, {np.sum(R[objects])=}, {scaled_volume=}")
#     return scaled_volume

def compute_salx(var_anom, var_anom_ref, threshold_factor=None):
    """
    Compute the Structure, Amplitude, and Location (SAL) scores
    between a model anomaly field and a reference (ERA5) anomaly field.
    
    Parameters:
    ----------
    var_anom : ndarray
        2D array [lat, lon] of model anomalies (shape 25 x 45).
    var_anom_ref : ndarray
        2D array [lat, lon] of reference anomalies (ERA5, shape 25 x 45).
    threshold_factor : float, optional
        A fractional multiplier (0.0 to 1.0) applied to the maximum absolute 
        anomaly to define the threshold for centroid calculations. 
        E.g., 0.1 means only anomalies stronger than 10% of the domain's 
        maximum anomaly are used for the Location metric. 
        If None, all grid points are used.
    
    Returns:
    -------
    S : float
        Structure score [-2, 2].
    A : float
        Amplitude score [-2, 2].
    L : float
        Location score [0, 2].
    """
    
    # Flatten arrays to work on 1D
    model = var_anom.flatten()
    ref = var_anom_ref.flatten()
    
    # ---------------------------------------------------------
    # 1. Amplitude (A)
    # ---------------------------------------------------------

    scale = np.nanstd( ref )

    mean_model = np.nanmean(model)
    mean_ref = np.nanmean(ref)

    A =  2 * np.tanh(( mean_model - mean_ref ) / scale + 1e-10 )

    # ---------------------------------------------------------
    # 2. Structure (S)
    # ---------------------------------------------------------
    std_model = np.nanstd(model)
    std_ref = np.nanstd(ref)
    S = 2 * (std_model - std_ref) / (std_model + std_ref + 1e-10)
    
    # ---------------------------------------------------------
    # 3. Location (L)
    # ---------------------------------------------------------
    # Location (L) : distance between centroids normalized by domain size
    # Compute centroids (weighted by absolute anomaly)
    x = np.arange(var_anom.shape[1])  # longitude indices
    y = np.arange(var_anom.shape[0])  # latitude indices

    xx, yy = np.meshgrid(x, y)
    
    # Absolute anomaly weights
    weight_model = np.abs(var_anom)
    weight_ref = np.abs(var_anom_ref)
    
    if threshold_factor is not None:
        # Calculate thresholds as factor * max_value
        thresh_m = threshold_factor * np.nanmax(weight_model)
        thresh_r = threshold_factor * np.nanmax(weight_ref)
        
        # Zero out weights below the threshold
        weight_model[weight_model < thresh_m] = np.nan
        weight_ref[weight_ref < thresh_r] = np.nan

    
    # Model centroid
    sum_weight_model = np.nansum(weight_model) + 1e-10
    x_model = np.nansum(xx * weight_model) / sum_weight_model
    y_model = np.nansum(yy * weight_model) / sum_weight_model
    
    # Reference centroid
    sum_weight_ref = np.nansum(weight_ref) + 1e-10
    x_ref = np.nansum(xx * weight_ref) / sum_weight_ref
    y_ref = np.nansum(yy * weight_ref) / sum_weight_ref
    
    # Euclidean distance between centroids
    dist = np.sqrt((x_model - x_ref)**2 + (y_model - y_ref)**2)
    
    # Normalize distance by domain diagonal
    domain_diag = np.sqrt(var_anom.shape[0]**2 + var_anom.shape[1]**2)
    L = 2 * dist / domain_diag
    
    return S, A, L


def safe_weighted_avg(val_p, val_n, w_p, w_n):
    # PySteps can sometimes return NaN if no objects pass the threshold factor.
    # This ensures the final score isn't destroyed if only one side (pos/neg) is valid.
    valid_p = not np.isnan(val_p)
    valid_n = not np.isnan(val_n)

    if valid_p and valid_n:
        return (val_p * w_p) + (val_n * w_n)
    elif valid_p:
        return val_p  # Fallback entirely to positive score if negative is NaN
    elif valid_n:
        return val_n  # Fallback entirely to negative score if positive is NaN
    else:
        return np.nan


def compute_bisal(prediction, reference, factor=1/15, tstorm_kwargs=None):
    """
    Compute the BiSAL (Structure, Amplitude, Location - Twice) metric for 
    fields containing both positive and negative values.
    """
    # 1. Calculate the proportional weights based on total absolute "mass"
    # This represents how much "weight" the positives have vs the negatives
    pos_mask_pred, pos_mask_ref = prediction > 0, reference > 0
    neg_mask_pred, neg_mask_ref = prediction < 0, reference < 0

    mass_p = np.nansum(prediction[pos_mask_pred])
    mass_n = np.nansum(np.abs(prediction[neg_mask_pred]))
    total_mass = mass_p + mass_n

    # If the domain is completely empty/zero
    if total_mass == 0:
        return np.nan, np.nan, np.nan

    weight_p = mass_p / total_mass
    weight_n = mass_n / total_mass

    # Initialize scores
    Sp, Ap, Lp = np.nan, np.nan, np.nan
    Sn, An, Ln = np.nan, np.nan, np.nan

    # 2. Compute Positive SAL
    if mass_p > 0:
        pos_pred = np.copy(prediction)
        pos_ref = np.copy(reference)
        
        pos_pred[neg_mask_pred] = 0.0
        pos_ref[neg_mask_ref] = 0.0
       
        
        Sp, Ap, Lp = pysteps.verification.salscores.sal(pos_pred, pos_ref, thr_factor=factor, tstorm_kwargs=tstorm_kwargs)

    # 3. Compute Negative SAL
    if mass_n > 0:
        neg_pred = np.copy(prediction)
        neg_ref = np.copy(reference)
        # Extract negatives, convert to absolute positive values, set everything else to 0
        
        neg_pred[pos_mask_pred] = 0.0
        neg_ref[pos_mask_ref] = 0.0
        
        neg_pred = np.abs(neg_pred)
        neg_ref = np.abs(neg_ref)
        
        Sn, An, Ln = pysteps.verification.salscores.sal(neg_pred, neg_ref, thr_factor=factor, tstorm_kwargs=tstorm_kwargs)

    # 5. Calculate final BiSAL scores
    S = safe_weighted_avg(Sp, Sn, weight_p, weight_n)
    A = safe_weighted_avg(Ap, An, weight_p, weight_n)
    L = safe_weighted_avg(Lp, Ln, weight_p, weight_n)

    return S, A, L
    


def infer_tstorm_kwargs(data_array, minref_p=85, minmax_p=95, mindiff_f=0.5):
    """
    Dynamically deduce PySteps tstorm parameters based on the 
    statistical distribution of a 2D data array.
    """
    # Use absolute values to treat positive and negative anomalies equally
    abs_data = np.abs(data_array)
    valid_data = abs_data[~np.isnan(abs_data)]
    
    valid_data = np.array(valid_data, copy=True)
    
    if len(valid_data) == 0:
        # Fallback if the array is completely empty/NaN
        return {'minref': 0.1, 'minmax': 0.2, 'mindiff': 0.1, 'minsize': 3, 'mindis': 2}

    # 1. Intensity Heuristics (Percentiles)
    minref = np.nanpercentile(valid_data, minref_p)  # Top 15% of data
    minmax = np.nanpercentile(valid_data, minmax_p)  # Top 5% of data
    
    # 2. Separation Heuristic
    std_dev = np.nanstd(valid_data)
    mindiff = std_dev * mindiff_f  # Half a standard deviation to split peaks
    
   # 3. Spatial Heuristics (Quartile Coefficient of Dispersion)
    q1 = np.nanpercentile(valid_data, 25)
    q3 = np.nanpercentile(valid_data, 75)
    
    # Calculate QCD. Add epsilon to prevent division by zero in perfectly flat fields
    qcd = (q3 - q1) / (q3 + q1 + 1e-10)
    
    # Base spatial parameters on how "noisy/spiky" the data is using QCD
    if qcd > 0.7:
        # Highly localized/spiky (e.g., Precipitation differences)
        # The lower 25% is near zero, while the upper 25% is distinctly higher.
        minsize = 3
        mindis = 2
    elif qcd > 0.4:
        # Moderate spread (e.g., Correlation patterns)
        minsize = 6
        mindis = 5
    else:
        # Smooth and broad (e.g., Z500 or MSLP differences)
        # The data is relatively uniform across the middle 50%.
        minsize = 15
        mindis = 10

    return {
        'minref': float(minref),
        'minmax': float(minmax),
        'mindiff': float(mindiff),
        'minsize': int(minsize),
        'mindis': int(mindis)
    }
    