import netCDF4 as nc
import os
import numpy as np

PATH = os.path.join('..', 'Data', 'CHIRPS2')

def read_data(file_path):
    data = nc.Dataset(file_path)
    return data

ds = read_data(os.path.join(PATH, 'chirps-v2.0.monthly.nc'))

time_var = ds.variables["time"][:]
october_indices = [i for i, t in enumerate(time_var) if (i + 1) % 11 == 0 ]
october_indices = october_indices[0:30]

lat_var = ds.variables["latitude"][:]
lon_var = ds.variables["longitude"][:]

lat_indices = np.where((lat_var >= 20) & (lat_var <= 40))[0]
lon_indices = np.where((lon_var >= 20) & (lon_var <= 50))[0]

precip_data = ds.variables["precip"]

october_data = precip_data[october_indices, :, :]
october_lat_lon_data = october_data[:, lat_indices, :][:, :, lon_indices]

# Create a new netCDF file to save the filtered data
output_nc_file = "chirps_octobers_middle_east_1981_2010_option3.nc"
nc_output = nc.Dataset(output_nc_file, "w", format="NETCDF4")

nc_output.createDimension("time", len(october_indices))
nc_output.createDimension('latitude', len(lat_indices))
nc_output.createDimension('longitude', len(lon_indices))

for name, variable in ds.variables.items():
    if name == 'latitude':
        new_var = nc_output.createVariable(name, variable.dtype, ("latitude",))
        new_var[:] = lat_var[lat_indices]
    elif name == 'longitude':
        new_var = nc_output.createVariable(name, variable.dtype,  ("longitude",))
        new_var[:] = lon_var[lon_indices]
    elif name == 'time':
        new_var = nc_output.createVariable(name, variable.dtype, ("time",))
        new_var[:] = variable[october_indices]
    elif name == 'precip':
        new_var = nc_output.createVariable(name, variable.dtype, ('time', 'latitude', 'longitude'))
        new_var[:] = october_lat_lon_data

# Copy global attributes
nc_output.setncatts({k: ds.getncattr(k) for k in ds.ncattrs()})

nc_output.close()
ds.close()
