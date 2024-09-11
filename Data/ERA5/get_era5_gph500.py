import cdsapi
import os
import tqdm

gph_path = os.path.join('GPH500')
dataset = "reanalysis-era5-pressure-levels-monthly-means"
request = {
    'product_type': ['monthly_averaged_reanalysis'],
    'variable': ['geopotential'],
    'pressure_level': ['500'],
    'year': [
                '1981', '1982', '1983',
                '1984', '1985', '1986',
                '1987', '1988', '1989',
                '1990', '1991', '1992',
                '1993', '1994', '1995',
                '1996', '1997', '1998',
                '1999', '2000', '2001',
                '2002', '2003', '2004',
                '2005', '2006', '2007',
                '2008', '2009', '2010',
            ],
    'month': ['10'],
    'time': ['00:00'],
    'data_format': 'netcdf',
    'download_format': 'unarchived',
    'area': [50, -10, -20, 110,]
}

c = cdsapi.Client()
try:
    c.retrieve(dataset, request, f'{gph_path}/ERA5_GPH500.nc').download()
    print("Retrieved ERA5 GPH500 data")
except Exception as e:
    print(f"Had trouble retrieving ERA5 GPH500 data, got error: {e}")
# for name in tqdm.tqdm(list(set(names) - set(failed))):
#     model = name.lower().replace('-', '_')
#     extract_nc_file(f'{gph_path}/{model}.zip', gph_path)

# print(f"Failed models: {failed}")