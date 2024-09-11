import cdsapi
import os
import tqdm

slp_path = os.path.join('SLP')
dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    'product_type': ['monthly_averaged_reanalysis'],
    'variable': ['mean_sea_level_pressure'],
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
    c.retrieve(dataset, request, f'{slp_path}/ERA5_MSLP.nc').download()
    print("Retrieved ERA5 MSLP data")
except Exception as e:
    print(f"Had trouble retrieving ERA5 MSLP data, got error: {e}")
# for name in tqdm.tqdm(list(set(names) - set(failed))):
#     model = name.lower().replace('-', '_')
#     extract_nc_file(f'{gph_path}/{model}.zip', gph_path)

# print(f"Failed models: {failed}")