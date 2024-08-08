import cdsapi
import os
import tqdm
from file_getting_funcs import get_filenames, extract_nc_file

files_path = os.path.join('Precipitation')
names = get_filenames(files_path)
gph_path = os.path.join('GPH500')

c = cdsapi.Client()
failed = []
for name in tqdm.tqdm(names):
    model = name.lower().replace('-', '_')
    try:
        c.retrieve(
            'projections-cmip6',
            {
                'format': 'zip',
                'temporal_resolution': 'monthly',
                'experiment': 'historical',
                'variable': 'geopotential_height',
                'level': '500',
                'model': model,
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
                'month': '10',
                'area': [
                    50, -10, -20, 110,
                ],
            },
            f'{gph_path}/{model}.zip')
    except:
        print(f"Model {model} not found")
        failed.append(name)
for name in tqdm.tqdm(list(set(names) - set(failed))):
    model = name.lower().replace('-', '_')
    extract_nc_file(f'{gph_path}/{model}.zip', gph_path)

print(f"Failed models: {failed}")