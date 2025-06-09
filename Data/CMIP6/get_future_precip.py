import cdsapi
import os
import tqdm
from file_getting_funcs import get_filenames, extract_nc_file

files_path = os.path.join('Precipitation')
names = get_filenames(files_path)
future_path = os.path.join('FuturePrecipitation')

c = cdsapi.Client()
failed = []
for name in tqdm.tqdm(names):
    model = name.lower().replace('-', '_')
    print(f"Getting {model}")
    try:
        c.retrieve(
            "projections-cmip6",
            {
                'format': 'zip',
                "temporal_resolution": "monthly",
                "experiment": "ssp5_8_5",
                "variable": "precipitation",
                # 'level': '500',
                "model": model,
                "year": [
                    "2071", "2072", "2073",
                    "2074", "2075", "2076",
                    "2077", "2078", "2079",
                    "2080", "2081", "2082",
                    "2083", "2084", "2085",
                    "2086", "2087", "2088",
                    "2089", "2090", "2091",
                    "2092", "2093", "2094",
                    "2095", "2096", "2097",
                    "2098", "2099", "2100"
                ],
                "month": ["10"],
                "area": [
                    40, 20, 20, 50
                ],
            },
            f'{future_path}/{model}.zip')
    except Exception as e:
        print(e)
        print(f"Model {model} not found")
        failed.append(name)
for name in tqdm.tqdm(list(set(names) - set(failed))):
    model = name.lower().replace('-', '_')
    extract_nc_file(f'{future_path}/{model}.zip', future_path)

print(f"Failed models: {failed}")