import os
import zipfile
import tqdm

def get_filenames(path):
    names = []
    for file in tqdm.tqdm(os.listdir(path)):
        filename = os.fsdecode(file)
        names.append(filename.split("_")[2])
    return names


def extract_nc_file(zip_file_path, output_directory):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.nc'):
                zip_ref.extract(file, output_directory)
                print(f"Extracted: {file}")
                break

    os.remove(zip_file_path)
    print(f"Deleted ZIP file: {zip_file_path}")
