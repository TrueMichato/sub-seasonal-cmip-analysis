import os
import tqdm
from file_getting_funcs import get_filenames

slp_path = os.path.join('SLP')
slp_names = set(get_filenames(slp_path))

gph_path = os.path.join('GPH500')
gph_names = set(get_filenames(gph_path))


print(slp_names - gph_names)


