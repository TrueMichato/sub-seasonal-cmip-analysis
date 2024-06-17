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


def plot_precp_heatmap(ax, mat: np.ndarray, boundaries: list, label: str, title: str, treshold: float = 0.05) -> None:
    cmap = plt.get_cmap('jet')
    cmap_with_transparency = cmap(np.linspace(0, 1, cmap.N))
    
    #A treshold of 0.01 is the same as 1 mm
    cmap_with_transparency[:, -1] = np.where(np.linspace(0, 1, cmap.N) <= treshold, 0, 1)  # Last column is the alpha channel
    transparent_cmap = mcolors.ListedColormap(cmap_with_transparency)
       
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='whitesmoke')
    
    norm = mcolors.Normalize(vmin=np.min(mat), vmax=np.max(mat))
    im = ax.imshow(mat, origin='lower', cmap=transparent_cmap, extent=boundaries,
                    transform=ccrs.PlateCarree(), interpolation='none', norm=norm)

    cbar = plt.colorbar(im, ax=ax, label=label, orientation='horizontal', pad=0.05, fraction=0.046)
    cbar.set_label(label)
    ax.set_title(f"{title}")
#     plt.show()
