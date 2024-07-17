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


def create_transparent_cmap():
    # Define a colormap (blue to red) with transparency in the middle
    cmap = plt.cm.RdBu_r
    transparent_cmap = cmap(np.linspace(0, 1, 256))
    
    # Set alpha (transparency) values near the middle (around zero correlation) to be transparent
    middle = int(256 / 2)
    transparent_cmap[middle - 10: middle + 10, -1] = np.linspace(0, 1, 20)

    # Create a colormap object
    return mcolors.ListedColormap(transparent_cmap)


def plot_corr_heatmap(ax, mat: np.ndarray, boundaries: list, label: str, title: str, threshold: float = 0) -> None:
    # Normalize always between -1 and 1
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    # Create a colormap with transparency between -0.1 and 0.1
    # Define a colormap with transparency in the middle
    cmap = plt.get_cmap('coolwarm').copy()  # Use a diverging colormap
    transparent_cmap = cmap(np.linspace(0, 1, 256))  # Get colormap colors

    # Set alpha (transparency) between -0.1 and 0.1
    midpoint = np.linspace(0, 1, 256)
    transparent_cmap[(midpoint > 0.48) & (midpoint < 0.52), -1] = 0  # Make the middle range transparent

    # Create the custom colormap
    transparent_cmap = mcolors.ListedColormap(transparent_cmap)

    # Plot using Cartopy
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())

    # Add features like coastlines, borders, and land
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='whitesmoke')

    # Display the correlation data with the custom colormap
    im = ax.imshow(mat, origin='lower', cmap=transparent_cmap, extent=boundaries,
                   transform=ccrs.PlateCarree(), interpolation='none', norm=norm)

    # Add a colorbar with a label, always between -1 and 1
    cbar = plt.colorbar(im, ax=ax, label='Correlation', orientation='horizontal', pad=0.05, fraction=0.046)
    cbar.set_label('Correlation')

    ax.set_title(title)
    ax.set_xlabel(label)
# def plot_corr_heatmap(ax, mat: np.ndarray, boundaries: list, label: str, title: str, treshold: float = 0) -> None:
#     # Set normalization to center around zero
#     norm = mcolors.TwoSlopeNorm(vmin=np.min(mat), vcenter=0, vmax=np.max(mat))

#     # Plot using Cartopy
#     ax.set_extent(boundaries, crs=ccrs.PlateCarree())

#     # Add features like coastlines, borders, and land
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.add_feature(cfeature.LAND, color='whitesmoke')

#     # Display the correlation data with the custom colormap
#     im = ax.imshow(mat, origin='lower', cmap=transparent_cmap, extent=boundaries,
#                    transform=ccrs.PlateCarree(), interpolation='none', norm=norm)

#     # Add a colorbar with a label
#     cbar = plt.colorbar(im, ax=ax, label='Correlation', orientation='horizontal', pad=0.05, fraction=0.046)
#     cbar.set_label('Correlation')
#     ax.set_title(f"{title}")


#     # Set the title of the plot
#     ax.set_extent(boundaries, crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.add_feature(cfeature.LAND, color='whitesmoke')
    
#     norm = mcolors.Normalize(vmin=np.min(mat), vmax=np.max(mat))
#     im = ax.imshow(mat, origin='lower', cmap=transparent_cmap, extent=boundaries,
#                     transform=ccrs.PlateCarree(), interpolation='none', norm=norm)

#     cbar = plt.colorbar(im, ax=ax, label=label, orientation='horizontal', pad=0.05, fraction=0.046)
#     cbar.set_label(label)
# #     plt.show()