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
import skill_metrics as sm
from adjustText import adjust_text
import seaborn as sns



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
    
def get_diff_range(diffs: dict) -> tuple:
    all_diffs = np.concatenate([arr.ravel() for arr in diffs.values()])
    diff_range = (np.nanmin(all_diffs), np.nanmax(all_diffs))
    return diff_range


def plot_diff_heatmap(ax, diff: np.ndarray, boundaries: list, label: str, title: str, force_cbar: tuple=False) -> None:
    """
    Plots a heatmap of the difference between the two arrays.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the heatmap on.
        diff (np.ndarray): The difference between the two arrays.
        boundaries (list): The boundaries of the plot.
        label (str): The label of the plot.
        title (str): The title of the plot.
        threshold (float, optional): The threshold for the colorbar. Defaults to 0.05.
    """
    # Create a diverging colormap
    cmap = plt.get_cmap('RdBu_r')
    
    # Determine the maximum absolute difference for symmetric color scaling
    set_vmin, set_vmax = None, None
    if not force_cbar:
        max_diff = max(abs(np.min(diff)), abs(np.max(diff)))
        set_vmin = -max_diff
        set_vmax = max_diff
    else:
        set_vmin, set_vmax = force_cbar
    
    # Create a normalize object with a center point
    norm = mcolors.TwoSlopeNorm(vmin=set_vmin, vcenter=0, vmax=set_vmax)
    
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='whitesmoke')
    
    im = ax.imshow(diff, origin='lower', cmap=cmap, extent=boundaries, 
                   transform=ccrs.PlateCarree(), interpolation='none', norm=norm)
    
    cbar = plt.colorbar(im, ax=ax, label=label, orientation='horizontal', pad=0.05, fraction=0.046)
    cbar.set_label(label)
    ax.set_title(f"{title}")
    
def create_transparent_cmap():
    # Define a colormap (blue to red) with transparency in the middle
    cmap = plt.cm.RdBu_r
    transparent_cmap = cmap(np.linspace(0, 1, 256))
    
    # Set alpha (transparency) values near the middle (around zero correlation) to be transparent
    middle = int(256 / 2)
    transparent_cmap[middle - 10: middle + 10, -1] = np.linspace(0, 1, 20)

    # Create a colormap object
    return mcolors.ListedColormap(transparent_cmap)


def plot_corr_heatmap_rigid(ax, mat: np.ndarray, boundaries: list, label: str, title: str, threshold: float = 0) -> None:
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

def plot_corr_heatmap_flexible(ax, mat: np.ndarray, boundaries: list, label: str, title: str, treshold: float = 0) -> None:
    transparent_cmap = create_transparent_cmap()
    # Set normalization to center around zero
    norm = mcolors.TwoSlopeNorm(vmin=np.min(mat), vcenter=np.mean(mat), vmax=np.max(mat))

    # Plot using Cartopy
    ax.set_extent(boundaries, crs=ccrs.PlateCarree())

    # Add features like coastlines, borders, and land
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='whitesmoke')

    # Display the correlation data with the custom colormap
    im = ax.imshow(mat, origin='lower', cmap=transparent_cmap, extent=boundaries,
                   transform=ccrs.PlateCarree(), interpolation='none', norm=norm)

    # Add a colorbar with a label
    cbar = plt.colorbar(im, ax=ax, label='Correlation', orientation='horizontal', pad=0.05, fraction=0.046)
    cbar.set_label('Correlation')
    ax.set_title(f"{title}")

def plot_corr_diff(ax, same_sign_diff: np.ndarray, different_sign_diff: np.ndarray, boundaries: list, label: str, title: str) -> None:
    # Define colormaps
    cmap_same = plt.get_cmap('Reds')
    cmap_diff = plt.get_cmap('Blues')

    ax.set_extent(boundaries, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='whitesmoke')

    # Plot same-sign differences
    norm_same = mcolors.Normalize(vmin=np.nanmin(same_sign_diff), vmax=np.nanmax(same_sign_diff))
    im_same = ax.imshow(same_sign_diff, origin='lower', cmap=cmap_same, extent=boundaries,
                        transform=ccrs.PlateCarree(), interpolation='none', norm=norm_same, alpha=0.7)

    # Plot different-sign differences
    norm_diff = mcolors.Normalize(vmin=np.nanmin(different_sign_diff), vmax=np.nanmax(different_sign_diff))
    im_diff = ax.imshow(different_sign_diff, origin='lower', cmap=cmap_diff, extent=boundaries,
                        transform=ccrs.PlateCarree(), interpolation='none', norm=norm_diff, alpha=0.7)

    # Add colorbars
    cbar_same = plt.colorbar(im_same, ax=ax, label=f'Same Sign {label}', orientation='vertical', location='left', pad=0.05, fraction=0.046)
    cbar_same.set_label(f'Same Sign {label}')
    cbar_diff = plt.colorbar(im_diff, ax=ax, label=f'Different Sign {label}', orientation='horizontal', pad=0.15, fraction=0.086)
    cbar_diff.set_label(f'Different Sign {label}')

    ax.set_title(f"{title}")

    # Add custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='red', edgecolor='r', label='Same Sign'),
        Patch(facecolor='blue', edgecolor='b', label='Different Sign')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

def plot_taylor(stds, rmsds, avg_corrs, labels):
    fig, ax = plt.subplots(figsize=(15, 6))
    sm.taylor_diagram(stds, rmsds, avg_corrs, markerLabel = labels, markerLegend = 'on',
                      markerLabelColor = 'r', 
                      colRMS = 'm', styleRMS = ':', 
                      colSTD = 'b', styleSTD = '-.', 
                      colCOR = 'k', styleCOR = '--',
                     markerobs = 'o', titleOBS = 'Ref',
                      titleSTD ='on',
                     alpha = 0)
    
def plot_SAL(sal_scores: dict, title: str):
    models = list(sal_scores.keys())
    data = list(sal_scores.values())
    # Separate the data into S, A, and L
    S, A, L = zip(*data)
    # replace nan values
    L = [x if not np.isnan(x) else 2 for x in L]
    S = [x if not np.isnan(x) else -2 for x in S]

    # Create the scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(S, A, c=L, cmap='viridis', vmin=0, vmax=2, s=50)

    # Set the axis limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='k', linewidth=1)
    plt.axvline(0, color='k', linewidth=1)

    # Add labels and title
    plt.xlabel('S', fontweight='bold')
    plt.ylabel('A', fontweight='bold')
    plt.title(title)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('L')

    # Create annotations
    texts = []
    for i, model in enumerate(models):
        texts.append(plt.text(S[i], A[i], model, fontsize=8))

    # Adjust text positions to minimize overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

def create_regular_SAL_table(sal_scores):
    res =pd.DataFrame(sal_scores, index=["S", "A", "L"]).T
    res.sort_index(inplace=True)
    plt.figure(figsize=(8,12))
    plt.title("SAL Scores Results")
    ax = sns.heatmap(res, annot=True, cmap='coolwarm', vmin=-2, vmax=2, center=0)
    ax.xaxis.tick_top()
    plt.show()
    
    
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
def plot_corr_heatmap_roey(lat, lon, ax, corrs, name):
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    fig = plt.figure()

    #map = Basemap(projection='cyl', llcrnrlon=33.,llcrnrlat=29.,urcrnrlon=37.,urcrnrlat=34., resolution='i', ax=ax)
    mp = Basemap(projection='merc',llcrnrlon=20,llcrnrlat=20,urcrnrlon=50,urcrnrlat=40,resolution='i') # projection, lat/lon extents and resolution of polygons to draw
    # resolutions: c - crude, l - low, i - intermediate, h - high, f - full
    mp.drawcoastlines()
    mp.drawstates()
    mp.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
    mp.drawcounties() # you can even add counties (and other shapefiles!)

    #map.pcolormesh(lon_2d, lat_2d, pressureS[t,:,:], latlon=True, cmap='coolwarm') #Can change variables and index for time - also in line 48
    cmap = 'coolwarm'
    #vmin = 100000
    #vmax = 102000

    #cs = map.contour(lon_2d, lat_2d, pressureS[t, :, :], latlon=True, cmap=cmap)#, vmin=vmin, vmax=vmax) #lines
    cs = mp.contourf(lon_2d, lat_2d, np.squeeze(corrs), latlon=True, cmap=cmap)#, vmin=vmin, vmax=vmax) #color lines
    #cs = map.pcolormesh(lon_2d, lat_2d, pressureS[t, :, :], latlon=True, cmap=cmap)#, vmin=vmin, vmax=vmax)
    #Diverging Colormaps: 'coolwarm', 'RdBu_r', 'seismic', 'viridis', 'plasma', 'inferno', 'cividis', 'magma', 'tab10', ''
    # Adding pressure isobar lines
    #levels = np.arange(100000, 102000, 100) # define pressure levels
    #contour = map.contour(lon_2d, lat_2d, pressureS[t, :, :], levels=levels, latlon=True, colors='k', linewidths=1)

    # Adding labels to contour lines
    #plt.clabel(cs, inline=1, fontsize=10, fmt='%1.0f')

    mp.colorbar()
    #cbar = map.colorbar(cs, location='bottom', pad="5%")
    #cbar.set_label('Pressure (Pa)')
    parallels = np.arange(min(lat),max(lat),1.) # make latitude lines ever 5 degrees from 30N-50N
    meridians = np.arange(min(lon),max(lon),1.) # make longitude lines every 5 degrees from 95W to 70W
    mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=8, dashes=[1, 3], color='lightgray', linewidth=0.5)
    mp.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8, dashes=[1, 3], color='lightgray', linewidth=0.5)

    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    ax.set_ylabel('Latitude', loc='center' , labelpad=20.0)
    ax.set_xlabel('Longitude', loc='center', labelpad=13.0)
    plt.title(f'Correlation between DMI and Precip, model {name}') #Change title per variable

    plt.show()