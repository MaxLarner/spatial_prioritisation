import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import os
import rasterio
import numpy as np
from rasterio.plot import plotting_extent
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar

# === File paths ===
CORINE_RASTER_PATH = '../data/base/corine_cumbria.tif'
CUMBRIA_SHAPEFILE = '../data/base/cumbria.shp'

# === Internal state ===
_cumbria_gdf = None
_urban_mask = None

# === Colour schemes ===
density_colours = [
    None,  # Bin 1 will be transparent
    "#a1d99b", "#66c2a5", "#31a354",  # Greens (Bin 2-4)
    "#ffffcc", "#ffe277", "#fee391",  # Yellows (Bin 5-7)
    "#fdd49e", "#fdae61", "#ec7014",  # Oranges (Bin 8-10)
    "#fb6a4a", "#e31a1c", "#b30000"   # Reds (Bin 11-13)
][1:]  # Slice to drop None for ListedColormap

natural_stronghold_colour = "#3182bd"
urban_colour = "#38617b"

# === Plotting setup ===
def configure_plotting_environment():
    mpl.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'serif',
        'text.usetex': False,
        'lines.linewidth': 2
    })

# === Add Cumbria backdrop ===
def add_cumbria_backdrop(ax=None, color='#f0f0f0', edgecolor='black', linewidth=0.5, zorder=0):
    global _cumbria_gdf
    if _cumbria_gdf is None:
        _cumbria_gdf = gpd.read_file(CUMBRIA_SHAPEFILE)
    if ax is None:
        ax = plt.gca()
    _cumbria_gdf.plot(ax=ax, color=color, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)

# === Add urban areas from CORINE raster ===
def get_urban_mask():
    global _urban_mask
    if _urban_mask is not None:
        return _urban_mask
    with rasterio.open(CORINE_RASTER_PATH) as src:
        corine = src.read(1)
        urban_mask = np.isin(corine, [111, 112, 141, 142])
        _urban_mask = (urban_mask, src.transform)
    return _urban_mask

# === Add colorbar ===
def add_colorbar(fig, ax, cmap, vmin, vmax, label="", orientation="vertical", shrink=0.9, pad=0.02):
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation=orientation, shrink=shrink, pad=pad)
    cbar.set_label(label)
    return cbar

# === Add scalebar ===
def add_scalebar(ax, location='lower right', units='km', length_fraction=0.25):
    with rasterio.open(CORINE_RASTER_PATH) as src:
        res = src.res[0]  # assumes square pixels
    scalebar = ScaleBar(dx=res, units=units, location=location,
                        length_fraction=length_fraction,
                        box_alpha=0, color='black', scale_loc='bottom')
    ax.add_artist(scalebar)

# === Save ===
def save_figure(name, formats=('pdf', 'png'), dpi=600, folder='figures'):
    os.makedirs(folder, exist_ok=True)
    for fmt in formats:
        plt.savefig(f"{folder}/{name}.{fmt}", dpi=dpi, bbox_inches='tight')

# === Main plotting function ===
def plot_suppression_map(raster_path, output_name=None, show_colorbar=False, show_scalebar=False, overlay_path=None, show_plot=False, ax=None):
    configure_plotting_environment()

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        standalone = True

    ax.set_axis_off()

    with rasterio.open(raster_path) as src:
        binned = src.read(1)
        extent = plotting_extent(src)

        cmap = ListedColormap(density_colours[::-1])  # Reverse colour order
        img = ax.imshow(np.where((binned >= 1) & (binned <= 12), binned, np.nan),
                        cmap=cmap, vmin=1, vmax=12, extent=extent, zorder=3)

    add_cumbria_backdrop(ax=ax)

    urban_mask, transform = get_urban_mask()
    with rasterio.open(CORINE_RASTER_PATH) as src:
        extent = plotting_extent(src)
        ax.imshow(np.where(urban_mask, 1, np.nan),
                  cmap=ListedColormap([urban_colour]), alpha=1.0, extent=extent, zorder=4)

    if overlay_path:
        with rasterio.open(overlay_path) as overlay:
            stronghold_data = overlay.read(1)
            extent_overlay = plotting_extent(overlay)
            ax.imshow(np.where(stronghold_data > 0, 1, np.nan),
                      cmap=ListedColormap([natural_stronghold_colour]),
                      alpha=1.0, extent=extent_overlay, zorder=5)

    if show_colorbar and standalone:
        add_colorbar(fig, ax, cmap=cmap, vmin=1, vmax=12, label="Grey density", orientation="vertical")

    if show_scalebar:
        add_scalebar(ax)

    if standalone:
        if output_name:
            save_figure(output_name)
        if show_plot:
            plt.show()
        else:
            plt.close()

    return ax
