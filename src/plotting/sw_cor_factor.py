import xarray as xr
import numpy as np
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from datetime import datetime, timezone, timedelta


from src.model.surface_GHI_model import get_solar_angle

def plot_sw_cor_for_timestep(sw_cor_file, elevation_file, ind=10):
    """Plot elevation the shortwave correction factor for a specific timestep"""

    # Load sw cor data
    ds = xr.open_dataset(sw_cor_file)
    sw_dir_cor = ds["sw_dir_cor"][ind, :, :].values
    timestamp = ds["time"].isel(time=ind).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    sun_x = ds["sun_x"].values[ind]
    sun_y = ds["sun_y"].values[ind]
    sun_z = ds["sun_z"].values[ind]
    ds.close()
    timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[s]").astype(int), tz=timezone.utc)
    print("timestamp date: ", timestamp)
    sun_alt = np.degrees(np.arcsin(sun_z / np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)))
    sun_az  = np.degrees(np.arctan2(sun_x, sun_y))
    print(f"Sun altitude: {np.round(sun_alt,2)}, sun azimuth: {np.round(sun_az,2)}")
    
    zenith, azimuth = get_solar_angle(timestamp)
    print(f"Solar angles from pvlib: alt: {np.round(90-zenith,2)}, azimuth: {np.round(azimuth,2)} ")
    
    # Load elevation data 
    with rasterio.open(elevation_file) as src:
        dsm = src.read(1)  # first band

    elevation_ortho = np.ascontiguousarray(dsm)
    
    # Plot
    ax_lim = (lon.min(), lon.max(),
            lat.min(), lat.max())
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.1, right=0.9, top=0.9,
                        hspace=0.05, wspace=0.05, width_ratios=[1.0, 0.027])
    ax = plt.subplot(gs[0, 0])
    ax.set_facecolor(plt.get_cmap("terrain")(0.15)[:3] + (0.25,))
    
    # mask lov elevation (water)
    masked_elev = np.ma.masked_where(elevation_ortho < 0.5, elevation_ortho)
    levels = np.arange(0.0, 800.0, 50.0)
    cmap_terrain = colors.LinearSegmentedColormap.from_list(
        "terrain", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 100))
    )
    cmap = colors.ListedColormap(['blue'] + list(cmap_terrain(np.linspace(0,1,100))))
    norm = colors.BoundaryNorm(np.concatenate(([0, 0.5], levels[1:])), cmap.N, extend='max')
    #data_plot = np.ma.masked_where(mask_ocean[slice_in], elevation_ortho)
    plt.pcolormesh(lon, lat, elevation_ortho,
                cmap=cmap, norm=norm)
    x_ticks = np.arange(np.nanmin(lon), np.nanmax(lon), 0.05)
    plt.xticks(x_ticks, ["" for i in x_ticks])
    y_ticks = np.arange(np.nanmin(lat), np.nanmax(lat), 0.02)
    plt.yticks(y_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
    plt.axis(ax_lim)
    ax = plt.subplot(gs[0, 1])
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    plt.ylabel("Elevation [m a.s.l.]", labelpad=10.0)
    ax = plt.subplot(gs[1, 0])
    vmax = np.nanmax(sw_dir_cor)
    levels = np.arange(0.0, 2.25, 0.25)
    ticks = np.arange(0.0, 2.5, 0.5)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
    plt.pcolormesh(lon, lat, sw_dir_cor,
                cmap=cmap, norm=norm)
    plt.xticks(x_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$W" for i in x_ticks])
    plt.yticks(y_ticks, ["%.2f" % np.abs(i) + r"$^{\circ}$S" for i in y_ticks])
    plt.axis(ax_lim)
    txt = timestamp.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    t = plt.text(0.835, 0.935, txt, fontsize=11, fontweight="bold",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    #ts = load.timescale()
    #astrometric = loc_or.at(ts.from_datetime(timestamp).observe(sun)
    #alt, az, d = astrometric.apparent().altaz()
    txt = "Mean solar elevation angle: %.1f" % sun_alt + "$^{\circ}$"
    t = plt.text(0.21, 0.06, txt, fontsize=11, fontweight="bold",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes)
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax = plt.subplot(gs[1, 1])
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=ticks,
                            orientation="vertical")
    plt.ylabel("${\downarrow}SW_{dir}$ correction factor [-]", labelpad=10.0)
    ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    outpath = f"data/processed/sw_cor/Elevation_sw_dir_cor_{ts_str}.png"
    fig.savefig(outpath, dpi=300,
                bbox_inches="tight")
    print(f"Saved elevation and sw dir plot to {outpath}.")
    plt.close(fig)

if __name__ == "__main__":
    sw_cor_filepath = "data/processed/sw_cor/sw_cor_bergen.nc"
    DSM_filepath = "data/processed/bergen_dsm_10m_epsg4326.tif"
    plot_sw_cor_for_timestep(sw_cor_file=sw_cor_filepath, elevation_file=DSM_filepath)