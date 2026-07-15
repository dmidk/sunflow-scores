import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import QuadMesh
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
from pathlib import Path
import datetime
import argparse
from pyproj import Transformer

def create_comparison_gif(nowcast_file_path):
    """
    Generates a GIF comparing DINI forecast, Sunflow nowcast, and satellite data.
    """
    # --- Configuration ---
    # Define paths to the data sources
    DINI_ZARR_PATH = "/dmidata/projects/energivejr-data/dini/consolidated/dini_sharded.zarr"
    
    # Extract year and month from nowcast path or filename for SATELLITE_DIR
    nowcast_path = Path(nowcast_file_path)
    # Try to find YYYYMM pattern in the path
    year_month_match = re.search(r'/(\d{4}\d{2})/', str(nowcast_path))
    if year_month_match:
        year_month_str = year_month_match.group(1)
    else:
        # Fall back to extracting from filename (YYYYMMDDHHMM)
        nowcast_filename = Path(nowcast_file_path).name
        filename_match = re.search(r'(\d{4}\d{2})\d{2}\d{4}', nowcast_filename)
        if not filename_match:
            raise ValueError("Could not determine year/month from nowcast file path or filename for SATELLITE_DIR.")
        year_month_str = filename_match.group(1)
    SATELLITE_DIR = f"/dmidata/projects/weather2x/Energivejr_historical_data/KNMI_MSGCPP_reproj_NW_EUROPE_SATELLITE/{year_month_str}"

    print("--- Loading Datasets ---")
    # 1. Nowcast data
    print(f"Loading nowcast data from: {nowcast_file_path}")
    ds_nowcast = xr.open_dataset(nowcast_file_path)
    if 'ensemble' in ds_nowcast.dims and ds_nowcast.sizes['ensemble'] == 1:
        ds_nowcast = ds_nowcast.squeeze('ensemble')

    # 2. Satellite data / time selection
    nowcast_filename = Path(nowcast_file_path).name
    match = re.search(r'(\d{8})(\d{4})', nowcast_filename)
    if not match:
        raise ValueError("Could not extract date and time from nowcast filename.")
    yyyymmdd, hhmm = match.groups()
    year, month, day = yyyymmdd[:4], yyyymmdd[4:6], yyyymmdd[6:8]
    date_str_hyphen = f"{year}-{month}-{day}"
    nowcast_init_time = datetime.datetime.strptime(f"{date_str_hyphen}T{hhmm[:2]}:{hhmm[2:]}:00", '%Y-%m-%dT%H:%M:%S')
    
    print(f"Inferred date: {date_str_hyphen} at {hhmm[:2]}:{hhmm[2:]}")

    # Satellite time window is the master timeline
    forecast_start_time = nowcast_init_time + datetime.timedelta(minutes=15)
    forecast_end_time = forecast_start_time + datetime.timedelta(hours=3)
    print(f"Satellite time window: {forecast_start_time} to {forecast_end_time}")

    # 1a. Load only the DINI initialization we need
    forecast_ref_time = (forecast_start_time - datetime.timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Selecting DINI data for forecast reference time: {forecast_ref_time}")
    ds_dini_source = xr.open_dataset(DINI_ZARR_PATH, engine='zarr', chunks={})
    ds_dini = ds_dini_source[['grad', 'latitude', 'longitude', 'forecast_reference_time', 'step']].sel(
        forecast_reference_time=forecast_ref_time, method='nearest'
    )
    ds_dini = (ds_dini['grad'] / 3600).to_dataset(name='grad')
    ds_dini = ds_dini.assign_coords(latitude=ds_dini_source.latitude, longitude=ds_dini_source.longitude)

    if 'step' in ds_dini.dims:
        ds_dini = ds_dini.rename({'step': 'time'})
        ds_dini['time'] = ds_dini_source['forecast_reference_time'].sel(forecast_reference_time=forecast_ref_time, method='nearest').values + ds_dini['time'].values
    
    # Convert accumulated values to timestep increments
    ds_dini = ds_dini.diff('time').fillna(0)

    # Filter satellite files to the forecast time window
    sat_files_pattern = f"{SATELLITE_DIR}/NetCDF4_sds_{year}-{month}-{day}T*.nc"
    all_sat_files = sorted(glob.glob(sat_files_pattern))
    
    sat_files = []
    for f in all_sat_files:
        name = Path(f).name
        ts_match = re.search(r'_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)\.nc', name)
        if ts_match:
            file_time_str = ts_match.group(1).replace('_', ':').replace('Z', '')
            file_time = datetime.datetime.fromisoformat(file_time_str)
            if forecast_start_time <= file_time < forecast_end_time:
                sat_files.append(f)

    if not sat_files:
        raise FileNotFoundError(f"No satellite files found for the time window in {SATELLITE_DIR} with pattern {sat_files_pattern}")
    
    print(f"Found {len(sat_files)} satellite files for the time window.")
    
    # Manually open and concatenate files to avoid memory issues with open_mfdataset
    datasets_to_concat = []
    for f in sat_files:
        ds = xr.open_dataset(f, chunks={'time': 1})
        datasets_to_concat.append(ds)
    
    ds_sat = xr.concat(datasets_to_concat, dim="time", data_vars="minimal")
    print("Satellite data loaded.")

    # Get satellite projection from the 'crs' variable attributes
    with xr.open_dataset(sat_files[0]) as first_sat_ds:
        crs_attrs = first_sat_ds['crs'].attrs
        # Extract the proj4 string directly to avoid parsing errors
        proj4_string = crs_attrs.get('proj4_params')
        if not proj4_string:
            raise ValueError("Could not find 'proj4_params' in the satellite file's 'crs' attributes.")
        sat_proj = ccrs.Projection(proj4_string)

    # --- Animation Setup ---
    print("--- Setting up Animation ---")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.patch.set_facecolor('#0B1220')

    lon_min_dini = float(ds_dini['longitude'].min().compute())
    lon_max_dini = float(ds_dini['longitude'].max().compute())
    lat_min_dini = float(ds_dini['latitude'].min().compute())
    lat_max_dini = float(ds_dini['latitude'].max().compute())

    lon_min_nowcast = float(ds_nowcast['lon'].min().compute())
    lon_max_nowcast = float(ds_nowcast['lon'].max().compute())
    lat_min_nowcast = float(ds_nowcast['lat'].min().compute())
    lat_max_nowcast = float(ds_nowcast['lat'].max().compute())

    # Transform satellite x/y corners to lon/lat to find extent
    sat_x = ds_sat.x.values
    sat_y = ds_sat.y.values
    transformer = Transformer.from_crs(sat_proj, ccrs.PlateCarree(), always_xy=True)
    # Transform corners of the grid
    lon_corners, lat_corners = transformer.transform(
        [sat_x.min(), sat_x.max(), sat_x.min(), sat_x.max()],
        [sat_y.min(), sat_y.min(), sat_y.max(), sat_y.max()]
    )
    lon_min_sat, lon_max_sat = min(lon_corners), max(lon_corners)
    lat_min_sat, lat_max_sat = min(lat_corners), max(lat_corners)

    common_extent = [
        max(lon_min_dini, lon_min_nowcast, lon_min_sat),
        min(lon_max_dini, lon_max_nowcast, lon_max_sat),
        max(lat_min_dini, lat_min_nowcast, lat_min_sat),
        min(lat_max_dini, lat_max_nowcast, lat_max_sat)
    ]

    titles = ['DINI (NWP Forecast)', 'Sunflow (Nowcast)', 'MSG CPP (Satellite)']
    datasets = [ds_dini, ds_nowcast, ds_sat]
    data_vars = ['grad', 'probabilistic_advection', 'sds']
    var_name = ['GHI [W/m2]', 'GHI [W/m2]', 'GHI [W/m2]']
    # Slice data to the 3-hour satellite window
    dini_sliced = ds_dini.sel(time=slice(forecast_start_time, forecast_end_time))
    nowcast_sliced = ds_nowcast.sel(time=slice(forecast_start_time, forecast_end_time))
    sat_sliced = ds_sat.sel(time=slice(forecast_start_time, forecast_end_time))

    datasets_sliced = [dini_sliced, nowcast_sliced, sat_sliced]

    # Total frames follow the 15-min satellite cadence over 3 hours
    num_frames = min(len(sat_sliced.time), len(nowcast_sliced.time))
    print(f"Animating {num_frames} frames over a 3-hour window.")

    def setup_ax(ax, title):
        ax.set_facecolor('#0E1A2B')
        ax.set_extent(common_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#0E1A2B', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, edgecolor='white', zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4, edgecolor='white', zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 8, 'color': 'white'}
        ax.set_title(title, fontsize=16, color='white', pad=10)

    ims = []
    for i, ax in enumerate(axes):
        setup_ax(ax, titles[i])
        ds = datasets_sliced[i]
        data_var = data_vars[i]
        
        # Determine transform and coordinates based on dataset
        transform = ccrs.PlateCarree()
        if 'x' in ds.coords and 'y' in ds.coords and 'sds' in ds.data_vars: # Satellite data signature
            lon, lat = ds['x'].values, ds['y'].values
            transform = sat_proj
        elif 'latitude' in ds.coords and 'longitude' in ds.coords: # DINI data signature
            lon, lat = ds['longitude'].values, ds['latitude'].values
        else: # Fallback for Nowcast data
            lon, lat = np.meshgrid(ds['lon'], ds['lat'])
        
        vmin = 0 #float(np.nanpercentile(ds[data_var], 2))
        vmax = 850 #float(np.nanpercentile(ds[data_var], 98))
        
        first_frame = ds[data_var].isel(time=0).values
        im = ax.pcolormesh(lon, lat, first_frame, cmap='gist_heat', vmin=vmin, vmax=vmax, transform=transform)
        ims.append(im)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(var_name[i], color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    def animate(frame_index):
        print(f"Processing frame {frame_index+1}/{num_frames}")
        
        # DINI (1-hour steps), updates every 4 frames
        dini_frame_index = frame_index // 4
        if dini_frame_index < len(dini_sliced.time):
            for coll in list(axes[0].collections):
                if isinstance(coll, QuadMesh): coll.remove()
            data = dini_sliced['grad'].isel(time=dini_frame_index).values
            lon, lat = dini_sliced['longitude'].values, dini_sliced['latitude'].values
            vmin = 0 #float(np.nanpercentile(dini_sliced['grad'], 2))
            vmax = 850 #float(np.nanpercentile(dini_sliced['grad'], 98))
            pc = axes[0].pcolormesh(lon, lat, data, cmap='gist_heat', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            tstamp = dini_sliced.time.isel(time=dini_frame_index).values
            axes[0].set_title(f"{titles[0]} {np.datetime_as_string(tstamp, unit='s')}", fontsize=12, color='white', pad=8)
            ims[0] = pc

        # Nowcast (15-min steps)
        if frame_index < len(nowcast_sliced.time):
            for coll in list(axes[1].collections):
                if isinstance(coll, QuadMesh): coll.remove()
            data = nowcast_sliced['probabilistic_advection'].isel(time=frame_index).values
            lon, lat = np.meshgrid(nowcast_sliced['lon'], nowcast_sliced['lat'])
            vmin = 0 #float(np.nanpercentile(nowcast_sliced['probabilistic_advection'], 2))
            vmax = 850 #float(np.nanpercentile(nowcast_sliced['probabilistic_advection'], 98))
            pc = axes[1].pcolormesh(lon, lat, data, cmap='gist_heat', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            tstamp = nowcast_sliced.time.isel(time=frame_index).values
            axes[1].set_title(f"{titles[1]} {np.datetime_as_string(tstamp, unit='s')}", fontsize=12, color='white', pad=8)
            ims[1] = pc

        # Satellite (15-min steps)
        if frame_index < len(sat_sliced.time):
            for coll in list(axes[2].collections):
                if isinstance(coll, QuadMesh): coll.remove()
            data = sat_sliced['sds'].isel(time=frame_index).values
            lon, lat = sat_sliced['x'].values, sat_sliced['y'].values
            vmin = 0 #float(np.nanpercentile(sat_sliced['sds'], 2))
            vmax = 850 #float(np.nanpercentile(sat_sliced['sds'], 98))
            pc = axes[2].pcolormesh(lon, lat, data, cmap='gist_heat', vmin=vmin, vmax=vmax, transform=sat_proj)
            tstamp = sat_sliced.time.isel(time=frame_index).values
            axes[2].set_title(f"{titles[2]} {np.datetime_as_string(tstamp, unit='s')}", fontsize=12, color='white', pad=8)
            ims[2] = pc
            
        return ims

    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=500, blit=False)

    output_gif = 'comparison_1045.gif'
    print(f"--- Saving GIF to {output_gif} ---")
    ani.save(output_gif, writer='pillow', fps=1, dpi=300)
    plt.close(fig)
    print("--- Done ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comparison GIF of weather forecast data.")
    parser.add_argument(
        "nowcast_file",
        nargs='?',
        default="/dmidata/projects/weather2x/Energivejr_historical_data/sunflow_validation_output/v1.0.0/202504/SolarNowcast_202504101045.nc",
        help="Path to the nowcast NetCDF file."
    )
    args = parser.parse_args()
    
    create_comparison_gif(args.nowcast_file)
