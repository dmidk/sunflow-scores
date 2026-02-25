import xarray as xr
import pandas as pd
import scores
from pathlib import Path
from typing import List
from typing import Optional

class SatelliteNowcastLoader:
    def __init__(self, data_dir: str = "/dmidata/projects/energivejr/nowcasts"):
        self.data_dir = Path(data_dir)

    def _extract_time_from_filename(self, ds: xr.Dataset) -> xr.Dataset:
        """
        A helper function just in case your NetCDF files only contain 2D (lat/lon) 
        data and don't explicitly have a 'time' dimension built inside them.
        This reads the filename and assigns the time coordinate.
        """
        # Get the filename from the encoding dictionary xarray creates
        filename = ds.encoding.get("source", "")
        
        # Extract the YYYYMMDDHHMM part (e.g., '202601021215')
        time_str = Path(filename).stem.split("_")[-1]
        
        # Convert to a pandas datetime object
        dt = pd.to_datetime(time_str, format="%Y%m%d%H%M")
        
        # Expand the 2D dataset to 3D by adding the time dimension
        return ds.assign_coords(time=dt).expand_dims("time")

    def load_data(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Lazily loads the nowcast NetCDF files within a specific date range.
        Format for dates: 'YYYY-MM-DD'
        """
        # 1. Grab only files matching the strict pattern (ignores t_2026...)
        file_pattern = str(self.data_dir / "SolarNowcast_*.nc")
        
        # 2. Lazily open all files and stitch them together along the time dimension
        try:
            ds = xr.open_mfdataset(
                file_pattern,
                combine="nested",
                concat_dim="time",
                preprocess=self._extract_time_from_filename, # Assigns time from filename
                parallel=True, # Uses dask to speed up reading
                engine="netcdf4"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load NetCDF files: {e}")

        # 3. Filter the massive cube down to the dates the user requested
        # Xarray allows you to slice strings naturally!
        ds_filtered = ds.sel(time=slice(start_date, end_date))
        
        return ds_filtered
    


def evaluate_nowcast(obs_ds: xr.Dataset, fcst_ds: xr.Dataset, ghi_threshold: float):
    """
    Evaluates a satellite GHI nowcast against observed GHI satellite images.
    Expects xarray Datasets with matching 'time', 'lat', and 'lon' coordinates.
    """
    
    # 1. Standard Continuous Metrics (Point-by-point)
    # Good for general bias, but harsh on spatial displacement
    mae = scores.continuous.mae(fcst_ds['GHI'], obs_ds['GHI'], preserve_dims=['time'])
    rmse = scores.continuous.rmse(fcst_ds['GHI'], obs_ds['GHI'], preserve_dims=['time'])
    
    # 2. Spatial Metrics (Fractions Skill Score)
    # Converts GHI to a binary threshold (e.g., "Is GHI > 300 W/m2?") 
    # and compares neighborhoods.
    obs_binary = obs_ds['GHI'] > ghi_threshold
    fcst_binary = fcst_ds['GHI'] > ghi_threshold
    
    # Calculate FSS over a specified spatial window (e.g., 5x5 pixels)
    # Note: Window sizes depend on your satellite grid resolution
    fss = scores.spatial.fss(
        fcst=fcst_binary, 
        obs=obs_binary, 
        window_size=[5, 5], 
        spatial_dims=['lat', 'lon']
    )
    
    # Combine results into a single dataset for easy plotting in Streamlit
    results = xr.Dataset({
        'MAE': mae,
        'RMSE': rmse,
        'FSS': fss
    })
    
    return results

class SatelliteObservationLoader:
    def __init__(self, data_dir: str = "/dmidata/projects/energivejr/satellite_data"):
        self.data_dir = Path(data_dir)

    def _extract_time_from_filename(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extracts time from filenames like NetCDF4_sds_2026-02-24T08_15_00Z.nc
        """
        filename = ds.encoding.get("source", "")
        
        # Extract the '2026-02-24T08_15_00Z' part
        # Path(filename).stem removes the '.nc'
        time_str = Path(filename).stem.split("_sds_")[-1]
        
        # Parse the datetime using pandas
        dt = pd.to_datetime(time_str, format="%Y-%m-%dT%H_%M_%SZ")
        
        # Strip timezone info to keep it naive (assuming your nowcasts are also naive UTC)
        # This prevents xarray from silently failing during alignment
        if dt.tzinfo is not None:
            dt = dt.tz_localize(None)
            
        return ds.assign_coords(time=dt).expand_dims("time")

    def load_data(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Lazily loads the ground truth satellite files.
        """
        file_pattern = str(self.data_dir / "NetCDF4_sds_*.nc")
        
        try:
            ds = xr.open_mfdataset(
                file_pattern,
                combine="nested",
                concat_dim="time",
                preprocess=self._extract_time_from_filename,
                parallel=True, 
                engine="netcdf4"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load observation NetCDF files: {e}")

        # 1. Slice by requested dates
        ds_filtered = ds.sel(time=slice(start_date, end_date))
        
        # 2. Sort by time (OS file systems don't always read files in chronological order!)
        ds_filtered = ds_filtered.sortby("time")
        
        return ds_filtered

class NowcastEvaluator:
    def __init__(self, obs_ds: xr.Dataset, fcst_ds: xr.Dataset):
        """
        Initializes the evaluator with observation and forecast datasets.
        Ensures the datasets are aligned in time and space before doing any math.
        """
        # Align the datasets to ensure we only evaluate where we have both obs and fcst
        self.obs, self.fcst = xr.align(obs_ds, fcst_ds, join='inner')
        self.results = xr.Dataset()

    def calc_continuous_metrics(self) -> xr.Dataset:
        """Calculates point-by-point deterministic metrics."""
        mae = scores.continuous.mae(self.fcst['GHI'], self.obs['GHI'], preserve_dims=['time'])
        rmse = scores.continuous.rmse(self.fcst['GHI'], self.obs['GHI'], preserve_dims=['time'])
        bias = scores.continuous.mean_error(self.fcst['GHI'], self.obs['GHI'], preserve_dims=['time'])
        
        self.results = self.results.assign(MAE=mae, RMSE=rmse, Bias=bias)
        return self.results

    def calc_spatial_metrics(self, thresholds: List[float], window_sizes: List[List[int]]) -> xr.Dataset:
        """
        Calculates Fractions Skill Score (FSS) for different GHI thresholds and 
        neighborhood sizes.
        """
        for thresh in thresholds:
            obs_binary = self.obs['GHI'] > thresh
            fcst_binary = self.fcst['GHI'] > thresh
            
            for window in window_sizes:
                fss = scores.spatial.fss(
                    fcst=fcst_binary, 
                    obs=obs_binary, 
                    window_size=window, 
                    spatial_dims=['lat', 'lon']
                )
                
                # Dynamic naming: e.g., 'FSS_300W_5x5'
                var_name = f"FSS_{thresh}W_{window[0]}x{window[1]}"
                self.results = self.results.assign({var_name: fss})
                
        return self.results

    def run_all_metrics(self, thresholds=[300, 500], window_sizes=[[5,5], [11,11]]) -> xr.Dataset:
        """Convenience method to run the standard suite of validation metrics."""
        self.calc_continuous_metrics()
        self.calc_spatial_metrics(thresholds, window_sizes)
        return self.results