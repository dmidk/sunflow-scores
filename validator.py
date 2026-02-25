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
        Extracts time from filename and handles existing 'time' variables safely.
        """
        # 1. Clear out any existing 'time' to prevent the ValueError
        if 'time' in ds.variables or 'time' in ds.coords:
            # Renaming is safer than dropping in some NetCDF versions
            ds = ds.rename({'time': 'time_file_internal'})
        
        # 2. Get the filename
        filename = ds.encoding.get("source", "")
        
        # 3. Parse the date (Choose the logic based on which loader this is)
        # For Nowcasts:
        time_str = Path(filename).stem.split("_")[-1]
        dt = pd.to_datetime(time_str, format="%Y%m%d%H%M")
        
        # For Observations (uncomment this if editing the Obs loader):
        # time_str = Path(filename).stem.split("_sds_")[-1]
        # dt = pd.to_datetime(time_str, format="%Y-%m-%dT%H_%M_%SZ")
        
        # 4. Assign the new dimension
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
                engine="h5netcdf"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load NetCDF files: {e}")

        # 3. Filter the massive cube down to the dates the user requested
        # Xarray allows you to slice strings naturally!
        ds_filtered = ds.sel(time=slice(start_date, end_date))
        
        return ds_filtered
    


class SatelliteObservationLoader:
    def __init__(self, data_dir: str = "/dmidata/projects/energivejr/satellite_data"):
        self.data_dir = Path(data_dir)

    def _extract_time_from_filename(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extracts time from filename and handles existing 'time' variables safely.
        """
        # 1. Clear out any existing 'time' to prevent the ValueError
        if 'time' in ds.variables or 'time' in ds.coords:
            # Renaming is safer than dropping in some NetCDF versions
            ds = ds.rename({'time': 'time_file_internal'})
        
        # 2. Get the filename
        filename = ds.encoding.get("source", "")
        
        # 3. Parse the date (Choose the logic based on which loader this is)
        # For Nowcasts:
        time_str = Path(filename).stem.split("_")[-1]
        dt = pd.to_datetime(time_str, format="%Y%m%d%H%M")
        
        # For Observations (uncomment this if editing the Obs loader):
        # time_str = Path(filename).stem.split("_sds_")[-1]
        # dt = pd.to_datetime(time_str, format="%Y-%m-%dT%H_%M_%SZ")
        
        # 4. Assign the new dimension
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
                engine="h5netcdf"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load observation NetCDF files: {e}")

        # 1. Slice by requested dates
        ds_filtered = ds.sel(time=slice(start_date, end_date))
        
        # 2. Sort by time (OS file systems don't always read files in chronological order!)
        ds_filtered = ds_filtered.sortby("time")
        
        return ds_filtered

class ScoresCalculator:
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