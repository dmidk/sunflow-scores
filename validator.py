"""
Satellite Nowcast Validation Framework
=======================================
Classes:
    SatelliteNowcastLoader     – loads nowcast NetCDF files into xarray
    SatelliteObservationLoader – loads satellite observation NetCDF files
    ScoreCalculator            – aligns the two datasets and computes MAE / RMSE

File naming conventions:
    Nowcasts:     SolarNowcast_YYYYMMDDHHMM.nc
    Observations: NetCDF4_sds_YYYY-MM-DDTHH_MM_SSZ.nc
"""

import xarray as xr
import pandas as pd
import numpy as np
import scores.continuous
from pathlib import Path


# =============================================================================
# 1. NOWCAST LOADER
# =============================================================================

class SatelliteNowcastLoader:
    """Loads solar irradiance nowcast files into a single xarray Dataset."""

    def __init__(self, data_dir: str = "/dmidata/projects/energivejr/nowcasts"):
        self.data_dir = Path(data_dir)

    def _preprocess_nowcast(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Prepare a single nowcast file for concatenation.

        Each file's 'time' coordinate holds valid datetimes (e.g. 10:30, 10:45 …
        for a 10:15 init run). This function:
          - computes lead_time = valid_time − initialization_time
          - replaces the 'time' dim with 'lead_time' so all files share the same axis
          - adds 'initialization_time' as a new scalar dim for stacking
          - stores valid_time as a 2D auxiliary coordinate for later reference
        """
        fname = Path(ds.encoding["source"]).name
        initialization_time = pd.to_datetime(
            fname.split("_")[1].replace(".nc", ""), format="%Y%m%d%H%M"
        )

        valid_times = ds["time"].values
        lead_times  = valid_times - initialization_time.to_datetime64()

        ds = ds.assign_coords(time=("time", lead_times))
        ds = ds.rename({"time": "lead_time"})
        ds = ds.expand_dims(initialization_time=[initialization_time.to_datetime64()])
        ds = ds.assign_coords(
            valid_time=(["initialization_time", "lead_time"], [valid_times])
        )
        return ds

    def load_data(self, start_date, end_date) -> xr.Dataset:
        """
        Load all nowcast files with initialization_time in [start_date, end_date].

        Returns a lazy Dataset with dims (initialization_time, lead_time, lat, lon).
        """
        start_date = pd.Timestamp(start_date)
        end_date   = pd.Timestamp(end_date)

        files_to_open = [
            str(self.data_dir / f"SolarNowcast_{ts.strftime('%Y%m%d%H%M')}.nc")
            for ts in pd.date_range(start=start_date, end=end_date, freq="15min")
            if (self.data_dir / f"SolarNowcast_{ts.strftime('%Y%m%d%H%M')}.nc").exists()
        ]

        if not files_to_open:
            raise ValueError(f"No nowcast files found between {start_date} and {end_date}.")

        print(f"  Found {len(files_to_open)} nowcast files.")

        return xr.open_mfdataset(
            files_to_open,
            combine="nested",
            concat_dim="initialization_time",
            preprocess=self._preprocess_nowcast,
            engine="h5netcdf",
            parallel=True,
        )


# =============================================================================
# 2. OBSERVATION LOADER
# =============================================================================

class SatelliteObservationLoader:
    """Loads satellite observation (ground-truth) files into a single xarray Dataset."""

    def __init__(self, data_dir: str = "/dmidata/projects/energivejr/satellite_data"):
        self.data_dir = Path(data_dir)

    def _preprocess_observation(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop the 'crs' scalar variable, which conflicts when concatenating files."""
        if "crs" in ds:
            ds = ds.drop_vars("crs")
        return ds

    def load_data(self, start_date, end_date) -> xr.Dataset:
        """
        Load observation files covering [start_date, end_date].

        Returns a Dataset with dim 'time' at 15-minute resolution.
        """
        start_date = pd.Timestamp(start_date)
        end_date   = pd.Timestamp(end_date)

        files_to_open = [
            str(fpath)
            for ts in pd.date_range(start=start_date, end=end_date, freq="15min")
            if (fpath := self.data_dir / f"NetCDF4_sds_{ts.strftime('%Y-%m-%dT%H_%M_%SZ')}.nc").exists()
        ]

        if not files_to_open:
            raise ValueError(f"No observation files found between {start_date} and {end_date}.")

        ds = xr.open_mfdataset(
            files_to_open,
            combine="by_coords",
            preprocess=self._preprocess_observation,
            engine="h5netcdf",
            parallel=True,
        )
        return ds.sel(time=slice(str(start_date), str(end_date))).sortby("time")


# =============================================================================
# 3. SCORE CALCULATOR
# =============================================================================

class ScoreCalculator:
    """Aligns nowcast and observation datasets and computes validation scores."""

    def __init__(self, nowcast_data: xr.Dataset, observation_data: xr.Dataset):
        self.nowcast_data     = nowcast_data
        self.observation_data = observation_data
        self.aligned_data     = None

    def align_data(self) -> xr.Dataset:
        """
        Match each nowcast (initialization_time, lead_time) step to its
        corresponding observation by valid_time.

        The result is stored in self.aligned_data as a Dataset with variables
        'nowcast' and 'observation', both with dims
        (initialization_time, lead_time, lat, lon).
        """
        # Use ensemble member 0 of the nowcast (deterministic baseline)
        nwc = self.nowcast_data["probabilistic_advection"].isel(ensemble=0)

        # valid_time is a 2D coordinate (n_init × n_lead) holding the wall-clock
        # UTC time of every forecast step. Flatten it for a single vectorised lookup.
        flat_valid_times = self.nowcast_data.valid_time.values.ravel()

        obs_flat = (
            self.observation_data["sds"]
            .sel(time=flat_valid_times, method="nearest")
            .rename({"y": "lat", "x": "lon"})
        )

        n_init = self.nowcast_data.sizes["initialization_time"]
        n_lead = self.nowcast_data.sizes["lead_time"]
        n_lat  = self.nowcast_data.sizes["lat"]
        n_lon  = self.nowcast_data.sizes["lon"]

        obs_da = xr.DataArray(
            obs_flat.values.reshape(n_init, n_lead, n_lat, n_lon),
            dims=["initialization_time", "lead_time", "lat", "lon"],
            coords={
                "initialization_time": self.nowcast_data.initialization_time,
                "lead_time":           self.nowcast_data.lead_time,
                "lat":                 self.nowcast_data.lat,
                "lon":                 self.nowcast_data.lon,
            },
        )

        self.aligned_data = xr.Dataset({"nowcast": nwc, "observation": obs_da})
        print("Data alignment complete.")
        return self.aligned_data

    def calculate_mae(self) -> xr.DataArray:
        """
        Mean Absolute Error averaged over all initialization_times.
        Returns an array with dims (lead_time, lat, lon).
        """
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        mae = scores.continuous.mae(
            self.aligned_data["nowcast"],
            self.aligned_data["observation"],
            reduce_dims="initialization_time",
        )
        mae.name = "mae"
        return mae

    def calculate_rmse(self) -> xr.DataArray:
        """
        Root Mean Squared Error averaged over all initialization_times.
        Returns an array with dims (lead_time, lat, lon).
        """
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        rmse = scores.continuous.rmse(
            self.aligned_data["nowcast"],
            self.aligned_data["observation"],
            reduce_dims="initialization_time",
        )
        rmse.name = "rmse"
        return rmse