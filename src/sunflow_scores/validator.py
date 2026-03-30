"""
Satellite Nowcast Score Framework
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
import argparse


# =============================================================================
# 1. NOWCAST LOADER
# =============================================================================

class SatelliteNowcastLoader:
    """Loads solar irradiance nowcast files into a single xarray Dataset."""

    def __init__(self, data_dir: str):
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

        def _find_files(start_dt, end_dt):
            return [
                str(self.data_dir / f"SolarNowcast_{ts.strftime('%Y%m%d%H%M')}.nc")
                for ts in pd.date_range(start=start_dt, end=end_dt, freq="15min")
                if (self.data_dir / f"SolarNowcast_{ts.strftime('%Y%m%d%H%M')}.nc").exists()
            ]

        files_to_open = _find_files(start_date, end_date)

        # Allow provided end to be slightly beyond available data (e.g. 00:00 boundary).
        if not files_to_open and end_date > start_date:
            candidate_end = end_date
            while candidate_end > start_date:
                candidate_end -= pd.Timedelta(minutes=15)
                files_to_open = _find_files(start_date, candidate_end)
                if files_to_open:
                    print(
                        f"  WARNING: no nowcast file found at end={end_date}. "
                        f"Using end={candidate_end} instead ({len(files_to_open)} files)."
                    )
                    end_date = candidate_end
                    break

        if not files_to_open:
            existing_files = sorted(self.data_dir.glob("SolarNowcast_*.nc"))
            if existing_files:
                available_times = [
                    pd.to_datetime(f.name.split("_")[1].replace(".nc", ""), format="%Y%m%d%H%M")
                    for f in existing_files
                ]
                earliest = min(available_times)
                latest = max(available_times)
                print(f"  WARNING: no nowcast files in requested interval.")
                print(f"  Available data ranges from {earliest} to {latest}.")
                # Clip requested range
                adjusted_start = max(start_date, earliest)
                adjusted_end = min(end_date, latest)
                if adjusted_start > adjusted_end:
                    raise ValueError(
                        f"No nowcast files found between {start_date} and {end_date}. "
                        f"Available range: {earliest} to {latest}."
                    )
                files_to_open = _find_files(adjusted_start, adjusted_end)
                if files_to_open:
                    print(
                        f"  Using adjusted nowcast range {adjusted_start} to {adjusted_end} "
                        f"({len(files_to_open)} files)."
                    )
                    start_date, end_date = adjusted_start, adjusted_end

        if not files_to_open:
            raise ValueError(f"No nowcast files found between {start_date} and {end_date}.")

        print(f"  Found {len(files_to_open)} nowcast files from {start_date} to {end_date}.")

        return xr.open_mfdataset(
            files_to_open,
            combine="nested",
            concat_dim="initialization_time",
            preprocess=self._preprocess_nowcast,
            engine="h5netcdf",
            parallel=True,
            chunks={
                "initialization_time": 4,
                "lead_time": 25,
                "lat": 128,
                "lon": 128,
            },
        )


# =============================================================================
# 2. OBSERVATION LOADER
# =============================================================================

class SatelliteObservationLoader:
    """Loads satellite observation (ground-truth) files into a single xarray Dataset."""

    def __init__(self, data_dir: str):
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

        all_hours = pd.date_range(start=start_date.floor("h"), end=end_date.ceil("h"), freq="h")
        files_to_open = sorted({
            str(self.data_dir / f"NetCDF4_sds_{ts.strftime('%Y-%m-%dT%H_%M_%SZ')}.nc")
            for ts in all_hours
            if (self.data_dir / f"NetCDF4_sds_{ts.strftime('%Y-%m-%dT%H_%M_%SZ')}.nc").exists()
        })

        if not files_to_open:
            raise ValueError(f"No observation files found between {start_date} and {end_date}.")

        print(f"  Loading {len(files_to_open)} observation files from {start_date.date()} to {end_date.date()}")

        ds = xr.open_mfdataset(
            files_to_open,
            combine="by_coords",
            preprocess=self._preprocess_observation,
            engine="h5netcdf",
            parallel=True,            chunks={"time": 96, "lat": 128, "lon": 128},        )
        return ds.sel(time=slice(start_date, end_date)).sortby("time")


# =============================================================================
# 3. SCORE CALCULATOR
# =============================================================================

class ScoreCalculator:
    """Aligns nowcast and observation datasets and computes validation scores."""

    def __init__(self, nowcast_data: xr.Dataset, observation_data: xr.Dataset, nowcast_ghi_var: str, obs_ghi_var: str, obs_cs_ghi_var: str):
        self.nowcast_data = nowcast_data
        self.observation_data = observation_data
        self.nowcast_ghi_var = nowcast_ghi_var
        self.obs_ghi_var = obs_ghi_var
        self.obs_cs_ghi_var = obs_cs_ghi_var
        self.aligned_data = None
        self.kt_data = None

    def align_data(self, chunk_size: int = 100) -> xr.Dataset:
        """
        Match each nowcast (initialization_time, lead_time) step to its
        corresponding observation by valid_time. This version processes data in chunks
        to avoid loading everything into memory at once.

        The result is stored in self.aligned_data as a Dataset with variables
        'nowcast', 'observation', and 'clearsky', all with dims
        (initialization_time, lead_time, lat, lon).
        """
        nwc = self.nowcast_data[self.nowcast_ghi_var].isel(ensemble=0)
        
        num_inits = len(self.nowcast_data.initialization_time)
        aligned_chunks = []

        for i in range(0, num_inits, chunk_size):
            chunk_slice = slice(i, i + chunk_size)
            nowcast_chunk = nwc.isel(initialization_time=chunk_slice)
            valid_time_chunk = self.nowcast_data.valid_time.isel(initialization_time=chunk_slice)
            
            flat_valid_times = valid_time_chunk.values.ravel()
            
            obs_chunk = self.observation_data.sel(time=flat_valid_times, method="nearest").rename({"y": "lat", "x": "lon"})
            obs_flat = obs_chunk[self.obs_ghi_var]
            cs_flat = obs_chunk[self.obs_cs_ghi_var]


            n_init_chunk = nowcast_chunk.sizes["initialization_time"]
            n_lead = nowcast_chunk.sizes["lead_time"]
            n_lat  = nowcast_chunk.sizes["lat"]
            n_lon  = nowcast_chunk.sizes["lon"]

            obs_da_chunk = xr.DataArray(
                obs_flat.values.reshape(n_init_chunk, n_lead, n_lat, n_lon),
                dims=["initialization_time", "lead_time", "lat", "lon"],
                coords={
                    "initialization_time": nowcast_chunk.initialization_time,
                    "lead_time":           nowcast_chunk.lead_time,
                    "lat":                 nowcast_chunk.lat,
                    "lon":                 nowcast_chunk.lon,
                },
            )
            cs_da_chunk = xr.DataArray(
                cs_flat.values.reshape(n_init_chunk, n_lead, n_lat, n_lon),
                dims=["initialization_time", "lead_time", "lat", "lon"],
                coords={
                    "initialization_time": nowcast_chunk.initialization_time,
                    "lead_time":           nowcast_chunk.lead_time,
                    "lat":                 nowcast_chunk.lat,
                    "lon":                 nowcast_chunk.lon,
                },
            )
            
            aligned_chunk = xr.Dataset({
                "nowcast": nowcast_chunk, 
                "observation": obs_da_chunk,
                "clearsky": cs_da_chunk,
            })
            aligned_chunks.append(aligned_chunk)
            print(f"  Processed chunk {i//chunk_size + 1}/{(num_inits + chunk_size - 1)//chunk_size}")

        self.aligned_data = xr.concat(aligned_chunks, dim="initialization_time")
        # Add valid_time as a coordinate to the aligned data
        self.aligned_data = self.aligned_data.assign_coords(
            valid_time=(("initialization_time", "lead_time"), self.nowcast_data.valid_time.data)
        )
        print(f"  Data alignment complete.")
        return self.aligned_data

    def calculate_kt(self) -> xr.Dataset:
        """
        Calculates the clear-sky index (kt) for nowcasts and observations.
        The result is stored in self.kt_data.
        """
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")

        # Avoid division by zero or near-zero clearsky values
        clearsky = self.aligned_data["clearsky"].where(self.aligned_data["clearsky"] > 1e-6)
        
        kt_nowcast = (self.aligned_data["nowcast"] / clearsky).fillna(0)
        kt_observation = (self.aligned_data["observation"] / clearsky).fillna(0)

        self.kt_data = xr.Dataset({
            "nowcast": kt_nowcast,
            "observation": kt_observation,
        })
        self.kt_data = self.kt_data.assign_coords(
            valid_time=(("initialization_time", "lead_time"), self.aligned_data.valid_time.data)
        )
        print(f"  Clear-sky index (kt) calculation complete.")
        return self.kt_data

    def calculate_mae(self, data: xr.Dataset, groupby_time_of_day=False) -> xr.DataArray:
        """
        Mean Absolute Error.
        Grouped by valid_time hour: dims (hour, lead_time, lat, lon)
        Otherwise dims (lead_time, lat, lon).
        """
        if data is None:
            raise RuntimeError("Provide a dataset to calculate scores on.")

        if isinstance(data, xr.DataArray):
            # Already-aggregated time-series (e.g. mae_by_init) can be passed through.
            if groupby_time_of_day:
                raise ValueError("groupby_time_of_day is not supported for DataArray input")
            return data

        if groupby_time_of_day:
            abs_err = np.abs(data["nowcast"] - data["observation"])
            stacked = abs_err.stack(init_lead=("initialization_time", "lead_time"))
            stacked = stacked.assign_coords(
                valid_time=data.valid_time.stack(init_lead=("initialization_time", "lead_time"))
            )
            mae = stacked.groupby(stacked.valid_time.dt.hour).mean(dim="init_lead")
            mae.name = "mae_by_hour"
        else:
            mae = scores.continuous.mae(
                data["nowcast"],
                data["observation"],
                reduce_dims="initialization_time",
            )
            mae.name = "mae"
        return mae

    def calculate_rmse(self, data: xr.Dataset, groupby_time_of_day=False) -> xr.DataArray:
        """
        Root Mean Squared Error.
        Grouped by valid_time hour: dims (hour, lead_time, lat, lon)
        Otherwise dims (lead_time, lat, lon).
        """
        if data is None:
            raise RuntimeError("Provide a dataset to calculate scores on.")

        if isinstance(data, xr.DataArray):
            if groupby_time_of_day:
                raise ValueError("groupby_time_of_day is not supported for DataArray input")
            return data

        if groupby_time_of_day:
            sq_err = (data["nowcast"] - data["observation"])**2
            rmse = np.sqrt(sq_err.groupby(data.valid_time.dt.hour).mean(dim="initialization_time"))
            rmse.name = "rmse_by_hour"
        else:
            rmse = scores.continuous.rmse(
                data["nowcast"],
                data["observation"],
                reduce_dims="initialization_time",
            )
            rmse.name = "rmse"
        return rmse

    def calculate_mae_by_init(self) -> xr.DataArray:
        """MAE for each init time and lead_time (avg over spatial dims)."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        abs_err = np.abs(self.aligned_data["nowcast"] - self.aligned_data["observation"])
        out = abs_err.mean(dim=["lat", "lon"])
        out.name = "mae_by_init"
        return out

    def calculate_rmse_by_init(self) -> xr.DataArray:
        """RMSE for each init time and lead_time (avg over spatial dims)."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        sq_err = (self.aligned_data["nowcast"] - self.aligned_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim=["lat", "lon"]))
        out.name = "rmse_by_init"
        return out

    def calculate_mae_kt_by_init(self) -> xr.DataArray:
        """KT MAE for each init time and lead_time (avg over spatial dims)."""
        if self.kt_data is None:
            raise RuntimeError("Call .calculate_kt() first.")
        abs_err = np.abs(self.kt_data["nowcast"] - self.kt_data["observation"])
        out = abs_err.mean(dim=["lat", "lon"])
        out.name = "mae_kt_by_init"
        return out

    def calculate_rmse_kt_by_init(self) -> xr.DataArray:
        """KT RMSE for each init time and lead_time (avg over spatial dims)."""
        if self.kt_data is None:
            raise RuntimeError("Call .calculate_kt() first.")
        sq_err = (self.kt_data["nowcast"] - self.kt_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim=["lat", "lon"]))
        out.name = "rmse_kt_by_init"
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Nowcast Validation")
    parser.add_argument("nowcast_dir", type=str, help="Directory with nowcast files")
    parser.add_argument("obs_dir", type=str, help="Directory with observation files")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("output_file", type=str, help="Path to save the output scores NetCDF file")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of initialization times to process per chunk")
    parser.add_argument("--nowcast_ghi_var", type=str, default="probabilistic_advection", help="Name of the GHI variable in the nowcast files")
    parser.add_argument("--obs_ghi_var", type=str, default="sds", help="Name of the GHI variable in the observation files")
    parser.add_argument("--obs_cs_ghi_var", type=str, default="sds_cs", help="Name of the clear-sky GHI variable in the observation files")

    args = parser.parse_args()

    print("1. Loading data...")
    nowcast_loader = SatelliteNowcastLoader(data_dir=args.nowcast_dir)
    nowcast_data = nowcast_loader.load_data(args.start_date, args.end_date)

    obs_loader = SatelliteObservationLoader(data_dir=args.obs_dir)
    obs_data = obs_loader.load_data(args.start_date, args.end_date)

    print("\n2. Aligning data...")
    calculator = ScoreCalculator(
        nowcast_data, 
        obs_data,
        nowcast_ghi_var=args.nowcast_ghi_var,
        obs_ghi_var=args.obs_ghi_var,
        obs_cs_ghi_var=args.obs_cs_ghi_var
    )
    aligned_data = calculator.align_data(chunk_size=args.chunk_size)
    
    print("\n3. Calculating clear-sky index (kt)...")
    kt_data = calculator.calculate_kt()

    print("\n4. Calculating scores...")
    mae = calculator.calculate_mae(aligned_data)
    rmse = calculator.calculate_rmse(aligned_data)
    
    mae_kt = calculator.calculate_mae(kt_data)
    rmse_kt = calculator.calculate_rmse(kt_data)

    mae_by_init = calculator.calculate_mae_by_init()
    rmse_by_init = calculator.calculate_rmse_by_init()
    mae_kt_by_init = calculator.calculate_mae_kt_by_init()
    rmse_kt_by_init = calculator.calculate_rmse_kt_by_init()

    scores_ds = xr.Dataset({
        "mae": mae,
        "rmse": rmse,
        "mae_kt": mae_kt,
        "rmse_kt": rmse_kt,
        "mae_by_init": mae_by_init,
        "rmse_by_init": rmse_by_init,
        "mae_kt_by_init": mae_kt_by_init,
        "rmse_kt_by_init": rmse_kt_by_init,
    })

    print(f"\n5. Saving scores to {args.output_file}...")
    scores_ds.to_netcdf(args.output_file)

    print("\nValidation complete.")
