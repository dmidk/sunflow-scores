"""
run_validation.py — Batch Solar Nowcast Validation
====================================================
Loads nowcasts and observations for a given date range, computes MAE and RMSE
per lead time, and writes the results to NetCDF.

Usage:
    uv run python run_validation.py --start 2026-02-01 --end 2026-02-28

Output (written to --output-dir, default: ./results/):
    mae_<start>_<end>.nc   — MAE(lead_time, lat, lon)
    rmse_<start>_<end>.nc  — RMSE(lead_time, lat, lon)

    lead_time is stored as integer minutes. To restore timedeltas when loading:
        da = xr.open_dataarray("results/mae_....nc")
        da["lead_time"] = pd.to_timedelta(da.lead_time, unit="min")
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys

project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


from sunflow_scores.validator import SatelliteNowcastLoader, SatelliteObservationLoader, ScoreCalculator
import dask
dask.config.set(scheduler="threads")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate solar nowcasts against satellite observations."
    )
    parser.add_argument(
        "--start", required=True,
        help="First nowcast initialization time, e.g. 2026-02-01",
    )
    parser.add_argument(
        "--end", required=True,
        help="Last nowcast initialization time, e.g. 2026-02-28 23:45",
    )
    parser.add_argument(
        "--nwc-dir", required=True,
        help=f"Directory containing nowcast files",
    )
    parser.add_argument(
        "--obs-dir", required=True,
        help=f"Directory containing observation files",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to write output NetCDF files (default: ./results/)",
    )
    parser.add_argument("--nowcast_ghi_var", type=str, default="probabilistic_advection", help="Name of the GHI variable in the nowcast files")
    parser.add_argument("--obs_ghi_var", type=str, default="sds", help="Name of the GHI variable in the observation files")
    parser.add_argument("--obs_cs_ghi_var", type=str, default="sds_cs", help="Name of the clear-sky GHI variable in the observation files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nwc_start = pd.Timestamp(args.start)
    nwc_end   = pd.Timestamp(args.end)

    # If date-only is given (midnight), interpret as full day.
    if nwc_start == nwc_start.normalize():
        nwc_start = nwc_start.normalize()
    if nwc_end == nwc_end.normalize():
        nwc_end = nwc_end.normalize() + pd.Timedelta(hours=23, minutes=45)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_tag = f"{nwc_start.strftime('%Y%m%d')}_{nwc_end.strftime('%Y%m%d')}"

    print(f"\n{'='*60}")
    print(f"  Validation run")
    print(f"  Nowcasts : {nwc_start}  →  {nwc_end}")
    print(f"  Output   : {output_dir}/")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load nowcasts
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("Step 1/4 — Loading nowcasts...")
    nwc_loader = SatelliteNowcastLoader(data_dir=args.nwc_dir)
    nowcast_ds = nwc_loader.load_data(nwc_start, nwc_end)
    print(f"  Loaded nowcast init range: {nowcast_ds.initialization_time.min().values} to {nowcast_ds.initialization_time.max().values}")
    print(f"  Loaded nowcast valid_time range: {nowcast_ds.valid_time.min().values} to {nowcast_ds.valid_time.max().values}")
    nowcast_ds = nowcast_ds.chunk({"initialization_time": 1, "lead_time": 1, "lat": 64, "lon": 64})
    print(f"  {nowcast_ds.sizes['initialization_time']} runs × "
          f"{nowcast_ds.sizes['lead_time']} lead steps loaded "
          f"({time.perf_counter() - t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # 2. Load observations
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    print("Step 2/4 — Loading observations...")
    obs_start = nwc_start
    obs_end   = nwc_end + pd.Timedelta(nowcast_ds.lead_time.max().values)
    obs_loader = SatelliteObservationLoader(data_dir=args.obs_dir)
    obs_ds = obs_loader.load_data(obs_start, obs_end)

    # Determine spatial dims for chunking (lat/lon or y/x).
    obs_chunk = {"time": 1}
    if "lat" in obs_ds.dims and "lon" in obs_ds.dims:
        obs_chunk.update({"lat": 64, "lon": 64})
    elif "y" in obs_ds.dims and "x" in obs_ds.dims:
        obs_chunk.update({"y": 64, "x": 64})
    else:
        # Fallback: chunk only time
        pass

    obs_ds = obs_ds.chunk(obs_chunk)
    print(f"  {obs_ds.sizes['time']} observation timesteps loaded "
          f"({time.perf_counter() - t1:.1f}s)\n")

    # ------------------------------------------------------------------
    # 3. Align and compute into memory
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    print("Step 3/4 — Aligning and computing (this will take time)...")
    scorer = ScoreCalculator(
        nowcast_ds,
        obs_ds,
        nowcast_ghi_var=args.nowcast_ghi_var,
        obs_ghi_var=args.obs_ghi_var,
        obs_cs_ghi_var=args.obs_cs_ghi_var
    )
    scorer.align_data()

    # Persist aligned data to avoid repeated graph recompute
    scorer.aligned_data = scorer.aligned_data.persist()

    print(f"  Done ({time.perf_counter() - t2:.1f}s)\n")

    # ------------------------------------------------------------------
    # 4. Calculate scores and save
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    print("Step 4/4 — Calculating scores and saving...")

    kt_data = scorer.calculate_kt()

    scores_to_calculate = {
        "mae": (scorer.aligned_data, False),
        "rmse": (scorer.aligned_data, False),
        "mae_kt": (kt_data, False),
        "rmse_kt": (kt_data, False),
        "mae_by_init": (scorer.calculate_mae_by_init(), False),
        "rmse_by_init": (scorer.calculate_rmse_by_init(), False),
        "mae_kt_by_init": (scorer.calculate_mae_kt_by_init(), False),
        "rmse_kt_by_init": (scorer.calculate_rmse_kt_by_init(), False),
    }

    results = {}
    for name, (data, by_hour) in scores_to_calculate.items():
        if name in ("mae_by_init", "rmse_by_init", "mae_kt_by_init", "rmse_kt_by_init"):
            results[name] = data
        elif name.startswith("mae"):
            results[name] = scorer.calculate_mae(data, groupby_time_of_day=by_hour)
        elif name.startswith("rmse"):
            results[name] = scorer.calculate_rmse(data, groupby_time_of_day=by_hour)

    # NetCDF4 cannot store timedelta64 coordinates directly.
    # Convert lead_time to integer minutes; record the unit in an attribute
    # so it can be reconstructed with pd.to_timedelta(da.lead_time, unit="min").
    # Also drop any auxiliary coords (e.g. valid_time) that aren't plain numerics.
    def _prepare_for_save(da: xr.DataArray) -> xr.DataArray:
        if "lead_time" in da.coords:
            lead_minutes = da.lead_time.values / np.timedelta64(1, "m")
            da = da.assign_coords(lead_time=lead_minutes.astype("int32"))
            da = da.assign_attrs({**da.attrs, "lead_time_units": "minutes"})
        aux_to_drop = [c for c in da.coords if c not in da.dims]
        if aux_to_drop:
            da = da.drop_vars(aux_to_drop)
        return da

    for name, da in results.items():
        path = output_dir / f"{name}_{date_tag}.nc"
        print(f"  Writing {name.upper()} to {path}...", flush=True)
        _prepare_for_save(da).compute().to_netcdf(path, engine="h5netcdf")
        print(f"  {name.upper()} → {path}")

    print(f"  ({time.perf_counter() - t3:.1f}s)\n")

    total = time.perf_counter() - t0
    print(f"{'='*60}")
    print(f"  Finished in {total/60:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
