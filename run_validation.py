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

from validator import SatelliteNowcastLoader, SatelliteObservationLoader, ScoreCalculator

NWC_DIR = Path("/dmidata/projects/energivejr/nowcasts")
OBS_DIR = Path("/dmidata/projects/energivejr/satellite_data")


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
        "--nwc-dir", default=str(NWC_DIR),
        help=f"Directory containing nowcast files (default: {NWC_DIR})",
    )
    parser.add_argument(
        "--obs-dir", default=str(OBS_DIR),
        help=f"Directory containing observation files (default: {OBS_DIR})",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to write output NetCDF files (default: ./results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nwc_start = pd.Timestamp(args.start)
    nwc_end   = pd.Timestamp(args.end)

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
    print(f"  {obs_ds.sizes['time']} observation timesteps loaded "
          f"({time.perf_counter() - t1:.1f}s)\n")

    # ------------------------------------------------------------------
    # 3. Align and compute into memory
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    print("Step 3/4 — Aligning and computing (this is the slow step)...")
    scorer = ScoreCalculator(nowcast_ds, obs_ds)
    scorer.align_data()
    scorer.aligned_data = scorer.aligned_data.compute()
    print(f"  Done ({time.perf_counter() - t2:.1f}s)\n")

    # ------------------------------------------------------------------
    # 4. Calculate scores and save
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    print("Step 4/4 — Calculating scores and saving...")

    mae  = scorer.calculate_mae()
    rmse = scorer.calculate_rmse()

    mae_path  = output_dir / f"mae_{date_tag}.nc"
    rmse_path = output_dir / f"rmse_{date_tag}.nc"

    # NetCDF4 cannot store timedelta64 coordinates directly.
    # Convert lead_time to integer minutes; record the unit in an attribute
    # so it can be reconstructed with pd.to_timedelta(da.lead_time, unit="min").
    # Also drop any auxiliary coords (e.g. valid_time) that aren't plain numerics.
    def _prepare_for_save(da: xr.DataArray) -> xr.DataArray:
        lead_minutes = da.lead_time.values / np.timedelta64(1, "m")
        da = da.assign_coords(lead_time=lead_minutes.astype("int32"))
        da = da.assign_attrs({**da.attrs, "lead_time_units": "minutes"})
        aux_to_drop = [c for c in da.coords if c not in da.dims]
        if aux_to_drop:
            da = da.drop_vars(aux_to_drop)
        return da

    _prepare_for_save(mae).to_netcdf(mae_path, engine="h5netcdf")
    _prepare_for_save(rmse).to_netcdf(rmse_path, engine="h5netcdf")

    print(f"  MAE  → {mae_path}")
    print(f"  RMSE → {rmse_path}")
    print(f"  ({time.perf_counter() - t3:.1f}s)\n")

    total = time.perf_counter() - t0
    print(f"{'='*60}")
    print(f"  Finished in {total/60:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
