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

    date_tag = nwc_start.strftime('%Y%m%d')

    print(f"\n{'='*60}")
    print(f"  Validation run")
    print(f"  Nowcasts : {nwc_start}  →  {nwc_end}")
    print(f"  Output   : {output_dir}/")
    print(f"{'='*60}\n")

    def _skip_day(message: str) -> None:
        print(f"  SKIP: {message}")
        print("  Finished in 0.0 min")
        print(f"{'='*60}\n")
        return

    # ------------------------------------------------------------------
    # 1. Load nowcasts
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("Step 1/4 — Loading nowcasts...")
    nwc_loader = SatelliteNowcastLoader(data_dir=args.nwc_dir)
    try:
        nowcast_ds = nwc_loader.load_data(nwc_start, nwc_end)
    except ValueError as exc:
        _skip_day(str(exc))
        return
    print(f"  Loaded nowcast init range: {nowcast_ds.initialization_time.min().values} to {nowcast_ds.initialization_time.max().values}")
    print(f"  Loaded nowcast valid_time range: {nowcast_ds.valid_time.min().values} to {nowcast_ds.valid_time.max().values}")
    nowcast_ds = nowcast_ds.chunk({"initialization_time": 1, "lead_time": 1, "lat": 64, "lon": 64})
    print(f"  {nowcast_ds.sizes['initialization_time']} runs × "
          f"{nowcast_ds.sizes['lead_time']} lead steps loaded "
          f"({time.perf_counter() - t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # 2. Load observations (only matching nowcast valid_time range)
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    print("Step 2/4 — Loading observations...")
    obs_start = pd.Timestamp(nowcast_ds.valid_time.min().values)
    obs_end = pd.Timestamp(nowcast_ds.valid_time.max().values)
    # Keep a tiny safety margin for rounding issues
    obs_start -= pd.Timedelta(minutes=15)
    obs_end += pd.Timedelta(minutes=15)
    obs_loader = SatelliteObservationLoader(data_dir=args.obs_dir)
    try:
        obs_ds = obs_loader.load_data(obs_start, obs_end)
    except ValueError as exc:
        _skip_day(str(exc))
        return
    print(f"  Loaded obs time range: {obs_ds.time.min().values} to {obs_ds.time.max().values}")

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

    # Persisting aligned data can be expensive for large datasets.
    # Instead keep it lazy and compute only when saving.
    # scorer.aligned_data = scorer.aligned_data.persist()

    print(f"  Done ({time.perf_counter() - t2:.1f}s)\n")

    # ------------------------------------------------------------------
    # 4. Calculate scores and save
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    print("Step 4/4 — Calculating scores and saving...")

    # Compute the 2D by-init outputs we actually need:
    # one row per (initialization_time, lead_time) pair.
    mae_by_init = scorer.calculate_mae_by_init().compute()
    rmse_by_init = scorer.calculate_rmse_by_init().compute()

    scores_ds = xr.Dataset({
        "mae_by_init": mae_by_init,
        "rmse_by_init": rmse_by_init,
    })

    # Save as a tidy CSV table so downstream plotting can pivot or line-plot easily.
    out_path = Path(args.output_dir) / f"scores_{date_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing by-init CSV to {out_path}")
    df = scores_ds.to_dataframe().reset_index()
    df["lead_time_minutes"] = (df["lead_time"] / np.timedelta64(1, "m")).astype("int32")
    df = df.drop(columns=["lead_time"])
    df.to_csv(out_path, index=False)

    print(f"  ALL METRICS → {out_path}")
    print(f"  ({time.perf_counter() - t3:.1f}s)\n")

    total = time.perf_counter() - t0
    print(f"{'='*60}")
    print(f"  Finished in {total/60:.1f} min")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
