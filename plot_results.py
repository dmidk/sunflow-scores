"""
plot_results.py — Solar Nowcast Visualisation
==============================================
Produces two figures:

  1. Nowcast vs. Observation     
     A 2-row grid of maps (Nowcast | Observation) stepping through the first
     N lead times of a chosen initialization run.

  2. MAE and RMSE vs. forecast horizon
     Spatially-averaged scores loaded from the NetCDF files produced by
     run_validation.py.

Usage:
    uv run python plot_results.py --init "2026-02-26 11:00" --scores-start 2026-02-26 --scores-end 2026-02-27

    # Only plot the sequence (no scores files needed):
    uv run python plot_results.py --init "2026-02-26 11:00" --no-scores

    # Only plot the scores (no nowcast/obs loading needed):
    uv run python plot_results.py --scores-only --scores-start 2026-02-26 --scores-end 2026-02-27
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr

from validator import SatelliteNowcastLoader, SatelliteObservationLoader

NWC_DIR     = Path("/dmidata/projects/energivejr/nowcasts")
OBS_DIR     = Path("/dmidata/projects/energivejr/satellite_data")
RESULTS_DIR = Path("results")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot nowcast vs. observations and/or validation scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- sequence plot ---
    p.add_argument("--init", default=None,
                   help="Initialization time to plot, e.g. '2026-02-26 11:00'")
    p.add_argument("--n-steps", type=int, default=4,
                   help="Number of lead-time steps to show in the sequence plot")
    p.add_argument("--step-every", type=int, default=1,
                   help="Stride between plotted lead steps (1 = every 15 min)")
    p.add_argument("--vmin", type=float, default=0,   help="Colourbar minimum (W/m²)")
    p.add_argument("--vmax", type=float, default=900, help="Colourbar maximum (W/m²)")
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   help="Zoom to bounding box, e.g. --bbox 7.5 54.5 13.0 58.0")
    p.add_argument("--show-diff", action="store_true",
                   help="Add a third row showing Nowcast − Observation difference")

    # --- scores plot ---
    p.add_argument("--scores-start", default=None,
                   help="Start date tag of the scores file, e.g. 2026-02-26")
    p.add_argument("--scores-end", default=None,
                   help="End date tag of the scores file, e.g. 2026-02-27")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help="Directory containing MAE/RMSE NetCDF files")

    # --- data dirs ---
    p.add_argument("--nwc-dir", default=str(NWC_DIR))
    p.add_argument("--obs-dir", default=str(OBS_DIR))

    # --- mode switches ---
    p.add_argument("--no-scores",   action="store_true",
                   help="Skip the scores plot")
    p.add_argument("--scores-only", action="store_true",
                   help="Skip the sequence plot (don't load nowcasts/obs)")

    # --- output ---
    p.add_argument("--save-dir", default=None,
                   help="If set, save figures here instead of showing them")

    return p.parse_args()


# =============================================================================
# Plot 1 — Nowcast vs. Observation sequence
# =============================================================================

def plot_sequence(args: argparse.Namespace) -> None:
    if args.init is None:
        raise ValueError("--init is required for the sequence plot. "
                         "Use --scores-only to skip it.")

    init_ts = pd.Timestamp(args.init)
    print(f"\nLoading nowcast for {init_ts} ...")
    nwc_loader = SatelliteNowcastLoader(data_dir=args.nwc_dir)
    nowcast_ds = nwc_loader.load_data(init_ts, init_ts).compute()

    obs_end = init_ts + pd.Timedelta(nowcast_ds.lead_time.max().values)
    print(f"Loading observations up to {obs_end} ...")
    obs_loader = SatelliteObservationLoader(data_dir=args.obs_dir)
    try:
        obs_ds = obs_loader.load_data(init_ts, obs_end).compute()
    except ValueError:
        print("  No observation files found — plotting nowcast only.")
        obs_ds = None

    INIT_IDX   = 0
    lead_steps = range(0, args.n_steps * args.step_every, args.step_every)
    n_steps    = len(lead_steps)
    init_time  = nowcast_ds.initialization_time.values[INIT_IDX]

    # Optional spatial subset
    bbox = getattr(args, "bbox", None)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
    else:
        lon_min = lon_max = lat_min = lat_max = None

    show_diff = getattr(args, "show_diff", False)
    n_rows    = 3 if show_diff else 2
    row_labels = ["Nowcast", "Observation", "Difference (NWC − OBS)"]

    fig, axes = plt.subplots(
        n_rows, n_steps,
        figsize=(3.5 * n_steps, 2.7 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    for row, label in enumerate(row_labels[:n_rows]):
        axes[row, 0].text(
            -0.12, 0.5, label,
            transform=axes[row, 0].transAxes,
            va="center", ha="right",
            fontsize=11, fontweight="bold",
            rotation=90,
        )

    img = img_diff = None
    for col, lead_idx in enumerate(lead_steps):
        valid_time = pd.Timestamp(nowcast_ds.valid_time.values[INIT_IDX, lead_idx])
        title      = valid_time.strftime("%H:%M UTC")

        nwc_step = nowcast_ds["probabilistic_advection"].isel(
            initialization_time=INIT_IDX, lead_time=lead_idx, ensemble=0
        )
        if bbox is not None:
            nwc_step = nwc_step.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

        obs_step = None
        if obs_ds is not None and valid_time.to_datetime64() <= obs_ds.time.values[-1]:
            obs_times   = obs_ds.time.values
            nearest_idx = np.argmin(np.abs(obs_times - valid_time.to_datetime64()))
            diff = abs(pd.Timestamp(obs_times[nearest_idx]) - valid_time)
            if diff <= pd.Timedelta("8min"):
                obs_step = obs_ds["sds"].isel(time=nearest_idx).rename({"y": "lat", "x": "lon"})
                if bbox is not None:
                    obs_step = obs_step.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

        # Difference panel (only defined when both exist and grids align)
        diff_step = None
        if show_diff and nwc_step is not None and obs_step is not None:
            try:
                diff_step = nwc_step - obs_step.values
            except Exception:
                diff_step = None

        panels = [(nwc_step, "inferno", args.vmin, args.vmax),
                  (obs_step,  "inferno", args.vmin, args.vmax)]
        if show_diff:
            abs_max = max(abs(args.vmin), abs(args.vmax)) / 2
            panels.append((diff_step, "RdBu_r", -abs_max, abs_max))

        for r, (data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, col]
            if bbox is not None:
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                ax.autoscale(False)
            if data is not None:
                mapped = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    add_colorbar=False,
                    vmin=vmin, vmax=vmax,
                    cmap=cmap,
                )
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="grey")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
                if r < 2:
                    img = mapped
                else:
                    img_diff = mapped
            else:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, "Not yet\navailable",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="white")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")

    # Colourbars — one for NWC/OBS rows, one for diff row
    if img is not None:
        fig.colorbar(img, ax=axes[:2, :], orientation="vertical",
                     fraction=0.02, pad=0.02, label="GHI (W/m²)")
    if show_diff and img_diff is not None:
        fig.colorbar(img_diff, ax=axes[2, :], orientation="vertical",
                     fraction=0.02, pad=0.02, label="Difference (W/m²)")
    fig.suptitle(
        f"Nowcast vs. Observation  |  init {pd.Timestamp(init_time).strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    _show_or_save(fig, args, f"sequence_{pd.Timestamp(init_time).strftime('%Y%m%d_%H%M')}.png")


# =============================================================================
# Plot 2 — MAE / RMSE vs. forecast horizon
# =============================================================================

def plot_scores(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)

    # Resolve which files to load
    if args.scores_start and args.scores_end:
        start_tag = pd.Timestamp(args.scores_start).strftime("%Y%m%d")
        end_tag   = pd.Timestamp(args.scores_end).strftime("%Y%m%d")
        mae_path  = results_dir / f"mae_{start_tag}_{end_tag}.nc"
        rmse_path = results_dir / f"rmse_{start_tag}_{end_tag}.nc"
        if not mae_path.exists() or not rmse_path.exists():
            raise FileNotFoundError(
                f"Could not find {mae_path} or {rmse_path}.\n"
                f"Run: uv run python run_validation.py "
                f"--start {args.scores_start} --end {args.scores_end}"
            )
    else:
        mae_files  = sorted(results_dir.glob("mae_*.nc"))
        rmse_files = sorted(results_dir.glob("rmse_*.nc"))
        if not mae_files or not rmse_files:
            raise FileNotFoundError(
                f"No score files found in {results_dir.resolve()}. "
                "Run run_validation.py first, or pass --scores-start / --scores-end."
            )
        mae_path  = mae_files[-1]   # use most recent
        rmse_path = rmse_files[-1]

    print(f"\nLoading scores from {mae_path.name} / {rmse_path.name} ...")
    mae_da  = xr.open_dataarray(mae_path,  engine="h5netcdf")
    rmse_da = xr.open_dataarray(rmse_path, engine="h5netcdf")

    # Restore lead_time from integer minutes
    for da in (mae_da, rmse_da):
        da["lead_time"] = pd.to_timedelta(da.lead_time, unit="min")

    # Optional spatial subset before averaging
    bbox = getattr(args, "bbox", None)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        mae_da  = mae_da.sel( lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        rmse_da = rmse_da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        print(f"  Averaging over bbox lon=[{lon_min}, {lon_max}] lat=[{lat_min}, {lat_max}]"
              f"  ({mae_da.sizes['lat']}×{mae_da.sizes['lon']} grid points)")

    mae_by_lead  = mae_da.mean(dim=["lat", "lon"])
    rmse_by_lead = rmse_da.mean(dim=["lat", "lon"])
    lead_hours   = mae_by_lead.lead_time.values / np.timedelta64(1, "h")

    # Derive a readable date range from the filename
    stem  = mae_path.stem          # e.g. mae_20260226_20260227
    parts = stem.split("_")        # ['mae', '20260226', '20260227']
    date_label = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:]} → {parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:]}"
    domain_label = f"bbox {bbox}" if bbox is not None else "full domain"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lead_hours, mae_by_lead.values,  label="MAE",  marker="o", linewidth=2)
    ax.plot(lead_hours, rmse_by_lead.values, label="RMSE", marker="s", linewidth=2)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Error (W/m²)")
    ax.set_title(f"Forecast error vs. lead time  |  {date_label}  |  {domain_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _show_or_save(fig, args, f"scores_{parts[1]}_{parts[2]}.png")


# =============================================================================
# Helpers
# =============================================================================

def _show_or_save(fig: plt.Figure, args: argparse.Namespace, filename: str) -> None:
    if args.save_dir:
        save_path = Path(args.save_dir) / filename
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = parse_args()

    if not args.scores_only:
        plot_sequence(args)

    if not args.no_scores:
        plot_scores(args)


if __name__ == "__main__":
    main()
