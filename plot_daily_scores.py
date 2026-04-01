#!/usr/bin/env python3
"""Quick plotting helper for daily validation CSVs.

The validation script writes one CSV per day with columns like:
    initialization_time, ensemble, valid_time, mae_by_init, rmse_by_init, lead_time_minutes

This helper can:
- plot a single day's CSV as heatmaps
- plot an annual daily summary from many CSVs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot daily validation CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one daily CSV or a directory containing scores_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="If set and --input is a directory, make an annual daily summary plot.",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="If set and --input is a directory, plot the mean score per lead time across all CSVs.",
    )
    parser.add_argument(
        "--average-init",
        action="store_true",
        help="If set and --input is a directory, plot the mean score per initialization time across all CSVs.",
    )
    parser.add_argument(
        "--average-heatmap",
        action="store_true",
        help="If set and --input is a directory, plot a heatmap averaged over all days by initialization time and lead time.",
    )
    return parser.parse_args()


def _load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["initialization_time", "valid_time"])
    if "lead_time_minutes" not in df.columns:
        raise ValueError(f"Expected lead_time_minutes column in {path}")
    return df


def plot_day_heatmap(csv_path: Path, output_dir: Path) -> Path:
    df = _load_daily_csv(csv_path)

    pivot_mae = df.pivot_table(
        index="initialization_time",
        columns="lead_time_minutes",
        values="mae_by_init",
        aggfunc="mean",
    ).sort_index()
    pivot_rmse = df.pivot_table(
        index="initialization_time",
        columns="lead_time_minutes",
        values="rmse_by_init",
        aggfunc="mean",
    ).sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    for ax, pivot, title, cmap in [
        (axes[0], pivot_mae, "MAE by init time", "RdYlGn_r"),
        (axes[1], pivot_rmse, "RMSE by init time", "RdYlGn_r"),
    ]:
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Lead time (minutes)")
        ax.set_ylabel("Initialization time")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([pd.Timestamp(v).strftime("%H:%M") for v in pivot.index])
        fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}_heatmap.png"
    fig.suptitle(csv_path.name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_summary(csv_dir: Path, output_dir: Path) -> Path:
    csv_files = sorted(csv_dir.glob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {csv_dir}")

    rows = []
    for path in csv_files:
        df = _load_daily_csv(path)
        day = path.stem.replace("scores_", "")
        daily = df.groupby("lead_time_minutes", as_index=False)[["mae_by_init", "rmse_by_init"]].mean()
        daily.insert(0, "day", day)
        rows.append(daily)

    summary = pd.concat(rows, ignore_index=True)
    summary["day"] = pd.to_datetime(summary["day"], format="%Y%m%d")

    pivot_mae = summary.pivot(index="day", columns="lead_time_minutes", values="mae_by_init").sort_index()
    pivot_rmse = summary.pivot(index="day", columns="lead_time_minutes", values="rmse_by_init").sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for ax, pivot, title, cmap in [
        (axes[0], pivot_mae, "Daily mean MAE by lead time", "RdYlGn_r"),
        (axes[1], pivot_rmse, "Daily mean RMSE by lead time", "RdYlGn_r"),
    ]:
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Lead time (minutes)")
        ax.set_ylabel("Day")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
        # keep y ticks sparse so 2025 is readable
        y_ticks = list(range(0, len(pivot.index), max(1, len(pivot.index) // 12)))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([pivot.index[i].strftime("%Y-%m-%d") for i in y_ticks])
        fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "daily_scores_summary.png"
    fig.suptitle(csv_dir.name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_leadtime_average(csv_dir: Path, output_dir: Path) -> Path:
    csv_files = sorted(csv_dir.glob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {csv_dir}")

    frames = []
    for path in csv_files:
        df = _load_daily_csv(path)
        daily = df.groupby("lead_time_minutes", as_index=False)[["mae_by_init", "rmse_by_init"]].mean()
        frames.append(daily)

    summary = pd.concat(frames, ignore_index=True)
    average = summary.groupby("lead_time_minutes", as_index=False)[["mae_by_init", "rmse_by_init"]].mean()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    ax.plot(average["lead_time_minutes"], average["mae_by_init"], marker="o", label="MAE")
    ax.plot(average["lead_time_minutes"], average["rmse_by_init"], marker="o", label="RMSE")
    ax.set_title("Mean score per lead time across all days")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "daily_scores_leadtime_average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_init_average(csv_dir: Path, output_dir: Path) -> Path:
    csv_files = sorted(csv_dir.glob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {csv_dir}")

    frames = []
    for path in csv_files:
        df = _load_daily_csv(path)
        daily = df.copy()
        daily["init_time"] = daily["initialization_time"].dt.strftime("%H:%M")
        frames.append(
            daily.groupby("init_time", as_index=False)[["mae_by_init", "rmse_by_init"]].mean()
        )

    summary = pd.concat(frames, ignore_index=True)
    average = summary.groupby("init_time", as_index=False)[["mae_by_init", "rmse_by_init"]].mean()

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
    ax.plot(average["init_time"], average["mae_by_init"], marker="o", label="MAE")
    ax.plot(average["init_time"], average["rmse_by_init"], marker="o", label="RMSE")
    ax.set_title("Mean score per initialization time across all days")
    ax.set_xlabel("Initialization time")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "daily_scores_init_average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_average_heatmap(csv_dir: Path, output_dir: Path) -> Path:
    csv_files = sorted(csv_dir.glob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {csv_dir}")

    frames = []
    for path in csv_files:
        df = _load_daily_csv(path)
        daily = df.copy()
        daily["init_time"] = daily["initialization_time"].dt.strftime("%H:%M")
        daily = (
            daily.groupby(["init_time", "lead_time_minutes"], as_index=False)[["mae_by_init", "rmse_by_init"]]
            .mean()
        )
        frames.append(daily)

    summary = pd.concat(frames, ignore_index=True)
    average = (
        summary.groupby(["init_time", "lead_time_minutes"], as_index=False)[["mae_by_init", "rmse_by_init"]]
        .mean()
    )

    pivot_mae = average.pivot(index="init_time", columns="lead_time_minutes", values="mae_by_init")
    pivot_rmse = average.pivot(index="init_time", columns="lead_time_minutes", values="rmse_by_init")

    fig, axes = plt.subplots(2, 1, figsize=(14, 14), constrained_layout=True)
    for ax, pivot, title, cmap in [
        (axes[0], pivot_mae, "Average MAE by initialization time and lead time", "RdYlGn_r"),
        (axes[1], pivot_rmse, "Average RMSE by initialization time and lead time", "RdYlGn_r"),
    ]:
        pivot = pivot.sort_index()
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Lead time (minutes)")
        ax.set_ylabel("Initialization time")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "daily_scores_average_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_file():
        out_path = plot_day_heatmap(input_path, output_dir)
        print(f"Wrote {out_path}")
        return

    if input_path.is_dir() and args.average:
        out_path = plot_leadtime_average(input_path, output_dir)
        print(f"Wrote {out_path}")
        return

    if input_path.is_dir() and args.average_init:
        out_path = plot_init_average(input_path, output_dir)
        print(f"Wrote {out_path}")
        return

    if input_path.is_dir() and args.average_heatmap:
        out_path = plot_average_heatmap(input_path, output_dir)
        print(f"Wrote {out_path}")
        return

    if input_path.is_dir() and args.summary:
        out_path = plot_summary(input_path, output_dir)
        print(f"Wrote {out_path}")
        return

    raise ValueError("Provide a single CSV file, or a directory with --summary, --average, --average-init, or --average-heatmap for directory views.")


if __name__ == "__main__":
    main()
