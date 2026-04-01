#!/usr/bin/env python3
"""Plot monthly validation summaries and averaged heatmaps from daily CSV outputs.

The validation pipeline writes one CSV per day named like:
    scores_YYYYMMDD.csv

This script groups those daily files by month and writes plots per month from
whatever daily files are available:
- a summary heatmap by day and lead time
- an averaged heatmap by initialization time and lead time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot monthly average validation heatmaps from daily CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing scores_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where monthly plots will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "both"],
        default="both",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write the month-by-day summary heatmap.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Write the averaged init-time/lead-time heatmap.",
    )
    return parser.parse_args()


def _load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["initialization_time", "valid_time"])
    if "lead_time_minutes" not in df.columns:
        raise ValueError(f"Expected lead_time_minutes column in {path}")
    return df


def _month_from_path(path: Path) -> str:
    return path.stem.replace("scores_", "")[:6]


def _prepare_day(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.copy()
    daily["init_time"] = daily["initialization_time"].dt.strftime("%H:%M")
    return (
        daily.groupby(["init_time", "lead_time_minutes"], as_index=False)[["mae_by_init", "rmse_by_init"]]
        .mean()
    )


def _metric_columns(metric: str) -> list[str]:
    if metric == "mae":
        return ["mae_by_init"]
    if metric == "rmse":
        return ["rmse_by_init"]
    return ["mae_by_init", "rmse_by_init"]


def _plot_month_summary(month: str, daily_frames: list[tuple[pd.DataFrame, str]], output_dir: Path, metric: str) -> Path:
    metric_col = _metric_columns(metric)[0]
    rows = []
    for frame, day in daily_frames:
        daily = frame.groupby("lead_time_minutes", as_index=False)[[metric_col]].mean()
        daily.insert(0, "day", day)
        rows.append(daily)

    summary = pd.concat(rows, ignore_index=True)
    summary["day"] = pd.to_datetime(summary["day"], format="%Y%m%d")

    pivot = summary.pivot(index="day", columns="lead_time_minutes", values=metric_col).sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn_r",
    )
    ax.set_title(f"{month} daily mean {metric.upper()} by lead time")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Day")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
    y_ticks = list(range(0, len(pivot.index), max(1, len(pivot.index) // 12)))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([pivot.index[i].strftime("%Y-%m-%d") for i in y_ticks])
    fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_month_average(month: str, frames: list[pd.DataFrame], output_dir: Path, metric: str) -> Path:
    summary = pd.concat(frames, ignore_index=True)
    metric_col = _metric_columns(metric)[0]
    average = summary.groupby(["init_time", "lead_time_minutes"], as_index=False)[[metric_col]].mean()

    pivot = average.pivot(index="init_time", columns="lead_time_minutes", values=metric_col).sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn_r",
    )
    ax.set_title(f"{month} average {metric.upper()} by init time and lead time")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("Initialization time")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, shrink=0.85)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"monthly_scores_{month}_{metric}_average_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    csv_files = sorted(input_dir.glob("scores_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scores_*.csv files found in {input_dir}")

    month_to_files: dict[str, list[Path]] = {}
    for path in csv_files:
        month_to_files.setdefault(_month_from_path(path), []).append(path)

    for month, paths in sorted(month_to_files.items()):
        print(f"Processing {month}: found {len(paths)} daily files")

        daily_frames = []
        average_frames = []
        for path in paths:
            df = _load_daily_csv(path)
            daily_frames.append((_prepare_day(df), path.stem.replace("scores_", "")))
            average_frames.append(_prepare_day(df))

        metrics = [args.metric] if args.metric != "both" else ["mae", "rmse"]
        wrote_any = False
        if args.summary or not args.heatmap:
            for metric in metrics:
                summary_path = _plot_month_summary(month, daily_frames, output_dir, metric)
                print(f"Wrote {summary_path}")
                wrote_any = True
        if args.heatmap or not args.summary:
            for metric in metrics:
                average_path = _plot_month_average(month, average_frames, output_dir, metric)
                print(f"Wrote {average_path}")
                wrote_any = True
        if not wrote_any:
            print(f"SKIP {month}: no plot type selected")


if __name__ == "__main__":
    main()