#!/usr/bin/env python3
"""Plot a validation bounding box on a map so you can see the selected domain.

The bounding box uses the same convention as run_validation.py --bbox:
    LON_MIN LAT_MIN LON_MAX LAT_MAX

Example (Denmark, the domain used by run_validation_denmark.sh):
    uv run python plot_domain.py --bbox 7.5 54.5 13.0 58.0 --output-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a validation bounding box on a map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--validation-bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=[7.5, 54.5, 15.5, 58.0],
        help="Validation bounding box: LON_MIN LAT_MIN LON_MAX LAT_MAX (Denmark by default).",
    )

    parser.add_argument(
        "--outer-domain-bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=[-20.75,37.25,30,73.5],
        help="Outer domain bounding box: LON_MIN LAT_MIN LON_MAX LAT_MAX (NW Europe nowcast domain by default).",
    )

    parser.add_argument(
        "--inner-domain-bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=[-10.75,47.25,20,63.5],
        help="Outer domain bounding box: LON_MIN LAT_MIN LON_MAX LAT_MAX (NW Europe nowcast domain by default).",
    )
    parser.add_argument(
        "--label",
        default="Denmark",
        help="Name of the domain, used in the title and file name.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the map PNG will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lon_min, lat_min, lon_max, lat_max = args.validation_bbox
    lon_min_outer, lat_min_outer, lon_max_outer, lat_max_outer = args.outer_domain_bbox
    lon_min_inner, lat_min_inner, lon_max_inner, lat_max_inner = args.inner_domain_bbox


    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [   lon_min_outer,
            lon_max_outer,
            lat_min_outer,
            lat_max_outer,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Base map features.
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dceefb")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f1ea")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle=":")

    # The selected validation domain (bounding box).
    rect = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=2.5,
        edgecolor="crimson",
        facecolor="crimson",
        alpha=0.18,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.add_patch(rect)
    # Outline on top (no fill) so the border is crisp.
    ax.add_patch(
        mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=2.5,
            edgecolor="crimson",
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    )

    # The inner nowcasting output domain (bounding box).
    rect = mpatches.Rectangle(
        (lon_min_inner, lat_min_inner),
        lon_max_inner - lon_min_inner,
        lat_max_inner - lat_min_inner,
        linewidth=2.5,
        edgecolor="green",
        facecolor="green",
        alpha=0.18,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.add_patch(rect)
    # Outline on top (no fill) so the border is crisp.
    ax.add_patch(
        mpatches.Rectangle(
            (lon_min_inner, lat_min_inner),
            lon_max_inner - lon_min_inner,
            lat_max_inner - lat_min_inner,
            linewidth=2.5,
            edgecolor="green",
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    )

    # Annotate the corner coordinates.
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min_outer, lat_min_outer, lat_max_outer, lat_max_outer, lat_min_outer],
        transform=ccrs.PlateCarree(),
        color="crimson",
        linewidth=0,
    )
    ax.text(
        (lon_min + lon_max) / 2,
        lat_max,
        f"Validation domain",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=10,
        color="crimson",
        #fontweight="bold",
    )


    ax.text(
        (lon_min_inner + lon_max_inner) / 2,
        lat_max_inner,
        f"Nowcast output domain",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=10,
        color="green",
        #fontweight="bold",
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"Nowcast input domain")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"domain_{args.label.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
