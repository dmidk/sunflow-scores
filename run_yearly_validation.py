
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import sys


project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def main():
    """
    Runs the validation script for an entire year, iterating month by month.

    This script constructs the necessary date ranges and directory paths for each
    month and calls the `run_validation.py` script via `uv run`.
    """
    parser = argparse.ArgumentParser(
        description="Run validation for a full year, month by month.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--year", type=int, required=True, help="Year to process.")
    parser.add_argument(
        "--nwc-base-dir",
        type=Path,
        required=True,
        help="Base directory for nowcast data. Should contain subfolders like '202501', '202502', etc."
    )
    parser.add_argument(
        "--obs-dir",
        type=Path,
        required=True,
        help="Directory for the reprojected satellite observation data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results",
        help="Directory to save the output score files."
    )
    # Add arguments for configurable variable names, passing them through
    parser.add_argument("--nowcast-ghi-var", default="GHI", help="Variable name for nowcast GHI.")
    parser.add_argument("--obs-ghi-var", default="GHI", help="Variable name for observation GHI.")
    parser.add_argument("--obs-cs-ghi-var", default="CLEARSKY_GHI", help="Variable name for observation clear-sky GHI.")

    args = parser.parse_args()

    print(f"Starting validation for the year {args.year}...")

    for month in range(1, 13):
        # Create start and end dates for the current month
        start_date = pd.Timestamp(f"{args.year}-{month:02d}-01")
        end_date = start_date + pd.offsets.MonthEnd(0)

        # Format dates for the command
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Construct the path to the nowcast data for the specific month
        month_str = start_date.strftime('%Y%m')
        nwc_month_dir = args.nwc_base_dir / month_str

        if not nwc_month_dir.exists():
            print(f"WARNING: Directory not found, skipping: {nwc_month_dir}")
            continue

        print(f"\n--- Processing {start_date.strftime('%B %Y')} ---")
        print(f"  Nowcast data: {nwc_month_dir}")
        print(f"  Date range: {start_str} to {end_str}")

        # Construct the command to run the validation script
        command = [
            "uv", "run", "python", "run_validation.py",
            "--start", start_str,
            "--end", end_str,
            "--nwc-dir", str(nwc_month_dir),
            "--obs-dir", str(args.obs_dir),
            "--output-dir", str(args.output_dir),
            "--nowcast_ghi_var", args.nowcast_ghi_var,
            "--obs_ghi_var", args.obs_ghi_var,
            "--obs_cs_ghi_var", args.obs_cs_ghi_var,
        ]

        try:
            # Execute the command
            process = subprocess.run(
                command,
                check=True,        # Raise an exception if the command fails
                capture_output=True, # Capture stdout and stderr
                text=True          # Decode output as text
            )
            print(process.stdout)
            if process.stderr:
                print("--- Stderr ---", file=sys.stderr)
                print(process.stderr, file=sys.stderr)

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to run validation for {month_str}", file=sys.stderr)
            print(f"  Return code: {e.returncode}", file=sys.stderr)
            print("--- Stdout ---", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print("--- Stderr ---", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            # Optional: decide if you want to stop on failure
            # sys.exit(1)
        except FileNotFoundError:
            print("ERROR: 'uv' command not found. Make sure uv is installed and in your PATH.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
