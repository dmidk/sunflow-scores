# 🌦️ sunflow-scores (currently *under construction*)

A Python framework for calculating scores of **solar irradiance (GHI)** nowcasts at the Danish Meteorological Institute (DMI). 



## ✨ Key Features
* **Solar Nowcasting Validation:** Evaluates 15-minute temporal resolution satellite-based predictions against satellite observations.
* **Robust Meteorological Metrics:** Powered by the `scores` package, calculating continuous metrics (RMSE, MAE) and spatial verification metrics like the Fractions Skill Score (FSS).
* **Dependency Management:** Uses `uv` for reproducible virtual environments.



## 🚀 Getting Started: Cloning & Setup
We use uv for dependency management. It is extremely fast and ensures that everyone on the team is running the exact same package versions.

Step 1: Install uv (One-time setup)
If you do not already have uv installed on your laptop, run this command:

`curl -LsSf https://astral.sh/uv/install.sh | sh `

Step 2: Clone the Repository
Download the project code from github to your laptop. Open your terminal and run:

`git clone https://github.com/dmidk/sunflow-scores.git`

`cd sunflow-scores`


Step 3: Build the Environment with uv
Instead of manually creating environments and installing packages, simply run:

`uv sync`
What this does: uv will read the pyproject.toml file, automatically download the correct Python version (if needed), create an isolated .venv folder, and install exact, locked versions of xarray, scores, dask, etc. It should take seconds.


## 🚀 Running the Validation
The main validator is `run_validation.py`. It takes a start date, an end date, the nowcast directory, and the observation directory.

### One day
Run a single validation day like this:
```bash
uv run python run_validation.py \
  --start 2025-01-02 \
  --end 2025-01-02 \
  --nwc-dir /path/to/sunflow_output/202501 \
  --obs-dir /path/to/satellite_GHI/202501 \
  --output-dir results
```

This writes a daily CSV such as `results/scores_20250102.csv`.

### A whole month
Because the inputs are organized in `YYYYMM` folders and the outputs are daily CSVs, the usual workflow is to loop over the days in a month.

Example for January 2025:
```bash
for day in $(seq -w 1 31); do
  uv run python run_validation.py \
    --start 2025-01-${day} \
    --end 2025-01-${day} \
    --nwc-dir /path/to/sunflow_output/202501 \
    --obs-dir /path/to/satellite_GHI/202501 \
    --output-dir results
done
```


### Custom variable names
If your files use different variable names, pass them explicitly:
```bash
uv run python run_validation.py \
  --start 2025-01-02 \
  --end 2025-01-02 \
  --nwc-dir /path/to/nowcasts \
  --obs-dir /path/to/observations \
  --nowcast_ghi_var GHI_nowcast \
  --obs_ghi_var GHI_observation \
 
```

### Command-line arguments
| Argument | Description | Required | Default |
|---|---|---|---|
| `--start` | First nowcast initialization time. | Yes | |
| `--end` | Last nowcast initialization time. | Yes | |
| `--nwc-dir` | Directory containing the nowcast NetCDF files. | Yes | |
| `--obs-dir` | Directory containing the observation NetCDF files. | Yes | |
| `--output-dir` | Directory where the output score files are written. | No | `results` |
| `--nowcast_ghi_var` | GHI variable in the nowcast files. | No | `probabilistic_advection` |
| `--obs_ghi_var` | GHI variable in the observation files. | No | `sds` |
| `--obs_cs_ghi_var` | Clear-sky GHI variable in the observation files. | No | `sds_cs` |

### What the validation writes
The current pipeline writes one CSV per day:
`scores_YYYYMMDD.csv`

Each file contains by-init scores and a `lead_time_minutes` column, which is what the plotting tools consume.

## 📊 Plotting the results

### Daily plots
Use `plot_daily_scores.py` for one daily CSV or for a directory of daily CSVs.

Single day heatmap:
```bash
uv run python plot_daily_scores.py \
  --input results/scores_20250102.csv \
  --output-dir results/plots
```

Monthly summary over all daily CSVs in a directory:
```bash
uv run python plot_daily_scores.py \
  --input results \
  --summary \
  --output-dir results/plots
```

Monthly average heatmap by initialization time and lead time:
```bash
uv run python plot_daily_scores.py \
  --input results \
  --average-heatmap \
  --output-dir results/plots
```

You can also choose a metric with `--metric mae`, `--metric rmse`, or `--metric both`.

### Monthly plots
Use `plot_monthly_heatmaps.py` to write one summary plot and one averaged heatmap for each month found in a directory of daily CSVs.

Plot both summary and averaged heatmaps for both metrics:
```bash
uv run python plot_monthly_heatmaps.py \
  --input results \
  --summary \
  --heatmap \
  --metric both \
  --output-dir results/monthly_plots
```

Only summary MAE plots:
```bash
uv run python plot_monthly_heatmaps.py \
  --input results \
  --summary \
  --metric mae \
  --output-dir results/monthly_plots
```

Only averaged RMSE heatmaps:
```bash
uv run python plot_monthly_heatmaps.py \
  --input results \
  --heatmap \
  --metric rmse \
  --output-dir results/monthly_plots
```

The monthly script does not require every day of the month to be present. It will plot whatever daily CSVs exist for that month, which is useful when some days were skipped because there was no data.

## 📂 Project Structure
The core library code lives under `src/sunflow_scores/`.
```text
sunflow-scores/
├── src/
│   └── sunflow_scores/
│       ├── __init__.py
│       ├── validator.py      # Core library: SatelliteNowcastLoader,
│                             # SatelliteObservationLoader, ScoreCalculator
├── run_validation.py         # Daily validation script: writes one scores_YYYYMMDD.csv per run
├── plot_daily_scores.py      # Plot one day or a directory of daily CSVs
├── plot_monthly_heatmaps.py  # Plot monthly summaries / averaged heatmaps from daily CSVs
├── pyproject.toml            # uv dependency definitions
├── uv.lock                   # Strictly locked dependency hashes
├── .gitignore                # Excludes data files and results
└── README.md
