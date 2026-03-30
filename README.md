# 🌦️ sunflow-scores

Note: This repo is currently *under construction*

A Python framework for calculating scores of **solar irradiance (GHI)** nowcasts at the Danish Meteorological Institute (DMI). 



## ✨ Key Features
* **Solar Nowcasting Validation:** Evaluates 15-minute temporal resolution satellite-based predictions against satellite observations.
* **Robust Meteorological Metrics:** Powered by the `scores` package, calculating continuous metrics (RMSE, MAE) and spatial verification metrics like the Fractions Skill Score (FSS).
* **Highly Scalable:** Built on `xarray` and `dask` to process gigabytes of NetCDF files out-of-core without crashing system memory.
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
To run the validation, use the `run_validation.py` script. You must provide the start and end dates for the validation period, and the directories containing the nowcast and observation data.

### Basic Usage
Here is a basic example of how to run the validation for a specific date range if you are using MSG-cpp data and the nowcasts produced at DMI:
```bash
python run_validation.py \
    --start 2025-01-01 \
    --end 2025-01-31 \
    --nwc-dir /path/to/your/nowcasts \
    --obs-dir /path/to/your/observations \
    --output-dir results/
```
This will process all nowcasts between January 1st and January 31st, 2025, and save the calculated scores in the `results/` directory.

### Command-Line Arguments
Here are all the available options:

| Argument | Description | Required | Default |
|---|---|---|---|
| `--start` | The first nowcast initialization time (e.g., `2023-01-01`). | Yes | |
| `--end` | The last nowcast initialization time (e.g., `2023-01-31 23:45`). | Yes | |
| `--nwc-dir` | Directory containing the nowcast NetCDF files. | Yes | |
| `--obs-dir` | Directory containing the observation NetCDF files. | Yes | |
| `--output-dir` | Directory where the output score files will be saved. | No | `results` |
| `--nowcast_ghi_var` | The name of the GHI variable in the nowcast files. | No | `probabilistic_advection` |
| `--obs_ghi_var` | The name of the GHI variable in the observation files. | No | `sds` |
| `--obs_cs_ghi_var` | The name of the clear-sky GHI variable in the observation files. | No | `sds_cs` |


### Advanced Usage (with custom variable names)
If your datasets use different variable names for GHI than the default for MSG-cpp data, you can specify them on the command line:
```bash
python run_validation.py \
    --start 2023-01-01 \
    --end 2023-01-31 \
    --nwc-dir /path/to/nowcasts \
    --obs-dir /path/to/observations \
    --nowcast_ghi_var GHI_nowcast \
    --obs_ghi_var GHI_observation \
    --obs_cs_ghi_var GHI_clearsky
```

### ✅ Yearly batch processing
If you have month-by-month nowcast data organized by `YYYYMM` folders, use:
```bash
python run_yearly_validation.py \
  --year 2025 \
  --nwc-base-dir /path/to/monthly-nowcasts \
  --obs-dir /path/to/observations \
  --output-dir results \
  --nowcast-ghi-var GHI \
  --obs-ghi-var GHI \
  --obs-cs-ghi-var CLEARSKY_GHI
```

This will call `run_validation.py` for each month and produce results files like:
`results/mae_20250101_20250131.nc`, `results/rmse_20250101_20250131.nc`, etc.

## 📂 Project Structure
```text
NowcastingValidation/
├── validator.py              # Core library: SatelliteNowcastLoader,
│                             #   SatelliteObservationLoader, ScoreCalculator
├── run_validation.py         # Batch Validation Script: loads data, computes MAE/RMSE,
│                             #   writes results to results/ as NetCDF
├── plot_results.py           # Plots nowcast vs. observation sequence and
│                             #   MAE/RMSE vs. lead time (terminal or notebook)
├── pyproject.toml            # uv dependency definitions
├── uv.lock                   # Strictly locked dependency hashes
├── .gitignore                # Excludes data files and results
└── README.md
