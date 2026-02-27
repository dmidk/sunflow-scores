# 🌦️ Nowcasting Validation Platform

A modular, scalable Python framework for evaluating high-resolution nowcasts at the Danish Meteorological Institute (DMI). 

While the platform is designed to be highly extensible for various meteorological parameters and numerical weather prediction (NWP) integrations, its **first operational component** focuses specifically on **solar irradiance (GHI)** nowcasting.



## ✨ Key Features
* **Modular Architecture:** Separates the core mathematical evaluation engine from the specific data loaders, making it trivial to add new variables (e.g., precipitation) or new forecast models in the future.
* **Component 1 - Solar Validation:** Evaluates 15-minute temporal resolution satellite-based predictions against ground-truth satellite observations.
* **Robust Meteorological Metrics:** Powered by the `scores` package, calculating continuous metrics (RMSE, MAE, Mean Bias) and spatial verification metrics like the Fractions Skill Score (FSS) to avoid the "double penalty" of spatial displacement.
* **Highly Scalable:** Built natively on `xarray` and `dask` to lazily process gigabytes of NetCDF files out-of-core without crashing system memory.
* **Modern Dependency Management:** Uses `uv` for lightning-fast, reproducible, and strictly locked virtual environments.



## 🚀 Getting Started: Cloning & Setup
We use uv for dependency management. It is extremely fast and ensures that everyone on the team is running the exact same package versions without the bloat of Conda.

Step 1: Install uv (One-time setup)
If you do not already have uv installed on your machine or the DMI server, run this command:

`curl -LsSf https://astral.sh/uv/install.sh | sh `
(You may need to restart your terminal after this completes).

Step 2: Clone the Repository
Download the project code from DMI's GitLab to your local machine or server. Open your terminal and run:

`git clone git@gitlab.dmi.dk:yourusername/NowcastingValidation.git`

`cd NowcastingValidation`
(Note: Replace your username with the appropriate GitLab path if this lives in a shared DMI group).

Step 3: Build the Environment with uv
Instead of manually creating environments and installing packages, simply run:

`uv sync`
What this does: uv will read the uv.lock file, automatically download the correct Python version (if needed), create an isolated .venv folder, and install exact, locked versions of xarray, scores, dask, etc. It takes seconds.




## 📂 Project Structure
```text
NowcastingValidation/
├── validator.py              # Core OOP classes (Data Loaders & Evaluator)
├── notebooks/                # Jupyter Notebooks for analysis and plotting
├── pyproject.toml            # uv dependency definitions
├── uv.lock                   # Strictly locked dependency hashes
├── .gitignore                # Protects the repo from massive data files
└── README.md


