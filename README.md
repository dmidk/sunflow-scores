# 🌦️ Nowcasting Validation Platform

A modular, scalable Python framework for evaluating high-resolution nowcasts at the Danish Meteorological Institute (DMI). 

While the platform is designed to be highly extensible for various meteorological parameters and numerical weather prediction (NWP) integrations, its **first operational component** focuses specifically on **solar irradiance (GHI)** nowcasting.



## ✨ Key Features
* **Modular Architecture:** Built using Abstract Base Classes. Separates the core mathematical evaluation engine from the specific data loaders, making it trivial to add new variables (e.g., precipitation) or new forecast models in the future.
* **Component 1 - Solar Validation:** Evaluates 15-minute temporal resolution satellite-based predictions against ground-truth satellite observations.
* **Robust Meteorological Metrics:** Powered by the `scores` package, calculating continuous metrics (RMSE, MAE, Mean Bias) and spatial verification metrics like the Fractions Skill Score (FSS) to avoid the "double penalty" of spatial displacement.
* **Highly Scalable:** Built natively on `xarray` and `dask` to lazily process gigabytes of NetCDF files out-of-core without crashing system memory.
* **Modern Dependency Management:** Uses `uv` for lightning-fast, reproducible, and strictly locked virtual environments.

## 📂 Project Structure
```text
NowcastingValidation/
├── validator.py              # Core OOP classes (Data Loaders & Evaluator)
├── notebooks/                # Jupyter Notebooks for analysis and plotting
├── pyproject.toml            # uv dependency definitions
├── uv.lock                   # Strictly locked dependency hashes
├── .gitignore                # Protects the repo from massive data files
└── README.md