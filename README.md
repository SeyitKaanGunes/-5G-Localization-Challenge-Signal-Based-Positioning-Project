# 📡 5G Signal-Based Localization – Path Loss & Ray Tracing

This repository contains two separate but related simulations used in a 5G signal-based localization project:

- `path_loss_model.py`: A path-loss optimization script using measurement data and building/vegetation geometries.
- `ray_tracing_simulation.py`: A 3D ray tracing setup using [Sionna RT](https://nvlabs.github.io/sionna/rt.html) to simulate signal propagation paths.

> ⚠️ **IMPORTANT:** Raw measurement data (`.xlsx`) and 3D object models (`.obj`) are **not included** due to privacy and competition restrictions.

---

## 📊 path_loss_model.py

An adaptive path-loss model that:
- Uses both geometric distance and Timing Advance (TA) data
- Automatically selects single or dual-slope models based on LOS/NLOS balance
- Performs grid search with Huber regression and multiple weighting strategies
- Outputs prediction results and residuals for evaluation

### Requirements:
```bash
pip install pandas numpy matplotlib scipy tqdm trimesh pyproj

## 🌐 ray_tracing_simulation.py
A minimal Sionna-RT setup that:

Loads ground, vegetation, building and base station meshes

Adds transmitters and receivers from Excel

Simulates ray paths and computes path loss

Visualizes path loss values for each receiver

🔒 Data Policy
All measurement files, geospatial Excel sheets, and .obj geometries used in this project are confidential and cannot be shared due to data restrictions.
