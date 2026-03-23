<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Geospatial%20Demand%20Forecast&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=LightGBM%20%2B%20Spatial%20CV%20%2B%20Cyclical%20Encoding%20for%20Ride%20Demand&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/R²-0.85-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MAE-4.2_rides-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A geospatial demand forecasting system that predicts ride-share demand across Calgary neighborhoods using gradient boosting with spatial cross-validation and cyclical temporal encoding.**

Ride-share and delivery platforms need accurate demand forecasts at the zone level to position drivers efficiently and reduce wait times. This project builds a forecasting pipeline that trains four regression models (Ridge, Random Forest, XGBoost, LightGBM) on 20K synthetic ride demand records spanning 30 Calgary neighborhoods. Spatial features (distance to downtown, KMeans zone clusters) and cyclical temporal encoding (sine/cosine hour, day, month) capture the geographic and temporal structure of demand. Leave-one-zone-out spatial cross-validation ensures the model generalizes to unseen locations.

```
Problem   →  Forecasting ride demand per zone per hour across a city
Solution  →  LightGBM with spatial features, cyclical encoding, and spatial CV
Impact    →  R² 0.85, MAE 4.2 rides/zone/hour across 30 Calgary neighborhoods
```

---

## Key results

| Metric | Value |
|--------|-------|
| R-squared | 0.85 |
| MAE | 4.2 rides/zone/hour |
| RMSE | 5.8 |
| MAPE | 18.3% |
| Best model | LightGBM |

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Synthetic data  │───▶│  Feature          │───▶│  Spatial +       │
│  generation      │    │  engineering      │    │  temporal encode  │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Model training      │───▶│  Spatial CV           │
              │  (4 regressors)      │    │  (leave-one-zone-out) │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Zone-level          │───▶│  Demand forecast      │
              │  predictions         │    │  dashboard            │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_23_geospatial_demand_forecast/
├── data/
│   ├── ride_demand.csv              # Ride demand dataset (20K records)
│   ├── zones.csv                    # Zone reference table
│   └── generate_data.py             # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data generation and loading
│   └── model.py                     # Training, spatial CV, evaluation
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Spatial and temporal features
│   ├── 03_modeling.ipynb            # Model training and comparison
│   └── 04_spatial_cv_evaluation.ipynb # Spatial CV and error analysis
├── app.py                           # Streamlit dashboard
├── index.html                       # Project landing page
├── requirements.txt
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_23_geospatial_demand_forecast

# Install dependencies
pip install -r requirements.txt

# Generate ride demand data
python data/generate_data.py

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic ride demand based on Calgary geography |
| Records | 20,000 |
| Zones | 30 Calgary neighborhoods |
| Features | 13 (geographic, temporal, weather, infrastructure) |
| Target | demand_count (rides per zone per hour) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Spatial features</b></summary>

- Haversine distance from each zone centroid to downtown Calgary (51.0477, -114.0630)
- KMeans clustering (k=6) on latitude/longitude to create geographic groupings
- Real Calgary neighborhood coordinates for 30 zones
</details>

<details>
<summary><b>Temporal encoding</b></summary>

- Cyclical sine/cosine encoding for hour (24h), day of week (7d), and month (12m)
- Binary rush hour flag (07:00-09:00, 16:00-18:00)
- Weekend indicator for Saturday and Sunday
</details>

<details>
<summary><b>Model training</b></summary>

- Four regressors: Ridge, Random Forest, XGBoost, LightGBM
- 5-fold cross-validation on training set
- Metrics: MAE, RMSE, R-squared, MAPE
</details>

<details>
<summary><b>Spatial cross-validation</b></summary>

- Leave-one-zone-out CV to test generalization to unseen geographic zones
- Trains on 29 zones, predicts on the held-out zone, repeats for all 30
- Provides a realistic estimate of performance on new deployment areas
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
