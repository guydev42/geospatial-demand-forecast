"""
Train, evaluate, and compare demand forecasting models with spatial cross-validation.
Includes: Ridge, Random Forest, XGBoost, LightGBM.
Uses spatial features (distance to downtown, zone clusters) and cyclical temporal encoding.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUTS_DIR = "outputs"

DOWNTOWN_LAT = 51.0477
DOWNTOWN_LON = -114.0630


def _ensure_dirs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def engineer_features(df):
    """
    Add spatial and temporal features to the demand dataframe.

    Spatial features:
        - distance_to_downtown (haversine km)
        - zone_cluster (KMeans on lat/lon, k=6)

    Temporal features:
        - hour_sin, hour_cos (cyclical hour encoding)
        - day_sin, day_cos (cyclical day-of-week encoding)
        - is_rush_hour (7-9 or 16-18)
        - is_weekend (day_of_week >= 5)

    Returns:
        df with new features, list of feature column names
    """
    df = df.copy()

    # Spatial: distance to downtown
    df["distance_to_downtown"] = df.apply(
        lambda r: _haversine_km(r["latitude"], r["longitude"],
                                DOWNTOWN_LAT, DOWNTOWN_LON),
        axis=1,
    )

    # Spatial: zone clustering
    coords = df[["latitude", "longitude"]].values
    kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
    df["zone_cluster"] = kmeans.fit_predict(coords)

    # Temporal: cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Temporal: rush hour flag
    df["is_rush_hour"] = ((df["hour"] >= 7) & (df["hour"] <= 9) |
                          (df["hour"] >= 16) & (df["hour"] <= 18)).astype(int)

    # Temporal: weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    feature_cols = [
        "latitude", "longitude", "hour_sin", "hour_cos",
        "day_sin", "day_cos", "month_sin", "month_cos",
        "is_holiday", "temperature", "precipitation",
        "event_nearby", "population_density", "num_restaurants",
        "transit_stops_nearby", "distance_to_downtown", "zone_cluster",
        "is_rush_hour", "is_weekend",
    ]

    return df, feature_cols, kmeans


def _get_models():
    """Return model instances for demand forecasting."""
    models = {
        "Ridge": {
            "model": Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "needs_scaling": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                random_state=RANDOM_STATE, n_estimators=200,
                max_depth=12, min_samples_split=5, n_jobs=-1,
            ),
            "needs_scaling": False,
        },
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = {
            "model": XGBRegressor(
                random_state=RANDOM_STATE, verbosity=0,
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
            ),
            "needs_scaling": False,
        }
    except ImportError:
        print("XGBoost not installed, skipping.")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = {
            "model": LGBMRegressor(
                random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                n_estimators=300, max_depth=10, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            ),
            "needs_scaling": False,
        }
    except ImportError:
        print("LightGBM not installed, skipping.")

    return models


def _mape(y_true, y_pred):
    """Mean absolute percentage error, handling zeros in y_true."""
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def spatial_cv_split(df, zone_col="zone_id"):
    """
    Leave-one-zone-out spatial cross-validation.
    Yields (train_idx, test_idx) for each held-out zone.
    """
    zones = df[zone_col].unique()
    for zone in zones:
        test_idx = df.index[df[zone_col] == zone].values
        train_idx = df.index[df[zone_col] != zone].values
        yield train_idx, test_idx


def train_and_evaluate(df, use_spatial_cv=True):
    """
    Train all models, evaluate with MAE, RMSE, R-squared, and MAPE.
    Optionally runs spatial (leave-one-zone-out) cross-validation.

    Returns:
        dict of results per model
    """
    _ensure_dirs()

    df_feat, feature_cols, kmeans = engineer_features(df)

    X = df_feat[feature_cols].values.astype(float)
    y = df_feat["demand_count"].values.astype(float)

    # Standard train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = _get_models()
    results = {}
    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    for name, config in models_config.items():
        print(f"\n--- {name} ---")

        model = config["model"]
        Xtr = X_train_scaled if config["needs_scaling"] else X_train
        Xte = X_test_scaled if config["needs_scaling"] else X_test

        # 5-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_r2 = cross_val_score(model, Xtr, y_train, cv=cv, scoring="r2")
        cv_neg_mae = cross_val_score(model, Xtr, y_train, cv=cv,
                                     scoring="neg_mean_absolute_error")
        print(f"  CV R2:  {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
        print(f"  CV MAE: {-cv_neg_mae.mean():.4f} +/- {cv_neg_mae.std():.4f}")

        # Fit on full training set
        model.fit(Xtr, y_train)
        trained_models[name] = {
            "model": model,
            "needs_scaling": config["needs_scaling"],
        }

        y_pred = model.predict(Xte)
        y_pred = np.maximum(y_pred, 0)  # demand cannot be negative

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape_val = _mape(y_test, y_pred)

        results[name] = {
            "mae": mae, "rmse": rmse, "r2": r2, "mape": mape_val,
            "y_pred": y_pred,
            "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        }

        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  MAPE: {mape_val:.2f}%")

    # --- Spatial cross-validation (on best model) ---
    if use_spatial_cv:
        print("\n" + "=" * 70)
        print("SPATIAL CROSS-VALIDATION (LEAVE-ONE-ZONE-OUT)")
        print("=" * 70)

        best_name = max(results, key=lambda n: results[n]["r2"])
        best_config = models_config[best_name]

        zone_mae_scores = []
        zone_r2_scores = []

        for train_idx, test_idx in spatial_cv_split(df_feat):
            X_tr = df_feat.iloc[train_idx][feature_cols].values.astype(float)
            X_te = df_feat.iloc[test_idx][feature_cols].values.astype(float)
            y_tr = df_feat.iloc[train_idx]["demand_count"].values.astype(float)
            y_te = df_feat.iloc[test_idx]["demand_count"].values.astype(float)

            if best_config["needs_scaling"]:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            from sklearn.base import clone
            m = clone(best_config["model"])
            m.fit(X_tr, y_tr)
            preds = np.maximum(m.predict(X_te), 0)

            zone_mae_scores.append(mean_absolute_error(y_te, preds))
            if len(y_te) > 1:
                zone_r2_scores.append(r2_score(y_te, preds))

        print(f"  Spatial CV MAE: {np.mean(zone_mae_scores):.4f} +/- {np.std(zone_mae_scores):.4f}")
        print(f"  Spatial CV R2:  {np.mean(zone_r2_scores):.4f} +/- {np.std(zone_r2_scores):.4f}")

        results["__spatial_cv__"] = {
            "model": best_name,
            "spatial_mae_mean": np.mean(zone_mae_scores),
            "spatial_mae_std": np.std(zone_mae_scores),
            "spatial_r2_mean": np.mean(zone_r2_scores),
            "spatial_r2_std": np.std(zone_r2_scores),
        }

    # --- Model comparison ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k not in ("y_pred",)}
        for name, r in results.items() if not name.startswith("__")
    }).T.round(4)
    print(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))

    best_name = max(
        (n for n in results if not n.startswith("__")),
        key=lambda n: results[n]["r2"],
    )
    best_r2 = results[best_name]["r2"]
    print(f"\nBest model: {best_name} (R2 = {best_r2:.4f})")

    best_info = trained_models[best_name]
    joblib.dump(best_info["model"], os.path.join(OUTPUTS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(OUTPUTS_DIR, "feature_names.joblib"))
    joblib.dump(kmeans, os.path.join(OUTPUTS_DIR, "zone_kmeans.joblib"))

    # --- Plots ---
    _plot_actual_vs_predicted(y_test, results, best_name)
    _plot_residuals(y_test, results, best_name)
    _plot_feature_importance(trained_models, feature_cols, best_name)

    return results, trained_models, X_test, X_test_scaled, y_test, feature_cols, scaler


def _plot_actual_vs_predicted(y_test, results, best_name):
    """Scatter plot of actual vs predicted demand."""
    y_pred = results[best_name]["y_pred"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, c="#3B6FD4")
    lim = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Actual demand")
    ax.set_ylabel("Predicted demand")
    ax.set_title(f"Actual vs predicted demand - {best_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)
    print("Saved actual vs predicted plot.")


def _plot_residuals(y_test, results, best_name):
    """Residual distribution plot."""
    y_pred = results[best_name]["y_pred"]
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, c="#3B6FD4")
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Predicted demand")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, color="#E8C230", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7)

    fig.suptitle(f"Residual analysis - {best_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "residuals.png"), dpi=150)
    plt.close(fig)
    print("Saved residual plots.")


def _plot_feature_importance(trained_models, feature_cols, best_name):
    """Plot feature importance for tree-based models."""
    info = trained_models[best_name]
    model = info["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"No feature_importances_ for {best_name}, skipping plot.")
        return

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color="#3B6FD4", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance - {best_name}")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("Saved feature importance plot.")


if __name__ == "__main__":
    from data_loader import load_demand_data
    df = load_demand_data()
    train_and_evaluate(df)
