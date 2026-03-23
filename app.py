"""Streamlit dashboard for geospatial ride demand forecasting."""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import generate_demand_data, CALGARY_ZONES
from src.model import engineer_features, _get_models, _mape, RANDOM_STATE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Geospatial demand forecast", layout="wide")

GOLD = "#E8C230"
NAVY = "#3B6FD4"


@st.cache_data
def load_data():
    path = os.path.join(PROJECT_DIR, "data", "ride_demand.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return generate_demand_data()


@st.cache_resource
def train_models(df):
    """Train all models and return results."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df_feat, feature_cols, kmeans = engineer_features(df)

    X = df_feat[feature_cols].values.astype(float)
    y = df_feat["demand_count"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = _get_models()
    results = {}
    trained = {}

    for name, config in models_config.items():
        model = config["model"]
        Xtr = X_train_scaled if config["needs_scaling"] else X_train
        Xte = X_test_scaled if config["needs_scaling"] else X_test

        model.fit(Xtr, y_train)
        y_pred = np.maximum(model.predict(Xte), 0)

        results[name] = {
            "y_pred": y_pred,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": _mape(y_test, y_pred),
        }
        trained[name] = {
            "model": model,
            "needs_scaling": config["needs_scaling"],
        }

    return results, trained, X_test, X_test_scaled, y_test, feature_cols, scaler, df_feat, kmeans


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Demand heatmap", "Zone forecast timeline", "Feature importance",
     "Model comparison", "Peak demand alerts"],
)

df = load_data()
results, trained, X_test, X_test_scaled, y_test, feature_cols, scaler, df_feat, kmeans = train_models(df)

best_name = max(results, key=lambda n: results[n]["r2"])


# ---------------------------------------------------------------------------
# Page: Demand heatmap
# ---------------------------------------------------------------------------
if page == "Demand heatmap":
    st.title("Predicted demand heatmap")
    st.markdown("Average predicted ride demand by zone and hour across Calgary.")

    selected_hour = st.slider("Select hour of day", 0, 23, 8)

    # Compute average demand per zone for the selected hour
    hour_df = df_feat[df_feat["hour"] == selected_hour]
    zone_demand = hour_df.groupby("zone_id")["demand_count"].mean().reset_index()
    zone_demand.columns = ["zone_id", "avg_demand"]

    # Add coordinates
    zone_demand["latitude"] = zone_demand["zone_id"].map(
        lambda z: CALGARY_ZONES.get(z, (51.05, -114.07))[0]
    )
    zone_demand["longitude"] = zone_demand["zone_id"].map(
        lambda z: CALGARY_ZONES.get(z, (51.05, -114.07))[1]
    )

    fig = px.scatter_mapbox(
        zone_demand,
        lat="latitude",
        lon="longitude",
        size="avg_demand",
        color="avg_demand",
        hover_name="zone_id",
        hover_data={"avg_demand": ":.1f", "latitude": ":.4f", "longitude": ":.4f"},
        color_continuous_scale=["#162240", NAVY, GOLD, "#ef4444"],
        size_max=30,
        zoom=10.5,
        center={"lat": 51.05, "lon": -114.07},
        mapbox_style="carto-darkmatter",
        title=f"Demand heatmap at {selected_hour:02d}:00",
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Demand by hour bar chart
    st.subheader("Demand distribution across hours")
    hourly_avg = df_feat.groupby("hour")["demand_count"].mean().reset_index()
    fig2 = px.bar(
        hourly_avg, x="hour", y="demand_count",
        color="demand_count",
        color_continuous_scale=[NAVY, GOLD],
        labels={"demand_count": "Avg demand", "hour": "Hour"},
    )
    fig2.add_vline(x=selected_hour, line_dash="dash", line_color="red",
                   annotation_text=f"Selected: {selected_hour:02d}:00")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Zone forecast timeline
# ---------------------------------------------------------------------------
elif page == "Zone forecast timeline":
    st.title("Zone-level forecast timeline")
    st.markdown("View demand patterns over a 24-hour cycle for any Calgary zone.")

    zone_list = sorted(df["zone_id"].unique())
    selected_zone = st.selectbox("Select zone", zone_list)

    zone_data = df_feat[df_feat["zone_id"] == selected_zone]

    # Average demand by hour for selected zone
    zone_hourly = zone_data.groupby("hour")["demand_count"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    zone_hourly.columns = ["hour", "mean_demand", "std_demand", "min_demand", "max_demand"]
    zone_hourly["lower"] = (zone_hourly["mean_demand"] - zone_hourly["std_demand"]).clip(0)
    zone_hourly["upper"] = zone_hourly["mean_demand"] + zone_hourly["std_demand"]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([zone_hourly["hour"], zone_hourly["hour"][::-1]]),
        y=pd.concat([zone_hourly["upper"], zone_hourly["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(59,111,212,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="+/- 1 std",
    ))

    fig.add_trace(go.Scatter(
        x=zone_hourly["hour"], y=zone_hourly["mean_demand"],
        mode="lines+markers",
        line=dict(color=GOLD, width=3),
        marker=dict(size=6),
        name="Mean demand",
    ))

    fig.update_layout(
        xaxis_title="Hour of day", yaxis_title="Demand count",
        title=f"24-hour demand profile: {selected_zone}",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Day of week pattern
    st.subheader("Demand by day of week")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    zone_daily = zone_data.groupby("day_of_week")["demand_count"].mean().reset_index()
    zone_daily["day_name"] = zone_daily["day_of_week"].map(lambda d: day_names[d])

    fig2 = px.bar(
        zone_daily, x="day_name", y="demand_count",
        color="demand_count",
        color_continuous_scale=[NAVY, GOLD],
        labels={"demand_count": "Avg demand", "day_name": "Day"},
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Zone stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg demand", f"{zone_data['demand_count'].mean():.1f}")
    col2.metric("Max demand", f"{zone_data['demand_count'].max()}")
    col3.metric("Records", f"{len(zone_data):,}")
    col4.metric("Peak hour", f"{zone_hourly.loc[zone_hourly['mean_demand'].idxmax(), 'hour']:02d}:00")


# ---------------------------------------------------------------------------
# Page: Feature importance
# ---------------------------------------------------------------------------
elif page == "Feature importance":
    st.title("Feature importance analysis")

    info = trained[best_name]
    model = info["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=[NAVY, GOLD],
            title=f"Feature importance ({best_name})",
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{best_name} does not have feature importances. Showing coefficient magnitudes.")
        if hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient magnitude": coefs,
            }).sort_values("Coefficient magnitude", ascending=True)
            fig = px.bar(
                imp_df, x="Coefficient magnitude", y="Feature", orientation="h",
                color="Coefficient magnitude",
                color_continuous_scale=[NAVY, GOLD],
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

    # Feature correlation with demand
    st.subheader("Feature correlations with demand")
    corr_with_demand = df_feat[feature_cols + ["demand_count"]].corr()["demand_count"].drop("demand_count")
    corr_df = corr_with_demand.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    corr_df = corr_df.sort_values("Correlation", ascending=True)

    fig2 = px.bar(
        corr_df, x="Correlation", y="Feature", orientation="h",
        color="Correlation",
        color_continuous_scale=["#ef4444", "#cccccc", "#22c55e"],
        color_continuous_midpoint=0,
        title="Pearson correlation with demand count",
    )
    fig2.update_layout(height=550)
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model comparison
# ---------------------------------------------------------------------------
elif page == "Model comparison":
    st.title("Model comparison")

    # Metrics table
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "y_pred"}
        for name, r in results.items()
    }).T.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    # Bar chart of R-squared
    st.subheader("R-squared comparison")
    r2_df = pd.DataFrame({
        "Model": list(results.keys()),
        "R2": [r["r2"] for r in results.values()],
    }).sort_values("R2", ascending=True)

    fig = px.bar(
        r2_df, x="R2", y="Model", orientation="h",
        color="R2", color_continuous_scale=[NAVY, GOLD],
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Actual vs predicted for best model
    st.subheader(f"Actual vs predicted ({best_name})")
    y_pred = results[best_name]["y_pred"]
    scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    fig2 = px.scatter(
        scatter_df, x="Actual", y="Predicted", opacity=0.3,
        color_discrete_sequence=[NAVY],
    )
    lim = max(y_test.max(), y_pred.max()) * 1.05
    fig2.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines",
        line=dict(dash="dash", color="red"), name="Perfect",
    ))
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # MAE comparison
    st.subheader("MAE comparison")
    mae_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE": [r["mae"] for r in results.values()],
    }).sort_values("MAE")

    fig3 = px.bar(
        mae_df, x="Model", y="MAE",
        color="MAE", color_continuous_scale=[GOLD, NAVY],
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Peak demand alerts
# ---------------------------------------------------------------------------
elif page == "Peak demand alerts":
    st.title("Peak demand alerts")
    st.markdown("Identify zones and time slots where demand exceeds configurable thresholds.")

    threshold = st.slider("Demand threshold (rides/zone/hour)", 5, 40, 15)

    # Aggregate average demand by zone and hour
    pivot = df_feat.groupby(["zone_id", "hour"])["demand_count"].mean().reset_index()
    alerts = pivot[pivot["demand_count"] >= threshold].sort_values(
        "demand_count", ascending=False
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Alert count", f"{len(alerts):,}")
    col2.metric("Zones affected", f"{alerts['zone_id'].nunique()}")
    col3.metric("Peak demand", f"{alerts['demand_count'].max():.1f}" if len(alerts) > 0 else "N/A")

    if len(alerts) > 0:
        # Heatmap: zone vs hour
        st.subheader("Zone-hour demand heatmap")
        pivot_wide = df_feat.groupby(["zone_id", "hour"])["demand_count"].mean().reset_index()
        heatmap_data = pivot_wide.pivot(index="zone_id", columns="hour", values="demand_count")
        heatmap_data = heatmap_data.fillna(0)

        # Sort zones by total demand
        heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values(ascending=False).index]

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour", y="Zone", color="Avg demand"),
            color_continuous_scale=["#162240", NAVY, GOLD, "#ef4444"],
            aspect="auto",
            title="Demand heatmap: zones vs hours",
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        # Top alerts table
        st.subheader("Top peak demand alerts")
        top_alerts = alerts.head(20).copy()
        top_alerts["demand_count"] = top_alerts["demand_count"].round(1)
        top_alerts.columns = ["Zone", "Hour", "Avg demand"]
        st.dataframe(top_alerts, use_container_width=True, hide_index=True)
    else:
        st.info("No zones exceed the selected threshold. Try lowering it.")
