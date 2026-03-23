"""
Synthetic ride demand data generator for Calgary zones.
Produces 20K records across Calgary neighborhoods with geographic, temporal,
and contextual features for demand forecasting.
"""

import numpy as np
import pandas as pd


# Calgary neighborhood centroids (real approximate coordinates)
CALGARY_ZONES = {
    "Beltline": (51.0407, -114.0700),
    "Downtown Commercial Core": (51.0477, -114.0630),
    "Kensington": (51.0530, -114.0860),
    "Bridgeland": (51.0550, -114.0440),
    "Inglewood": (51.0370, -114.0270),
    "Mission": (51.0350, -114.0660),
    "Marda Loop": (51.0260, -114.0960),
    "17th Avenue SW": (51.0380, -114.0800),
    "Eau Claire": (51.0530, -114.0710),
    "Victoria Park": (51.0410, -114.0550),
    "Sunnyside": (51.0580, -114.0830),
    "Hillhurst": (51.0570, -114.0920),
    "Mount Royal": (51.0310, -114.0830),
    "Bankview": (51.0330, -114.0930),
    "Cliff Bungalow": (51.0340, -114.0770),
    "Sunalta": (51.0430, -114.0920),
    "Ramsay": (51.0380, -114.0350),
    "East Village": (51.0460, -114.0490),
    "Chinatown": (51.0510, -114.0670),
    "Crescent Heights": (51.0580, -114.0630),
    "University District": (51.0770, -114.1300),
    "Bowness": (51.0870, -114.1860),
    "Signal Hill": (51.0180, -114.1700),
    "Crowfoot": (51.1230, -114.1600),
    "Marlborough": (51.0540, -113.9720),
    "Forest Lawn": (51.0440, -113.9720),
    "Coventry Hills": (51.1540, -114.0840),
    "McKenzie Towne": (50.9210, -113.9580),
    "Shawnessy": (50.9110, -114.0670),
    "Tuscany": (51.1290, -114.2240),
}

DOWNTOWN_COORDS = (51.0477, -114.0630)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Compute haversine distance in kilometers between two lat/lon points."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def generate_demand_data(n_samples=20000, random_state=42):
    """
    Generate synthetic ride demand data for Calgary zones.

    Features:
        zone_id             - neighborhood name
        latitude            - zone centroid latitude
        longitude           - zone centroid longitude
        hour                - hour of day (0-23)
        day_of_week         - day (0=Mon, 6=Sun)
        month               - month of year (1-12)
        is_holiday          - binary holiday flag
        temperature         - temperature in Celsius
        precipitation       - precipitation in mm
        event_nearby        - binary flag for nearby event
        population_density  - people per sq km for zone
        num_restaurants     - restaurant count in zone
        transit_stops_nearby - transit stop count in zone

    Target:
        demand_count        - number of ride requests per zone per hour

    Returns:
        pd.DataFrame with all features and target
    """
    rng = np.random.RandomState(random_state)
    zone_names = list(CALGARY_ZONES.keys())
    n_zones = len(zone_names)

    # Assign static zone attributes
    zone_pop_density = {
        name: rng.uniform(800, 12000) if i < 20 else rng.uniform(300, 3000)
        for i, name in enumerate(zone_names)
    }
    # Downtown-adjacent zones get higher density
    for name in ["Beltline", "Downtown Commercial Core", "Eau Claire",
                 "East Village", "Chinatown", "Victoria Park"]:
        zone_pop_density[name] = rng.uniform(8000, 15000)

    zone_restaurants = {
        name: int(rng.uniform(5, 80)) if i < 20 else int(rng.uniform(2, 25))
        for i, name in enumerate(zone_names)
    }
    for name in ["Beltline", "Downtown Commercial Core", "17th Avenue SW",
                 "Kensington", "Mission"]:
        zone_restaurants[name] = int(rng.uniform(60, 150))

    zone_transit = {
        name: int(rng.uniform(2, 20)) if i < 20 else int(rng.uniform(1, 8))
        for i, name in enumerate(zone_names)
    }
    for name in ["Downtown Commercial Core", "Beltline", "Victoria Park",
                 "Bridgeland", "Sunnyside"]:
        zone_transit[name] = int(rng.uniform(15, 35))

    records = []

    for _ in range(n_samples):
        zone_idx = rng.randint(0, n_zones)
        zone_name = zone_names[zone_idx]
        lat, lon = CALGARY_ZONES[zone_name]

        # Add small jitter to coordinates
        lat += rng.normal(0, 0.002)
        lon += rng.normal(0, 0.003)

        hour = rng.randint(0, 24)
        day_of_week = rng.randint(0, 7)
        month = rng.randint(1, 13)
        is_holiday = int(rng.random() < 0.05)

        # Temperature depends on month (Calgary climate)
        month_temps = {
            1: -10, 2: -7, 3: -2, 4: 5, 5: 11, 6: 16,
            7: 20, 8: 19, 9: 13, 10: 6, 11: -3, 12: -8,
        }
        temperature = month_temps[month] + rng.normal(0, 5)

        # Precipitation depends on month
        month_precip = {
            1: 1.0, 2: 0.8, 3: 1.5, 4: 2.5, 5: 5.0, 6: 8.0,
            7: 6.0, 8: 5.5, 9: 4.0, 10: 2.0, 11: 1.2, 12: 1.0,
        }
        precipitation = max(0, rng.exponential(month_precip[month]))

        event_nearby = int(rng.random() < 0.08)

        pop_density = zone_pop_density[zone_name]
        restaurants = zone_restaurants[zone_name]
        transit = zone_transit[zone_name]

        # Distance from downtown
        dist_dt = _haversine_km(lat, lon, DOWNTOWN_COORDS[0], DOWNTOWN_COORDS[1])

        # Compute demand count based on realistic factors
        base_demand = 3.0

        # Zone popularity (closer to downtown = more demand)
        base_demand += max(0, 8 - dist_dt * 0.6)

        # Population density effect
        base_demand += pop_density / 3000.0

        # Restaurant density effect
        base_demand += restaurants / 30.0

        # Transit proximity effect
        base_demand += transit / 10.0

        # Hour of day effect (rush hours and nightlife peaks)
        hour_multipliers = [
            0.2, 0.15, 0.10, 0.08, 0.10, 0.20,  # 0-5
            0.50, 1.20, 1.80, 1.40, 1.00, 1.10,  # 6-11
            1.30, 1.20, 1.00, 1.20, 1.60, 1.90,  # 12-17
            1.50, 1.20, 1.00, 0.90, 0.70, 0.40,  # 18-23
        ]
        base_demand *= hour_multipliers[hour]

        # Weekend effect (more late night, less morning rush)
        if day_of_week >= 5:
            if hour >= 20 or hour <= 2:
                base_demand *= 1.5
            elif 7 <= hour <= 9:
                base_demand *= 0.6

        # Event effect
        if event_nearby:
            base_demand *= 1.8

        # Holiday effect
        if is_holiday:
            base_demand *= 0.7

        # Weather: rain increases demand, extreme cold reduces it
        if precipitation > 5:
            base_demand *= 1.3
        if temperature < -15:
            base_demand *= 0.6
        elif temperature < -5:
            base_demand *= 0.85

        # Month seasonality (summer has more activity)
        if month in [6, 7, 8]:
            base_demand *= 1.15
        elif month in [12, 1, 2]:
            base_demand *= 0.85

        # Add noise
        demand_count = max(0, int(rng.poisson(max(0.5, base_demand))))

        records.append({
            "zone_id": zone_name,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_holiday": is_holiday,
            "temperature": round(temperature, 1),
            "precipitation": round(precipitation, 1),
            "event_nearby": event_nearby,
            "population_density": round(pop_density, 0),
            "num_restaurants": restaurants,
            "transit_stops_nearby": transit,
            "demand_count": demand_count,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def generate_zones_data():
    """Generate a zones reference table with static attributes."""
    rows = []
    rng = np.random.RandomState(42)
    for name, (lat, lon) in CALGARY_ZONES.items():
        dist = _haversine_km(lat, lon, DOWNTOWN_COORDS[0], DOWNTOWN_COORDS[1])
        rows.append({
            "zone_id": name,
            "latitude": lat,
            "longitude": lon,
            "distance_to_downtown_km": round(dist, 2),
        })
    return pd.DataFrame(rows)


def load_demand_data(filepath="data/ride_demand.csv"):
    """Load ride demand data from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    print(f"Zones: {df['zone_id'].nunique()}")
    print(f"Demand range: {df['demand_count'].min()} - {df['demand_count'].max()}")
    return df


if __name__ == "__main__":
    df = generate_demand_data()
    print(f"Generated {len(df)} ride demand records")
    print(f"Zones: {df['zone_id'].nunique()}")
    print(f"Demand stats:\n{df['demand_count'].describe().round(2)}")
