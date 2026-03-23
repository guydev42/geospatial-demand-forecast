"""Generate synthetic ride demand data and zone reference table, save to CSV."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_demand_data, generate_zones_data

DATA_DIR = os.path.dirname(__file__)


def main():
    # Generate ride demand data
    df = generate_demand_data(n_samples=20000, random_state=42)
    demand_path = os.path.join(DATA_DIR, "ride_demand.csv")
    df.to_csv(demand_path, index=False)

    print(f"Saved {len(df)} demand records to {demand_path}")
    print(f"Zones: {df['zone_id'].nunique()}")
    print(f"Demand range: {df['demand_count'].min()} - {df['demand_count'].max()}")
    print(f"Columns: {list(df.columns)}")

    # Generate zones reference table
    zones_df = generate_zones_data()
    zones_path = os.path.join(DATA_DIR, "zones.csv")
    zones_df.to_csv(zones_path, index=False)

    print(f"\nSaved {len(zones_df)} zones to {zones_path}")
    print(zones_df.to_string(index=False))


if __name__ == "__main__":
    main()
