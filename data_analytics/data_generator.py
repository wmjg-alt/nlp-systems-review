"""
Data Analytics Series: Step 0 - Synthetic Data Generation

Focus:
Creating a "Real-World Messy" dataset. We demonstrate how to programmatically 
inject structural traps (nesting), schema drift (key variance), and 
statistical correlations that require advanced Pandas skills to uncover.

Traps Included:
- Structural: Root metadata wrap and nested payloads.
- Schema: Inconsistent keys for the same data point (Price vs MSRP).
- Temporal: Mixed date formats and Unix epoch timestamps.
- Linguistic: Dirty strings with Unicode noise.
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- TUNABLE PARAMETERS ---
TOTAL_RECORDS = 1000
OUTPUT_FILE = "messy_inventory.json"

# Schema Drift: What % of records should use the "Complex/Nested" pricing key?
NESTED_PRICING_PROB = 0.30

# Missing Data: A hidden bias - Electronics will fail more often
CATEGORY_FAULT_RATES = {
    "Electronics": 0.15,  # 15% missing ratings
    "Books": 0.02,
    "Home & Kitchen": 0.05
}

def generate_dirty_id(base_id: int) -> str:
    """Injects Unicode noise and whitespace to trip up exact-match joins."""
    traps = [
        f"ID_{base_id} ",         # Trailing space
        f" ID_{base_id}",         # Leading space
        f"ID_{base_id}\u200b",    # Zero-width space
        f"ID_{base_id}"           # Clean
    ]
    return random.choice(traps)

def generate_mixed_date(base_date: datetime) -> str:
    """Randomly selects a format to simulate heterogeneous data sources."""
    formats = [
        lambda d: d.strftime("%Y-%m-%d"),         # ISO
        lambda d: d.strftime("%m/%d/%Y"),         # US Standard
        lambda d: str(int(d.timestamp()))         # Unix Epoch
    ]
    return random.choice(formats)(base_date)

def create_record(idx: int) -> dict:
    """Builds a single product record with potential schema drift."""
    category = random.choice(list(CATEGORY_FAULT_RATES.keys()))
    base_price = round(np.random.gamma(2, 50) + 10, 2)
    
    # 1. Simulate Schema Drift (Inconsistent Keys)
    # Junior engineers expect 'price' everywhere. We bury it for 30% of docs.
    if random.random() < NESTED_PRICING_PROB:
        price_data = {"pricing_info": {"msrp_usd": base_price, "currency": "USD"}}
    else:
        price_data = {"price": f"${base_price}"} # String with symbol to force cleaning

    # 2. Statistical Poisoning (Category-specific missingness)
    rating = None
    if random.random() > CATEGORY_FAULT_RATES[category]:
        rating = round(random.uniform(3.0, 5.0), 1)

    # 3. Temporal Drift
    launch_date = generate_mixed_date(datetime(2023, 1, 1) + timedelta(days=idx))

    # Assemble the base record
    record = {
        "metadata": {
            "internal_id": generate_dirty_id(idx),
            "source_node": random.choice(["AWS-USE1", "GCP-USW2", "LOCAL-DB"])
        },
        "payload": {
            "product_name": f"Product_{idx}",
            "dept": category,
            "availability": random.choice(["In Stock", "Out of Stock", "Backorder"]),
            "metrics": {
                "rating": rating,
                "weight_oz": random.randint(5, 500)
            },
            "history": {
                "created_at": launch_date
            }
        }
    }
    
    # Merge the inconsistent pricing data
    record["payload"].update(price_data)
    return record

def main():
    print("🚀 Initializing Synthetic Data Generation...")
    print(f"Target: {TOTAL_RECORDS} records with intentional sabotage.")

    # TRAP 1: The Root Wrap
    # We don't return a list. We return an object with a 'Ghost Tag' and a deep root.
    final_output = {
        "status": "SUCCESS",
        "generated_at": datetime.now().isoformat(),
        "total_count": TOTAL_RECORDS,
        "data_lake_exports": {
            "inventory_items": [create_record(i) for i in range(TOTAL_RECORDS)]
        }
    }

    print("\n--- GENERATION LOGS: Sabotage Applied ---")
    print(f"[*] Nested Structure: Data is buried under 'data_lake_exports' -> 'inventory_items'")
    print(f"[*] Schema Drift: ~{NESTED_PRICING_PROB*100}% of records use nested 'pricing_info' keys.")
    print(f"[*] Statistical Bias: 'Electronics' has a {CATEGORY_FAULT_RATES['Electronics']*100}% fault rate.")
    print(f"[*] Temporal Chaos: 3 different date formats (including Unix) injected.")
    print(f"[*] Dirty Strings: IDs contains zero-width spaces and irregular padding.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print(f"\n✅ File '{OUTPUT_FILE}' written successfully.")
    print("Next Step: Try loading this into Pandas without manually editing the file.")

def lets_try():
    print()
    try: 
        df = pd.read_json(OUTPUT_FILE)
        print(df.head())
    except pd.FileReadException as e:
        raise(e)

if __name__ == "__main__":
    main()
    lets_try()