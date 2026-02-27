"""
Data Analytics Series: Step 1 - The Data Janitor (Audit & ETL)

Focus:
Converting deeply nested, heterogeneous JSON into a "Tidy" DataFrame while 
performing a high-level data audit. We expose the structural sabotage from 
the generator and quantify the effort required to fix it.

Audit Points:
- Structural Navigability: Identifying the "Root Wrap" and nested paths.
- ID Integrity: Quantifying invisible Unicode noise that causes join failures.
- Schema Drift: Measuring the impact of inconsistent keys (Price vs MSRP).
- Temporal Logic: Detecting mixed-format date failures.
"""

import pandas as pd
import numpy as np
import json
import logging
import re
from pathlib import Path

# --- TUNABLE PARAMETERS ---
INPUT_FILE = "messy_inventory.json"
CLEANING_REGEX = r'[^\x20-\x7E]' # Regex to identify non-printable/Unicode noise

# Configure logging for professional feedback
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DataJanitor")

def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Handles heterogeneous temporal formats: ISO, US, and Unix."""
    # Logic: Convert to numeric first to isolate Epoch timestamps
    as_numeric = pd.to_numeric(series, errors='coerce')
    unix_mask = as_numeric.notna()
    
    result = pd.to_datetime(series, errors='coerce')
    # Apply Unix logic ONLY where numeric values were found
    result[unix_mask] = pd.to_datetime(as_numeric[unix_mask], unit='s')
    
    return result

def run_etl_pipeline():
    logger.info("=== STEP 2: DATA AUDIT & STRUCTURAL CLEANING ===\n")

    if not Path(INPUT_FILE).exists():
        logger.error(f"❌ Error: {INPUT_FILE} not found. Run data_generator.py first.")
        return

    # --- 1. THE INGEST TRAP ---
    logger.info("[Step 1]: Navigating nested JSON...")
    with open(INPUT_FILE, 'r') as f:
        raw_obj = json.load(f)
    
    # Audit the root structure
    root_keys = list(raw_obj.keys())
    logger.info(f"   Detected Root Keys: {root_keys}")
    
    # Extract data buried in 'data_lake_exports' -> 'inventory_items'
    items = raw_obj['data_lake_exports']['inventory_items']
    initial_count = len(items)
    logger.info(f"   Successfully extracted {initial_count} records from nested payload.")

    # Flatten the JSON using a professional recursive flattener
    df = pd.json_normalize(items, sep='_')

    # --- 2. ID INTEGRITY AUDIT ---
    logger.info("\n[Step 2]: Auditing Identifier Integrity...")
    
    # Identify records with non-printable characters or whitespace padding
    id_col = 'metadata_internal_id'
    noise_mask = df[id_col].str.contains(CLEANING_REGEX, regex=True)
    whitespace_mask = (df[id_col].str.startswith(' ')) | (df[id_col].str.endswith(' '))
    
    affected_ids = df[noise_mask | whitespace_mask]
    logger.info(f"   ⚠️  ID SABOTAGE: {len(affected_ids)} records ({len(affected_ids)/initial_count:.1%}) "
                "had invisible Unicode noise or whitespace that would break database joins.")
    
    # Apply Fix
    df[id_col] = df[id_col].str.replace(CLEANING_REGEX, '', regex=True).str.strip()

    # --- 3. SCHEMA DRIFT ANALYSIS ---
    logger.info("\n[Step 3]: Analyzing Schema Drift (Price Unification)...")
    
    # Identify which records used which key
    std_price_count = df['payload_price'].notna().sum()
    nested_msrp_count = df['payload_pricing_info_msrp_usd'].notna().sum()
    
    logger.info(f"   Key 'payload_price' count: {std_price_count}")
    logger.info(f"   Key 'payload_pricing_info_msrp_usd' count: {nested_msrp_count}")

    # Professional Fix: Vectorized cleaning and Coalescing
    def clean_currency_str(val):
        if isinstance(val, str):
            return float(re.sub(r'[^\d.]', '', val)) # Remove symbols
        return val

    price_alt_1 = df['payload_price'].apply(clean_currency_str)
    price_alt_2 = df['payload_pricing_info_msrp_usd']
    
    # Coalesce: Take the precise MSRP first, fallback to standard price string
    df['price_unified'] = price_alt_2.combine_first(price_alt_1)
    
    # Validation: Did any prices fail to resolve?
    failed_prices = df['price_unified'].isna().sum()
    if failed_prices > 0:
        logger.warning(f"   🚨 ALERT: {failed_prices} records failed price unification.")

    # --- 4. TEMPORAL AUDIT ---
    logger.info("\n[Step 4]: Parsing and Auditing Mixed-Format Dates...")
    df['launch_date_dt'] = parse_mixed_dates(df['payload_history_created_at'])
    
    # Check for NaT (Not a Time) failures
    date_failures = df['launch_date_dt'].isna().sum()
    logger.info(f"   Success: {initial_count - date_failures} / {initial_count} dates parsed.")
    if date_failures > 0:
        logger.warning(f"   ⚠️  Dropped {date_failures} records due to unparseable date formats.")

    # --- 5. FINAL HEALTH REPORT ---
    # Construct the final "Tidy" DataFrame
    tidy_df = df[[
        'metadata_internal_id', 
        'payload_product_name', 
        'payload_dept', 
        'price_unified', 
        'payload_metrics_rating', 
        'launch_date_dt'
    ]].copy()
    
    # Apply standard column names
    tidy_df.columns = ['id', 
                       'name', 
                       'dept', 
                       'price', 
                       'rating', 
                       'launch_date']
    
    # Final data-loss check
    tidy_df = tidy_df.dropna(subset=['id', 'launch_date'])
    final_count = len(tidy_df)

    logger.info("\n" + "="*40)
    logger.info("FINAL DATA HEALTH REPORT")
    logger.info("="*40)
    logger.info(f"Initial Records:      {initial_count}")
    logger.info(f"Usable Records:       {final_count} ({final_count/initial_count:.1%})")
    logger.info(f"Total Records Lost:   {initial_count - final_count}")
    logger.info(f"Missing Ratings:      {tidy_df['rating'].isna().sum()} (Statistical Bias preserved)")
    logger.info(f"Unique Departments:   {tidy_df['dept'].nunique()}")
    logger.info("="*40)

    logger.info("\n[Step 5]: Clean Dataset Preview (Sorted by ID):")
    logger.info(tidy_df.sort_values('id').head(5).to_string(index=False))
    
    return tidy_df

if __name__ == "__main__":
    clean_df = run_etl_pipeline()
    if clean_df is not None:
        # Save as pickle to preserve the specific DateTime and Float types
        clean_df.to_pickle("inventory_tidy.pkl")
        logger.info("\n✅ ETL Complete. Tidy data saved to 'inventory_tidy.pkl'.")