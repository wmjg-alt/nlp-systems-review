"""
Data Analytics Series: Step 2 - Business Intelligence & Statistical Outliers

Focus:
Transforming tidy data into actionable insights. We move from row-level 
cleaning to group-level analysis, identifying performance anomalies 
that global averages would miss.

Techniques Demonstrated:
1. Relational Joins: Merging our tidy inventory with a "Cost Master List".
2. Multi-Metric Aggregation: Using .agg() to calculate complex department stats.
3. Window Functions: Using .transform() for category-specific Z-scores.
4. Logic Traps: Identifying 'Loss Leaders' (Negative Profit) hidden in high revenue.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# --- TUNABLE PARAMETERS ---
INPUT_FILE = "inventory_tidy.pkl"
# Number of Standard Deviations to flag a "Price Outlier"
Z_THRESHOLD = 2.0 
# Target Margin - Flag any department falling below this %
MIN_TARGET_MARGIN = 15.0

# Configure logging for professional feedback
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("BusinessAnalyst")

def run_business_audit():
    logger.info("=== STEP 3: BUSINESS AUDIT & OUTLIER DETECTION ===\n")

    if not Path(INPUT_FILE).exists():
        logger.error(f"❌ Error: {INPUT_FILE} not found. Run data_processing_1.py first.")
        return

    # Load the high-fidelity pickle from the previous step
    df = pd.read_pickle(INPUT_FILE)
    
    # --- 1. RELATIONAL DATA HANDLING (The Join) ---
    logger.info("[Step 1]: Simulating Relational Join with Cost Master...")
    
    # We simulate a 'Cost List' that might come from a separate Finance DB
    # Most items cost 70% of their price, but we add some variance
    np.random.seed(42)
    df['unit_cost'] = df['price'] * np.random.uniform(0.5, 0.9, size=len(df))
    
    # Inject an "Economic Trap": Product ID_100 is a Loss Leader (Cost > Price)
    df.loc[df['id'] == 'ID_100', 'unit_cost'] = df.loc[df['id'] == 'ID_100', 'price'] * 1.5

    # --- 2. WINDOW FUNCTIONS (The Intra-Category Z-Score) ---
    logger.info("[Step 2]: Calculating Category-Relative Price Z-Scores...")
    
    # Senior Strategy: Normalize price relative to the DEPARTMENT, not the whole store.
    # This prevents 'Electronics' from making all 'Books' look like outliers.
    dept_group = df.groupby('dept')['price']
    
    df['dept_avg_price'] = dept_group.transform('mean')
    df['dept_std_price'] = dept_group.transform('std')
    
    # Z-Score = (Value - Mean) / StdDev
    df['price_zscore'] = (df['price'] - df['dept_avg_price']) / df['dept_std_price']
    
    outlier_count = (df['price_zscore'].abs() > Z_THRESHOLD).sum()
    logger.info(f"   Detected {outlier_count} statistical price outliers relative to dept peers.")

    # --- 3. MULTI-METRIC AGGREGATION ---
    logger.info("\n[Step 3]: Generating Department Performance Matrix...")
    
    # Calculate Unit Economics
    df['unit_profit'] = df['price'] - df['unit_cost']
    df['margin_pct'] = (df['unit_profit'] / df['price']) * 100

    # Professional .agg() pattern
    dept_stats = df.groupby('dept').agg(
        total_skus=('id', 'count'),
        avg_price=('price', 'mean'),
        avg_rating=('rating', 'mean'),
        avg_margin=('margin_pct', 'mean'),
        total_potential_loss=('unit_profit', lambda x: x[x < 0].sum())
    ).round(2)

    logger.info("\n--- DEPARTMENT SUMMARY ---")
    logger.info(dept_stats.to_string())

    # --- 4. THE INSIGHT AUDIT (Business Logic) ---
    logger.info("\n[Step 4]: Running High-Value Insight Audit...")

    # A. Finding the "Margin Bleed"
    underperformers = dept_stats[dept_stats['avg_margin'] < MIN_TARGET_MARGIN]
    if not underperformers.empty:
        logger.warning(f"🚨 MARGIN ALERT: The following depts are below {MIN_TARGET_MARGIN}% margin:")
        for dept in underperformers.index:
            logger.warning(f"   - {dept}: {underperformers.loc[dept, 'avg_margin']}%")

    # B. Identifying the "Loss Leader" Trap
    # Find items that lose money but might have high ratings
    loss_leaders = df[df['unit_profit'] < 0].sort_values('unit_profit')
    logger.info(f"\n📉 LOSS LEADERS: {len(loss_leaders)} items are selling below cost!")
    logger.info(loss_leaders[['id', 'dept', 'price', 'unit_cost', 'unit_profit']].head(3).to_string(index=False))

    # C. The "Rating Paradox"
    # Find highly rated items that are significantly cheaper than their category average
    hidden_gems = df[
        (df['rating'] >= 4.5) & 
        (df['price_zscore'] < -1.0)
    ]
    logger.info(f"\n🏆 HIDDEN GEMS: {len(hidden_gems)} top-rated products priced 'Value' for their dept.")
    logger.info(hidden_gems[['id', 'dept', 'rating', 'price', 'dept_avg_price']].head(3).to_string(index=False))

    return df

if __name__ == "__main__":
    enriched_df = run_business_audit()
    enriched_df.to_pickle("inventory_enriched.pkl")
    logger.info("\n✅ Audit Complete. Enriched data saved to 'inventory_enriched.pkl'.")