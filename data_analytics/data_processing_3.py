"""
Data Analytics Series: Step 3 - Advanced Transforms & Executive Reporting

Focus:
Converting cleaned, enriched data into high-level strategic insights. 
We explore how time-series analysis and multi-dimensional reshaping (pivoting) 
uncover trends that aggregate department averages hide.

Techniques Demonstrated:
1. Time-Series Resampling: Aggregating performance by Month/Quarter.
2. Pivot Tables: Cross-referencing Department vs. Time for heat-map analysis.
3. Method Chaining: Writing an entire reporting pipeline in a single fluid block.
4. Logic Traps: Identifying 'Growth Decay' (Monthly average ratings dropping).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# --- TUNABLE PARAMETERS ---
INPUT_FILE = "inventory_enriched.pkl"
# Smoothing factor for time trends (Monthly, Quarterly, etc.)
RESAMPLE_PERIOD = 'ME'  # 'ME' = Month End
# Threshold for flagging 'Rating Decay'
DECAY_THRESHOLD = -0.1  # Flag if current month is >0.1 lower than avg

# Configure logging for professional feedback
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AppliedScientist")

def generate_executive_report(df: pd.DataFrame):
    """
    Expert Technique: Method Chaining.
    Building a clean, immutable-style pipeline for a final report.
    """
    logger.info("[Step 1]: Generating Temporal Growth Matrix via Resampling...")
    
    # We want to see how Depts are launching products over time.
    # We set the index to the date to enable time-series math.
    growth_report = (
        df.set_index('launch_date')
        .groupby('dept')
        .resample(RESAMPLE_PERIOD)['price']
        .count()
        .unstack(level=0)
        .fillna(0)
    )
    
    logger.info("\n--- MONTHLY PRODUCT LAUNCH VOLUME ---")
    logger.info(growth_report.tail(6).to_string())

    logger.info("\n[Step 2]: Creating Pivot Table (Dept vs. Margin/Rating)...")
    # Pivot tables are essential for 'Heat Map' style analysis
    pivot_matrix = df.pivot_table(
        values=['margin_pct', 'rating'],
        index='dept',
        aggfunc={'margin_pct': 'mean', 'rating': 'mean'}
    ).round(2)
    
    logger.info("\n--- CATEGORY PERFORMANCE PIVOT ---")
    logger.info(pivot_matrix.to_string())

    return growth_report, pivot_matrix

def detect_rating_decay(df: pd.DataFrame):
    """
    Identifies if a department's quality is dropping over time.
    Uses a rolling window comparison.
    """
    logger.info("\n[Step 3]: Auditing Temporal Rating Decay...")
    
    # 1. Monthly Average Rating per Dept
    monthly_ratings = (
        df.set_index('launch_date')
        .groupby('dept')
        .resample(RESAMPLE_PERIOD)['rating']
        .mean()
        .ffill() # Handle months with zero launches
    )
    
    # 2. Compare latest month to previous average (Decay)
    for dept in df['dept'].unique():
        series = monthly_ratings.loc[dept]
        if len(series) < 2: continue
        
        current = series.iloc[-1]
        historic_avg = series.iloc[:-1].mean()
        delta = current - historic_avg
        
        if delta < DECAY_THRESHOLD:
            logger.warning(f" -- QUALITY ALERT: {dept} rating is decaying! "
                           f"(Current: {current:.2f} vs Historic: {historic_avg:.2f})")
        else:
            logger.info(f" STABLE: {dept} maintaining quality (Delta: {delta:+.2f})")

def run_final_analytics():
    logger.info("=== STEP 4: EXECUTIVE REPORTING & TIME-SERIES ===\n")

    if not Path(INPUT_FILE).exists():
        logger.error(f"❌ Error: {INPUT_FILE} not found. Run data_processing_2.py first.")
        return

    df = pd.read_pickle(INPUT_FILE)
    
    # 1. Generate core report
    growth_matrix, performance_pivot = generate_executive_report(df)
    
    # 2. Run advanced diagnostic
    detect_rating_decay(df)

    logger.info("\n" + "="*50)
    logger.info("STRATEGIC SUMMARY")
    logger.info("="*50)
    best_dept = performance_pivot['margin_pct'].idxmax()
    logger.info(f" Most Profitable: {best_dept} ({performance_pivot.loc[best_dept, 'margin_pct']}% margin)")
    
    highest_vol = growth_matrix.sum().idxmax()
    logger.info(f" Most Active Launcher: {highest_vol} ({int(growth_matrix.sum().max())} units)")
    logger.info("="*50)

if __name__ == "__main__":
    run_final_analytics()