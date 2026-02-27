"""
Data Analytics Series: Step 5 - Strategic Visualization (The Capstone)

Focus:
Translating complex statistical insights into persuasive visual narratives. 
We demonstrate how to use distributions to justify normalization, how to 
visualize relationships with anomaly highlighting, and how to use Faceting 
to reduce cognitive load in multi-dimensional datasets.

Techniques Demonstrated:
1. Distribution Analysis (KDE): Justifying category-specific Z-scores.
2. Box Plots: Highlighting variance and outliers across departments.
3. Annotated Scatter Plots: Highlighting the "Hidden Gems" from Level 2.
4. Temporal Faceting: Showing growth trends identified in Level 3.

Note:
This script requires 'matplotlib' and 'seaborn'.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# --- TUNABLE PARAMETERS ---
INPUT_FILE = "inventory_enriched.pkl"
PLOT_STYLE = "whitegrid"
PALETTE = "viridis"
FIG_SIZE = (12, 8)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DataViz")

# --- DEPENDENCY CHECK ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    logger.error("❌ CRITICAL ERROR: This lesson requires 'matplotlib' and 'seaborn'.")
    logger.error("Please run: pip install matplotlib seaborn")
    sys.exit(1)


class VisualStoryteller:
    """
    Expert Technique: Informative Visuals.
    We don't just plot data; we plot 'Answers' to business questions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Set Global Seaborn Aesthetics
        sns.set_theme(style=PLOT_STYLE, palette=PALETTE)

    def plot_price_distributions(self):
        """
        Insight: Why did we need Z-scores?
        This plot proves that 'Electronics' and 'Books' have entirely 
        different price scales, making global averages misleading.
        """
        plt.figure(figsize=FIG_SIZE)
        sns.kdeplot(data=self.df, x='price', hue='dept', fill=True, common_norm=False, alpha=0.5)
        
        plt.title("Price Distribution by Department (The Need for Normalization)", fontsize=16)
        plt.xlabel("Price ($)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.tight_layout()
        plt.savefig("viz_1_price_distributions.png")
        logger.info("✅ Saved distribution analysis to 'viz_1_price_distributions.png'")

    def plot_profit_outliers(self):
        """
        Insight: Where is our margin bleeding?
        Uses a Box Plot to show the 'Loss Leader' we injected in Level 2.
        """
        plt.figure(figsize=FIG_SIZE)
        sns.boxplot(data=self.df, x='dept', y='unit_profit', hue='dept', legend=False)
        
        # Add a horizontal line at 0 to highlight losses
        plt.axhline(0, color='red', linestyle='--', alpha=0.6, label="Break Even")
        
        plt.title("Unit Profit Variance by Department", fontsize=16)
        plt.ylabel("Profit per Unit ($)", fontsize=12)
        plt.tight_layout()
        plt.savefig("viz_2_profit_outliers.png")
        logger.info("✅ Saved profit audit to 'viz_2_profit_outliers.png'")

    def plot_hidden_gems_scatter(self):
        """
        Insight: The Rating Paradox.
        A scatter plot showing Price vs. Rating. We highlight the 
        'Hidden Gems' (Low Price, High Rating) identified in Step 3.
        """
        plt.figure(figsize=FIG_SIZE)
        
        # Plot the background data
        sns.scatterplot(data=self.df, x='price', y='rating', hue='dept', alpha=0.4)
        
        # HIGHLIGHT: Find the same Hidden Gems from Level 3 logic
        gems = self.df[(self.df['rating'] >= 4.5) & (self.df['price_zscore'] < -1.0)]
        
        plt.scatter(gems['price'], gems['rating'], color='red', s=100, edgecolors='black', label='Hidden Gems')
        
        # Annotate one of the gems to show 'Pro' attention to detail
        if not gems.empty:
            sample_gem = gems.iloc[0]
            plt.annotate(
                f"Top Value: {sample_gem['id']}",
                xy=(sample_gem['price'], sample_gem['rating']),
                xytext=(sample_gem['price']+50, sample_gem['rating']-0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5)
            )

        plt.title("Price vs. Rating (Highlighting Hidden Gems)", fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.savefig("viz_3_hidden_gems.png")
        logger.info("✅ Saved scatter analysis to 'viz_3_hidden_gems.png'")

    def plot_temporal_trends(self):
        """
        Insight: Growth & Decay.
        Uses Faceting (subplots) to show volume trends per department.
        """
        # Prepare data (Daily count of launches)
        trend_df = (
            self.df.set_index('launch_date')
            .groupby('dept')
            .resample('W')['id'] # Weekly resampling
            .count()
            .reset_index()
            .rename(columns={'id': 'launch_count'})
        )

        g = sns.FacetGrid(trend_df, col="dept", hue="dept", height=4, aspect=1.2)
        g.map(sns.lineplot, "launch_date", "launch_count")
        g.set_xticklabels(rotation=45)
        g.set_titles("{col_name} Launch Velocity")
        
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('Weekly Product Launch Trends (Temporal Faceting)', fontsize=16)
        
        plt.savefig("viz_4_temporal_trends.png")
        logger.info("✅ Saved temporal trends to 'viz_4_temporal_trends.png'")


def run_viz_lesson():
    logger.info("=== STEP 5: STRATEGIC VISUALIZATION ===\n")

    if not Path(INPUT_FILE).exists():
        logger.error(f"❌ Error: {INPUT_FILE} not found. Run previous steps first.")
        return

    # Load the enriched data
    df = pd.read_pickle(INPUT_FILE)
    
    viz = VisualStoryteller(df)
    
    # 1. Distribution (Why normalize?)
    viz.plot_price_distributions()
    
    # 2. Variance (Where are the errors?)
    viz.plot_profit_outliers()
    
    # 3. Relationships (Where is the value?)
    viz.plot_hidden_gems_scatter()
    
    # 4. Faceting (The Pro Move)
    viz.plot_temporal_trends()

    logger.info("\n=== VISUALIZATION AUDIT COMPLETE ===")
    logger.info("Logic Summary:")
    logger.info("1. Distribution analysis justified our department-level Z-scores.")
    logger.info("2. Boxplots exposed the 'Loss Leader' as a visual anomaly.")
    logger.info("3. Scatter Plot annotations highlighted actionable 'Hidden Gems'.")
    logger.info("4. Faceting identified the temporal velocity of each department separately.")

if __name__ == "__main__":
    run_viz_lesson()