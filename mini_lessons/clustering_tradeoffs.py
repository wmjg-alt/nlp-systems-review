"""
Mini-Lesson: Clustering Algorithms & Geometric Trade-offs

Context:
In Data Sampling Level 4, we discussed using Clustering to find diverse topics.
However, choosing the wrong clustering algorithm can ruin your data curation.

The Conflict:
1. K-Means: Fast, but assumes clusters are spherical blobs of similar density. 
   It forces every point into a cluster (even noise).
2. DBSCAN: Density-Based. Finds arbitrary shapes (like crescents or rings) 
   and explicitly identifies "Noise" (outliers), but is harder to tune.

The Demonstration:
We generate a "Donut" dataset (a dense inner circle surrounded by an outer ring).
We will see K-Means fail to separate the ring from the center, while DBSCAN 
succeeds.
"""

import sys
import math
import random
import logging
from collections import Counter
from typing import List, Dict, Tuple

# --- TUNABLE PARAMETERS ---
NUM_POINTS_INNER = 200    # The "Hole" of the donut
NUM_POINTS_OUTER = 400    # The "Ring" of the donut
NUM_NOISE_POINTS = 50     # Random scattered outliers

# Algorithm Tunables
K_MEANS_K = 2             # We know there are 2 main clusters
DBSCAN_EPS = 1.5          # Max distance between points to be "neighbors"
DBSCAN_MIN_SAMPLES = 5    # Min points to form a dense region

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ClusteringLesson")

# --- DEPENDENCY CHECK ---
try:
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN
except ImportError:
    logger.error("❌ CRITICAL ERROR: This lesson requires 'scikit-learn' and 'numpy'.")
    logger.error("Please run: pip install scikit-learn numpy")
    sys.exit(1)


class GeometricGenerator:
    """Generates tricky geometric shapes that break simple algorithms."""
    
    @staticmethod
    def generate_donut() -> Tuple[np.ndarray, List[str]]:
        """
        Creates two clusters: 
        1. A dense inner circle ("City Center")
        2. A surrounding ring ("Suburbs")
        3. Random noise ("Rural")
        """
        points = []
        labels = []

        # 1. Inner Circle (The City)
        for _ in range(NUM_POINTS_INNER):
            # Random angle, small radius (0 to 5)
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, 5)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append([x, y])
            labels.append("City_Center")

        # 2. Outer Ring (The Suburbs)
        for _ in range(NUM_POINTS_OUTER):
            # Random angle, large radius (10 to 15) - gap between 5 and 10
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(10, 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append([x, y])
            labels.append("Suburbs")

        # 3. Noise (Rural Outliers)
        for _ in range(NUM_NOISE_POINTS):
            # Random points anywhere in the 20x20 space
            x = random.uniform(-20, 20)
            y = random.uniform(-20, 20)
            points.append([x, y])
            labels.append("Noise")

        return np.array(points), labels


class ClusterEvaluator:
    """Analyzes how well the algorithm separated the ground truth labels."""
    
    @staticmethod
    def evaluate(algo_name: str, predicted_labels: List[int], ground_truth: List[str]):
        logger.info(f"\n--- EVALUATING: {algo_name} ---")
        
        # Group indices by predicted cluster
        clusters = {}
        for i, label in enumerate(predicted_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(ground_truth[i])
            
        # Print stats for each cluster
        # Note: DBSCAN uses -1 for Noise
        sorted_keys = sorted(clusters.keys())
        
        for k in sorted_keys:
            cluster_name = f"Cluster {k}" if k != -1 else "NOISE (-1)"
            counts = Counter(clusters[k])
            total = len(clusters[k])
            
            # Formatting the breakdown
            breakdown = ", ".join([f"{count} {name}" for name, count in counts.items()])
            logger.info(f"   {cluster_name:<12} ({total:>3} points): {breakdown}")

        # Pedagogical Check
        if algo_name == "K-Means":
            # K-Means usually splits the ring in half or merges distinct groups
            logger.info(f"[Analysis]: Notice how K-Means forces 'Noise' into clusters. It also fails to separate the Ring from the Center because it relies on simple distance from a center point.")
        elif algo_name == "DBSCAN":
            logger.info(f"[Analysis]: DBSCAN successfully identified the 'City' vs 'Suburbs' by following the density chain. Crucially, it isolated the 'Noise' as -1.")


def run_lesson():
    logger.info("=== MINI-LESSON: CLUSTERING ALGORITHMS & TRADEOFFS ===")
    logger.info(f"Generating 'Donut' dataset with {NUM_POINTS_INNER + NUM_POINTS_OUTER + NUM_NOISE_POINTS} points...")
    logger.info("Ground Truth: Inner Circle (City), Outer Ring (Suburbs), and Random Noise.\n")
    
    X, truth_labels = GeometricGenerator.generate_donut()

    # --- 1. K-MEANS ---
    # K-Means tries to find K centroids and draws straight lines between them (Voronoi).
    # It cannot handle rings/crescents.
    kmeans = KMeans(n_clusters=K_MEANS_K, random_state=42, n_init=10)
    kmeans_preds = kmeans.fit_predict(X)
    ClusterEvaluator.evaluate("K-Means", kmeans_preds, truth_labels)

    # --- 2. DBSCAN ---
    # Density-Based Spatial Clustering of Applications with Noise.
    # It groups points that are packed closely together.
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    dbscan_preds = dbscan.fit_predict(X)
    ClusterEvaluator.evaluate("DBSCAN", dbscan_preds, truth_labels)

if __name__ == "__main__":
    run_lesson()