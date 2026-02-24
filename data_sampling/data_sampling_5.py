"""
Data Sampling Series: Level 5 - Hybrid Active Learning (Density-Weighted)

Focus:
The "Holy Grail" of data curation. We combine Uncertainty (Level 3) with 
Diversity/Density (Level 4) to solve the "Noise Trap."

The Problem: 
Uncertainty Sampling picks Garbage (because the model is confused by it).
Diversity Sampling picks Easy Data (because it wants to cover the whole map).

The Solution: 
We pick data that is BOTH Uncertain (Confusing) AND Dense (Representative).
Score = Uncertainty * Density.

Key Concepts:
- Euclidean Density Estimation: Measuring how "isolated" a point is.
- The Hybrid Score: Balancing the need for answers vs. the need for valid data.
- Outlier Rejection: Automatically culling garbage without human review.
"""

import random
import math
import logging
import statistics
from typing import List, Dict, Tuple

# --- TUNABLE PARAMETERS ---
ANNOTATION_BUDGET = 5
K_NEIGHBORS = 5           # Look at 5 nearest neighbors to estimate density
ALPHA = 1.0               # Weight for Uncertainty (Higher = prefer harder data)
BETA = 1.0                # Weight for Density (Higher = prefer common data/reject noise)

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level5_Hybrid")


class VectorMath:
    """Geometric operations for 2D Semantic Space."""
    @staticmethod
    def euclidean_distance(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)


class DensityEstimator:
    """
    Calculates how 'Representative' a point is.
    High Density = Surrounded by other data (Valid).
    Low Density = Isolated (Outlier/Noise).
    """
    
    @staticmethod
    def calculate_density_scores(dataset: List[Dict], k: int) -> List[Dict]:
        """
        Assigns a 'density_score' (0.0 to 1.0) to every doc.
        Metric: 1 / (Average distance to k nearest neighbors).
        """
        for i, doc in enumerate(dataset):
            distances = []
            # Calculate distance to all other points (O(N^2) - slow, but fine for demo)
            for j, other in enumerate(dataset):
                if i == j: continue
                dist = VectorMath.euclidean_distance(doc['vector'], other['vector'])
                distances.append(dist)
            
            # Find k nearest neighbors
            distances.sort()
            k_nearest = distances[:k]
            avg_dist = statistics.mean(k_nearest) if k_nearest else float('inf')
            
            # Invert distance: Small distance = High Density
            # We normalize crudely for the demo (assuming max dist is around 20)
            # In production, you would use proper normalization/softmax
            density = 1.0 / (avg_dist + 0.001) # Avoid div/0
            doc['density_raw'] = density

        # Normalize scores 0.0 to 1.0 for easier weighting
        max_dens = max(d['density_raw'] for d in dataset)
        min_dens = min(d['density_raw'] for d in dataset)
        
        for doc in dataset:
            norm = (doc['density_raw'] - min_dens) / (max_dens - min_dens)
            doc['density_score'] = norm
            
        return dataset


class MockModel:
    """
    Simulates a classifier.
    Notice: It assigns High Uncertainty (0.5) to both HARD data and NOISE.
    """
    @staticmethod
    def predict_prob(doc_type: str) -> float:
        if doc_type == "EASY":
            return 0.95 # Very confident
        elif doc_type == "HARD":
            return 0.51 # Very confused (The Boundary)
        elif doc_type == "NOISE":
            return 0.50 # Very confused (Garbage)
        return 0.0


class HybridSampler:
    """Core Sampling Strategies."""

    @staticmethod
    def uncertainty_sampling(dataset: List[Dict], budget: int) -> List[Dict]:
        """Level 3 Baseline: Pick absolute closest to 0.5."""
        # Calculate Uncertainty (1 - |prob - 0.5| * 2) -> 1.0 is max uncertainty
        for doc in dataset:
            prob = doc['prob']
            doc['uncertainty_score'] = 1.0 - (abs(prob - 0.5) * 2)
            
        # Sort by Uncertainty Descending
        return sorted(dataset, key=lambda x: x['uncertainty_score'], reverse=True)[:budget]

    @staticmethod
    def hybrid_sampling(dataset: List[Dict], budget: int) -> List[Dict]:
        """
        Level 5: Density-Weighted Uncertainty.
        Score = (Uncertainty^Alpha) * (Density^Beta).
        """
        # First, ensure we have Uncertainty and Density scores calculated
        # (Assuming dataset comes pre-processed by DensityEstimator)
        
        for doc in dataset:
            unc = doc['uncertainty_score']
            den = doc['density_score']
            
            # The Magic Formula
            hybrid_score = (unc ** ALPHA) * (den ** BETA)
            doc['hybrid_score'] = hybrid_score
            
        return sorted(dataset, key=lambda x: x['hybrid_score'], reverse=True)[:budget]


# --- TUTORIAL DEMONSTRATIONS ---

def generate_semantic_map():
    """Generates 3 types of data in Vector Space."""
    data = []
    
    # 1. EASY CLUSTER (Tight group, Model is Confident)
    # Location: (2, 2)
    for i in range(20):
        data.append({
            "id": f"easy_{i}", "type": "EASY", 
            "vector": (random.uniform(1.5, 2.5), random.uniform(1.5, 2.5)),
            "prob": MockModel.predict_prob("EASY")
        })

    # 2. HARD CLUSTER (Tight group, Model is Confused - VALID DATA)
    # Location: (8, 8)
    for i in range(10):
        data.append({
            "id": f"hard_{i}", "type": "HARD", 
            "vector": (random.uniform(7.5, 8.5), random.uniform(7.5, 8.5)),
            "prob": MockModel.predict_prob("HARD")
        })

    # 3. NOISE (Scattered Randomly, Model is Confused - GARBAGE)
    # Location: Random outliers
    for i in range(5):
        data.append({
            "id": f"noise_{i}", "type": "NOISE", 
            "vector": (random.uniform(0, 10), random.uniform(0, 10)),
            "prob": MockModel.predict_prob("NOISE")
        })
        
    return data

def run_lesson():
    logger.info("=== LESSON 5: HYBRID ACTIVE LEARNING ===")
    
    # 1. Setup
    dataset = generate_semantic_map()
    logger.info(f"Generated {len(dataset)} points.")
    logger.info("  - 20 Easy (Cluster at 2,2)")
    logger.info("  - 10 Hard (Cluster at 8,8)")
    logger.info("  - 5 Noise (Scattered Randomly)")
    
    # 2. Calculate Density (The Level 5 Magic)
    logger.info("\nCalculating Density Estimation (k-NN)...")
    dataset = DensityEstimator.calculate_density_scores(dataset, k=K_NEIGHBORS)
    
    # 3. Run Strategies
    logger.info("\n--- STRATEGY A: PURE UNCERTAINTY (Level 3) ---")
    unc_selection = HybridSampler.uncertainty_sampling(dataset, ANNOTATION_BUDGET)
    
    types = [d['type'] for d in unc_selection]
    logger.info(f"Selected: {types}")
    logger.info(f"  - NOISE (Trap): {types.count('NOISE')}")
    logger.info(f"  - HARD (Valid): {types.count('HARD')}")
    logger.info("[Analysis]: Pure uncertainty failed. It picked NOISE because the model was 0.50 (Max Uncertainty), whereas valid HARD data was 0.51.")

    logger.info("\n--- STRATEGY B: DENSITY-WEIGHTED UNCERTAINTY (Level 5) ---")
    hyb_selection = HybridSampler.hybrid_sampling(dataset, ANNOTATION_BUDGET)
    
    types = [d['type'] for d in hyb_selection]
    logger.info(f"Selected: {types}")
    logger.info(f"  - NOISE (Trap): {types.count('NOISE')}")
    logger.info(f"  - HARD (Valid): {types.count('HARD')}")
    
    logger.info("\n[Detailed Logic Check]:")
    # Show specific scores for a Noise item vs a Hard item
    noise_item = next(d for d in dataset if d['type'] == "NOISE")
    hard_item = next(d for d in dataset if d['type'] == "HARD")
    
    logger.info(f"Sample Noise Item:")
    logger.info(f"  Uncertainty: {noise_item['uncertainty_score']:.2f} (MAX)")
    logger.info(f"  Density:     {noise_item['density_score']:.2f} (LOW - Isolated)")
    logger.info(f"  Hybrid Score: {noise_item['hybrid_score']:.4f}")
    
    logger.info(f"Sample Hard Item:")
    logger.info(f"  Uncertainty: {hard_item['uncertainty_score']:.2f} (High)")
    logger.info(f"  Density:     {hard_item['density_score']:.2f} (HIGH - Clustered)")
    logger.info(f"  Hybrid Score: {hard_item['hybrid_score']:.4f}")

    logger.info("\n[Takeaway]: The Hybrid score correctly penalized the Noise for being isolated in vector space, allowing the 'Dense' Hard data to win the budget.")

if __name__ == "__main__":
    run_lesson()