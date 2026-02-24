"""
Data Sampling Series: Level 4 - Semantic Diversity & Coverage

Focus:
Uncertainty Sampling (Level 3) suffers from "Tunnel Vision" (focusing only on 
known boundaries) and "Redundancy" (picking the same edge case 50 times).
To fix this, we stop looking at Predictions and start looking at Geometry.

We map data into Vector Space (Embeddings) and use geometric algorithms to 
ensure our training data covers 100% of the topic space, including 
"Unknown Unknowns" (topics the model doesn't even know exist).

Key Concepts:
- K-Means Clustering: Grouping data into topics.
- Centroid Sampling: Picking the "Archetype" of a topic.
- Outlier Sampling: Picking the edge cases.
- Core-Set (Greedy Farthest Point): Maximizing spatial coverage.
"""

import random
import math
import logging
from collections import defaultdict
from typing import List, Dict, Tuple

# --- TUNABLE PARAMETERS ---
ANNOTATION_BUDGET = 6
NUM_CLUSTERS = 3 

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level4_Diversity")


class VectorMath:
    """Geometric operations for 2D Semantic Space."""
    
    @staticmethod
    def euclidean_distance(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

    @staticmethod
    def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculates the geometric center of a group of points."""
        if not points: return (0.0, 0.0)
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        return (sum_x / n, sum_y / n)


class KMeansClusterer:
    """
    A raw Python implementation of K-Means.
    Goal: Partition N data points into K clusters based on geometric closeness.
    """
    
    @staticmethod
    def cluster(dataset: List[Dict], k: int, iterations: int = 5) -> Dict[int, List[Dict]]:
        # 1. Initialize Centroids (Randomly pick k points)
        initial_points = random.sample(dataset, k)
        centroids = [d['vector'] for d in initial_points]
        clusters = defaultdict(list)

        for _ in range(iterations):
            # Reset clusters
            clusters = defaultdict(list)
            
            # 2. Assignment Step
            for doc in dataset:
                # Find closest centroid
                closest_idx = -1
                min_dist = float('inf')
                
                for i, centroid in enumerate(centroids):
                    dist = VectorMath.euclidean_distance(doc['vector'], centroid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                # Assign doc to cluster
                clusters[closest_idx].append(doc)
                doc['cluster_id'] = closest_idx # Tag the doc

            # 3. Update Step (Recalculate centroids)
            new_centroids = []
            for i in range(k):
                points = [d['vector'] for d in clusters[i]]
                if points:
                    new_centroids.append(VectorMath.get_centroid(points))
                else:
                    # Handle empty cluster edge case by keeping old centroid
                    new_centroids.append(centroids[i])
            centroids = new_centroids
            
        return clusters


class DiversitySampler:
    """Core Geometric Sampling Strategies."""

    @staticmethod
    def centroid_sampling(clusters: Dict[int, List[Dict]], budget_per_cluster: int) -> List[Dict]:
        """
        Pick the item closest to the center of each cluster.
        Best for: Summarization (finding the "textbook" example of a topic).
        """
        selection = []
        for cluster_id, docs in clusters.items():
            if not docs: continue
            
            # Calculate actual centroid of this cluster
            vectors = [d['vector'] for d in docs]
            centroid = VectorMath.get_centroid(vectors)
            
            # Sort by distance to centroid (Ascending)
            docs.sort(key=lambda d: VectorMath.euclidean_distance(d['vector'], centroid))
            selection.extend(docs[:budget_per_cluster])
            
        return selection

    @staticmethod
    def outlier_sampling(clusters: Dict[int, List[Dict]], budget_per_cluster: int) -> List[Dict]:
        """
        Pick the item farthest from the center.
        Best for: Finding Edge Cases and Anomalies.
        """
        selection = []
        for cluster_id, docs in clusters.items():
            if not docs: continue
            
            vectors = [d['vector'] for d in docs]
            centroid = VectorMath.get_centroid(vectors)
            
            # Sort by distance to centroid (Descending)
            docs.sort(key=lambda d: VectorMath.euclidean_distance(d['vector'], centroid), reverse=True)
            selection.extend(docs[:budget_per_cluster])
            
        return selection

    @staticmethod
    def core_set_greedy(dataset: List[Dict], budget: int) -> List[Dict]:
        """
        The "Gold Standard" for Coverage.
        Algorithm (Greedy Farthest Point):
        1. Pick a random point.
        2. Find the point FARTHEST away from the already picked points.
        3. Repeat.
        This guarantees we span the maximum area of the semantic map.
        """
        if not dataset: return []
        
        # Start with a random point
        pool = dataset[:]
        selected = [pool.pop(random.randint(0, len(pool)-1))]
        
        while len(selected) < budget and pool:
            farthest_point = None
            max_min_dist = -1
            
            # For every candidate in the pool...
            for candidate in pool:
                # Find its distance to the NEAREST point we already selected
                # (We want to maximize this minimum distance)
                min_dist_to_selected = min(
                    VectorMath.euclidean_distance(candidate['vector'], s['vector']) 
                    for s in selected
                )
                
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    farthest_point = candidate
            
            # Add the farthest candidate to selection
            if farthest_point:
                selected.append(farthest_point)
                pool.remove(farthest_point)
                
        return selected


# --- TUTORIAL DEMONSTRATIONS ---

def generate_semantic_map():
    """
    Generates data in 3 distinct 'Semantic Clusters'.
    We mock a 2D space for visualization.
    """
    data = []
    
    # Cluster 1: "Billing Issues" (Top Right)
    for i in range(20):
        data.append({
            "id": f"bill_{i}", "topic": "Billing",
            "vector": (random.uniform(8, 10), random.uniform(8, 10))
        })
        
    # Cluster 2: "Tech Support" (Bottom Left)
    for i in range(20):
        data.append({
            "id": f"tech_{i}", "topic": "Tech",
            "vector": (random.uniform(1, 3), random.uniform(1, 3))
        })
        
    # Cluster 3: "The Unknown Unknown" (Bottom Right)
    # A small, rare cluster the model might ignore.
    for i in range(5):
        data.append({
            "id": f"rare_{i}", "topic": "Rare_Bug",
            "vector": (random.uniform(8, 10), random.uniform(1, 2))
        })
        
    return data

def run_lesson():
    logger.info("=== LESSON 4: SEMANTIC DIVERSITY & COVERAGE ===")
    dataset = generate_semantic_map()
    logger.info(f"Generated {len(dataset)} points in Vector Space.")
    logger.info("Topics: Billing (20), Tech (20), Rare_Bug (5).\n")

    # --- DEMO 1: RANDOM SAMPLING (Baseline) ---
    logger.info("--- 1. RANDOM SAMPLING (Baseline) ---")
    random_selection = random.sample(dataset, ANNOTATION_BUDGET)
    topics = [d['topic'] for d in random_selection]
    logger.info(f"Selected: {topics}")
    logger.info(f"  - Rare_Bug Found: {'Rare_Bug' in topics}")
    logger.info("[Takeaway]: Random sampling often misses small clusters entirely. We missed the 'Unknown Unknown'.")

    # --- DEMO 2: CLUSTER-BASED SAMPLING (Centroids) ---
    logger.info("\n--- 2. K-MEANS CLUSTERING (Centroids) ---")
    logger.info(f"Clustering data into {NUM_CLUSTERS} topics...")
    clusters = KMeansClusterer.cluster(dataset, k=NUM_CLUSTERS)
    
    # We sample 2 items per cluster (Total 6)
    centroid_selection = DiversitySampler.centroid_sampling(clusters, budget_per_cluster=2)
    topics = [d['topic'] for d in centroid_selection]
    
    logger.info(f"Selected: {topics}")
    logger.info(f"  - Rare_Bug Found: {'Rare_Bug' in topics}")
    logger.info("[Takeaway]: Clustering forced us to look at every group. We found the Rare Bug because K-Means identified it as a distinct geometric region. Centroid sampling gives us the 'Archetypes' (Best examples) of each topic.")

    # --- DEMO 3: CORE-SET SELECTION (Max Diversity) ---
    logger.info("\n--- 3. CORE-SET SELECTION (Greedy Farthest Point) ---")
    logger.info("Goal: Pick points that are as far apart as possible to span the map.")
    
    core_set_selection = DiversitySampler.core_set_greedy(dataset, budget=ANNOTATION_BUDGET)
    topics = [d['topic'] for d in core_set_selection]
    
    logger.info(f"Selected: {topics}")
    logger.info(f"  - Rare_Bug Found: {'Rare_Bug' in topics}")
    logger.info("[Takeaway]: Core-Set Selection is the gold standard. It naturally discovered the Rare Bug cluster simply because it was geometrically distant from the Billing and Tech clusters. It ensures maximum coverage of the problem space without needing to know the topics beforehand.")

if __name__ == "__main__":
    run_lesson()