"""
Data Sampling Series: Level 3 - Uncertainty Sampling (Active Learning)

Focus:
We have a V1 model. It works okay, but we have budget to label 10 more items.
Instead of random sampling, we use the model's own confusion to find the 
"Decision Boundary."

We compare three strategies to demonstrate the "Noise Trap":
1. Margin Sampling: Finding the edge case between two classes.
2. Entropy Sampling: Measuring total confusion across all classes.
3. Query By Committee (QBC): Using model disagreement to distinguish 
   between "Hard Data" (valuable) and "Noise" (garbage).

Key Takeaway:
Simple uncertainty (Margin/Entropy) often selects garbage data because models 
are "uncertain" about noise. QBC fixes this by checking if models *agree* 
that they are confused.
"""

import random
import math
import logging
import statistics
from typing import List, Dict

# --- TUNABLE PARAMETERS ---
ANNOTATION_BUDGET = 10
NUM_COMMITTEE_MEMBERS = 3 

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level3_ActiveLearning")


class MathUtils:
    """Helper for Information Theory calculations."""
    
    @staticmethod
    def calculate_entropy(probabilities: List[float]) -> float:
        """
        Entropy = -Sum(p * log(p)).
        High Entropy = High Confusion (Flat distribution).
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p, 2)
        return entropy

    @staticmethod
    def calculate_margin(probabilities: List[float]) -> float:
        """
        Margin = Prob(Best Class) - Prob(Second Best Class).
        Small Margin = High Confusion (The model is torn).
        """
        sorted_probs = sorted(probabilities, reverse=True)
        return sorted_probs[0] - sorted_probs[1]

    @staticmethod
    def calculate_disagreement(predictions: List[List[float]]) -> float:
        """
        Calculates how much the committee members disagree.
        We use the Variance of the predicted probability for Class A.
        """
        # Extract the probability of Class 0 from each model
        class_0_probs = [p[0] for p in predictions]
        
        # Calculate Variance (High Variance = High Disagreement)
        if len(class_0_probs) < 2: return 0.0
        return statistics.variance(class_0_probs)


class MockModel:
    """
    Simulates a classifier. 
    Crucially, we simulate the difference between "I don't know" (Noise)
    and "I disagree" (Hard).
    """
    
    @staticmethod
    def predict_single(doc_type: str) -> List[float]:
        """Returns a probability distribution over 3 classes [A, B, C]."""
        if doc_type == "EASY":
            return [0.90, 0.05, 0.05] # Confident
        elif doc_type == "HARD":
            # Confused between A and B (The Boundary)
            # Margin: 0.02 (Very small!)
            return [0.49, 0.51, 0.00] 
        elif doc_type == "NOISE":
            # Complete uniform randomness (Garbage input)
            # Margin: 0.01 (Even smaller! This tricks Margin Sampling)
            return [0.33, 0.33, 0.34] 
        return [0.0, 0.0, 0.0]

    @staticmethod
    def predict_committee(doc_type: str, num_members: int) -> List[List[float]]:
        """
        Simulates N models predicting on the same data.
        
        The Secret Sauce:
        - NOISE: Models agree that it is garbage. (Low Variance).
        - HARD:  Models disagree on the answer. (High Variance).
        """
        preds = []
        for _ in range(num_members):
            if doc_type == "EASY":
                # Everyone agrees it's easy
                preds.append([0.9, 0.05, 0.05])
                
            elif doc_type == "NOISE":
                # Everyone agrees it's garbage (Uniform distribution)
                # Jitter is small because they all see the same garbage features.
                p = random.uniform(0.30, 0.36)
                remaining = 1.0 - p
                preds.append([p, remaining/2, remaining/2])
                
            elif doc_type == "HARD":
                # DISAGREEMENT! 
                # Model 1 might learn Feature X and say Class A.
                # Model 2 might learn Feature Y and say Class B.
                lean = random.choice(["A", "B"])
                if lean == "A":
                    preds.append([0.70, 0.30, 0.0])
                else:
                    preds.append([0.30, 0.70, 0.0])
                    
        return preds


class ActiveLearner:
    """Core Sampling Strategies."""

    @staticmethod
    def margin_sampling(dataset: List[Dict]) -> List[Dict]:
        """
        Strategy 1: Margin Sampling.
        Trap: Picks Noise because [0.33, 0.33, 0.34] has a smaller margin (0.01)
        than valid Hard data [0.49, 0.51] (0.02).
        """
        for doc in dataset:
            doc["margin_score"] = MathUtils.calculate_margin(doc["probs"])
        
        # Sort Ascending (Smallest Margin = Most Uncertain)
        return sorted(dataset, key=lambda x: x["margin_score"])

    @staticmethod
    def entropy_sampling(dataset: List[Dict]) -> List[Dict]:
        """
        Strategy 2: Entropy Sampling.
        Trap: Picks Noise because [0.33, 0.33, 0.33] is mathematically 
        'Maximum Entropy'.
        """
        for doc in dataset:
            doc["entropy_score"] = MathUtils.calculate_entropy(doc["probs"])
        
        # Sort Descending (Highest Entropy = Most Uncertain)
        return sorted(dataset, key=lambda x: x["entropy_score"], reverse=True)

    @staticmethod
    def query_by_committee(dataset: List[Dict]) -> List[Dict]:
        """
        Strategy 3: Query By Committee (QBC).
        Solution: Ignores Noise because models 'Agree to be confused'.
        Prioritizes Hard data because models conflict on the answer.
        """
        for doc in dataset:
            committee_preds = doc["committee_probs"] 
            doc["qbc_score"] = MathUtils.calculate_disagreement(committee_preds)
            
        # Sort Descending (Highest Variance = Most Disagreement)
        return sorted(dataset, key=lambda x: x["qbc_score"], reverse=True)


# --- TUTORIAL DEMONSTRATIONS ---

def generate_dataset():
    """Generates a mix of Easy, Hard, and Noise data."""
    data = []
    # 80 Easy docs
    for i in range(80):
        data.append({
            "type": "EASY", 
            "probs": MockModel.predict_single("EASY"), 
            "committee_probs": MockModel.predict_committee("EASY", NUM_COMMITTEE_MEMBERS)
        })
    # 15 Hard docs
    for i in range(15):
        data.append({
            "type": "HARD", 
            "probs": MockModel.predict_single("HARD"), 
            "committee_probs": MockModel.predict_committee("HARD", NUM_COMMITTEE_MEMBERS)
        })
    # 5 Noise docs (Garbage)
    for i in range(5):
        data.append({
            "type": "NOISE", 
            "probs": MockModel.predict_single("NOISE"), 
            "committee_probs": MockModel.predict_committee("NOISE", NUM_COMMITTEE_MEMBERS)
        })
    
    random.shuffle(data)
    return data

def run_lesson():
    logger.info("=== LESSON 3: UNCERTAINTY vs DISAGREEMENT ===")
    dataset = generate_dataset()
    logger.info(f"Generated 100 docs: 80 Easy, 15 Hard (Valid), 5 Noise (Garbage).")
    logger.info(f"Budget: Annotate {ANNOTATION_BUDGET} items.\n")

    # --- DEMO 1: MARGIN SAMPLING ---
    logger.info("--- 1. MARGIN SAMPLING (Distance between Top-1 and Top-2) ---")
    logger.info("Goal: Find the finest decision boundary.")
    
    margin_selected = ActiveLearner.margin_sampling(dataset)[:ANNOTATION_BUDGET]
    types_marg = [d["type"] for d in margin_selected]
    
    logger.info(f"Selected: {types_marg}")
    logger.info(f"  - NOISE (Trap): {types_marg.count('NOISE')} / 5")
    logger.info(f"  - HARD (Valid): {types_marg.count('HARD')}")
    logger.info("[Takeaway]: Margin Sampling failed. The Noise [0.33, 0.33, 0.34] has a smaller margin (0.01) than the Hard data (0.02). We prioritized garbage.")


    # --- DEMO 2: ENTROPY SAMPLING (Information Theory) ---
    logger.info("\n--- 2. ENTROPY SAMPLING (Total Confusion) ---")
    logger.info("Goal: Find the flattest probability distribution.")
    
    entropy_selected = ActiveLearner.entropy_sampling(dataset)[:ANNOTATION_BUDGET]
    types_ent = [d["type"] for d in entropy_selected]
    
    logger.info(f"Selected: {types_ent}")
    logger.info(f"  - NOISE (Trap): {types_ent.count('NOISE')} / 5")
    logger.info(f"  - HARD (Valid): {types_ent.count('HARD')}")
    logger.info("[Takeaway]: Entropy failed. Pure garbage produces the highest possible entropy. This is 'Aleatoric Uncertainty' (The data itself is confusing), and we wasted budget on it.")


    # --- DEMO 3: QUERY BY COMMITTEE (Multi-Model Disagreement) ---
    logger.info("\n--- 3. QUERY BY COMMITTEE (Epistemic Uncertainty) ---")
    logger.info("Goal: Pick items where models DISAGREE with each other.")
    
    qbc_selected = ActiveLearner.query_by_committee(dataset)[:ANNOTATION_BUDGET]
    types_qbc = [d["type"] for d in qbc_selected]
    
    logger.info(f"Selected: {types_qbc}")
    logger.info(f"  - NOISE (Trap): {types_qbc.count('NOISE')} / 5")
    logger.info(f"  - HARD (Valid): {types_qbc.count('HARD')}")
    logger.info("[Takeaway]: QBC worked! Even though the Noise was confusing, all models *agreed* it was confusing (Low Disagreement). But for Hard data, one model said A and another said B (High Disagreement).")
    
    logger.info("\n[Critical Analysis]:")
    logger.info("Senior Engineers prefer QBC or Hybrid Methods (Level 5) because they filter out Aleatoric Uncertainty (Noise) and focus on Epistemic Uncertainty (Things the model CAN learn but hasn't yet).")

if __name__ == "__main__":
    run_lesson()