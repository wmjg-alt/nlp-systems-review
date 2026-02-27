"""
Metrics Series: Lesson 2 - ROC-AUC vs. PR-AUC (The Confidence Game)

The Challenge:
Your model outputs a probability (0.85, 0.12, etc.). To actually *do* something, 
you must pick a threshold. If Threshold = 0.5, anything above is 'True'. 
But how do you evaluate the model's quality across ALL possible thresholds?

The Trap:
ROC-AUC (Receiver Operating Characteristic) is the standard 'go-to'. 
However, ROC-AUC is dangerously optimistic on imbalanced data. Because it 
calculates the 'False Positive Rate' (FP / Total Negatives), if you have 
millions of negatives, the denominator is so huge that the model looks 
perfect even if it is failing.

The Solution:
PR-AUC (Precision-Recall AUC). By ignoring True Negatives and focusing 
strictly on Precision and Recall, it provides the "honest" mathematical 
truth for rare events.
"""

import random
import logging
import math
from typing import List, Tuple, Dict

# --- TUNABLE PARAMETERS ---
TOTAL_SAMPLES = 10_000
IMBALANCE_RATIO = 0.01  # 1% are positive cases
NUM_THRESHOLDS = 100    # Granularity of the curve calculation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Metrics_Lesson_2")


class CurveCalculator:
    """
    Manually calculates points on a curve by sliding a threshold from 1.0 to 0.0.
    """

    @staticmethod
    def get_curve_points(y_true: List[int], y_probs: List[float]) -> List[Dict[str, float]]:
        points = []
        # Slide threshold from 1.0 down to 0.0
        for i in range(NUM_THRESHOLDS + 1):
            threshold = i / NUM_THRESHOLDS
            
            tp, fp, tn, fn = 0, 0, 0, 0
            for truth, prob in zip(y_true, y_probs):
                pred = 1 if prob >= threshold else 0
                if truth == 1 and pred == 1: tp += 1
                elif truth == 0 and pred == 1: fp += 1
                elif truth == 0 and pred == 0: tn += 1
                elif truth == 1 and pred == 0: fn += 1
            
            # Metrics for ROC: TPR (Recall) vs FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # Metrics for PR: Precision vs Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tpr # Recall is the same as TPR
            
            points.append({
                "threshold": threshold,
                "tpr": tpr,
                "fpr": fpr,
                "precision": precision,
                "recall": recall
            })
        return points

    @staticmethod
    def calculate_auc(x_coords: List[float], y_coords: List[float]) -> float:
        """
        Calculates the Area Under Curve (AUC) using the Trapezoidal Rule.
        Formula: Sum of ((y_next + y_curr) / 2) * (x_next - x_curr)
        """
        auc = 0.0
        # Sort by x-axis to ensure we integrate left-to-right
        pairs = sorted(zip(x_coords, y_coords))
        
        for i in range(len(pairs) - 1):
            x1, y1 = pairs[i]
            x2, y2 = pairs[i+1]
            dx = x2 - x1
            avg_y = (y1 + y2) / 2
            auc += avg_y * dx
        return auc


class MockModel:
    """Simulates a model that is 'okayish' at its job."""
    
    @staticmethod
    def get_predictions(y_true: List[int], noise_level: float) -> List[float]:
        """
        Generates probability scores. 
        If truth is 1, score centers around 0.7.
        If truth is 0, score centers around 0.3.
        noise_level: how much overlap there is between classes.
        """
        probs = []
        for val in y_true:
            if val == 1:
                # Target: High score (centered at 0.7)
                score = random.gauss(0.7, noise_level)
            else:
                # Target: Low score (centered at 0.3)
                score = random.gauss(0.3, noise_level)
            # Clip to 0-1 range
            probs.append(max(0.0, min(1.0, score)))
        return probs


def run_comparison(name: str, prevalence: float):
    logger.info(f"\n--- DATASET: {name.upper()} (Prevalence: {prevalence*100}%) ---")
    
    # 1. Generate Data
    y_true = [1 if random.random() < prevalence else 0 for _ in range(TOTAL_SAMPLES)]
    y_probs = MockModel.get_predictions(y_true, noise_level=0.25)
    
    # 2. Generate Curve Points
    points = CurveCalculator.get_curve_points(y_true, y_probs)
    
    # 3. Calculate AUCs
    roc_auc = CurveCalculator.calculate_auc([p['fpr'] for p in points], [p['tpr'] for p in points])
    pr_auc = CurveCalculator.calculate_auc([p['recall'] for p in points], [p['precision'] for p in points])
    
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC:  {pr_auc:.4f}")
    
    if roc_auc > pr_auc + 0.2:
        logger.info(f"[Insight]: Huge gap detected! ROC-AUC is being inflated by the {TOTAL_SAMPLES} negative cases.")
        logger.info(f"           PR-AUC is the 'honest' metric for this imbalanced data.")
    else:
        logger.info(f"[Insight]: Metrics are similar. ROC-AUC is a safe choice for balanced data.")


def run_lesson():
    logger.info("=== LESSON 2: ROC-AUC vs. PR-AUC ===\n")
    
    # Scenario A: Balanced (50/50)
    run_comparison("Balanced Data", prevalence=0.50)
    
    # Scenario B: Imbalanced (1%)
    run_comparison("Highly Imbalanced Data", prevalence=IMBALANCE_RATIO)


if __name__ == "__main__":
    run_lesson()

# ----------------------------------------------------------------------
# REFERENCE: ROC vs PR AUC
# ----------------------------------------------------------------------
# 1. ROC-AUC (The Standard)
#    - Measures: Separation between classes.
#    - Use when: Classes are balanced OR you care about both classes equally.
#    - Baseline: 0.5 (Random guessing).
#
# 2. PR-AUC (The Specialist)
#    - Measures: Quality of the Positive class specifically.
#    - Use when: Classes are imbalanced (Fraud, Search, NLP triggers).
#    - Baseline: Equals the Prevalence (e.g., if only 1% is positive, baseline is 0.01).
#
# 3. WHAT IS A GOOD SCORE?
#    - 0.90+: Excellent. Model is very "separable".
#    - 0.70-0.80: Acceptable for most production NLP.
#    - 0.50: No better than a coin flip.
#
# Rule of Thumb: If prevalence < 5%, ignore ROC-AUC. It will lie to you.
# ----------------------------------------------------------------------