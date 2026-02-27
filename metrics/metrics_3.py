"""
Metrics Series: Lesson 3 - The Human Element (Multi-Class & Agreement)

The Challenge:
You are training a "Safety Guardrail" model for an LLM with 4 classes:
1. SAFE (80% of traffic)
2. TOXIC (15%)
3. SELF_HARM (4% - Rare but Critical)
4. GIBBERISH (1%)

The Trap:
1. Micro-F1: If your model identifies 'SAFE' perfectly but misses every 'SELF_HARM' 
   event, Micro-F1 stays high (~80%+). It hides the failure of rare classes.
2. Raw Agreement: If two humans agree 80% of the time, is that good? 
   If 80% of the data is 'SAFE', they could both just close their eyes, 
   guess 'SAFE' every time, and get 80% agreement by accident.

The Solution:
1. Macro-F1: Averages performance per-class, giving the rare 'SELF_HARM' class 
   equal voting power to 'SAFE'.
2. Cohen's Kappa: Mathematically subtracts the "Probability of Random Agreement" 
   from the observed agreement.
"""

import random
import logging
from collections import Counter
from typing import List, Dict, Tuple

# --- TUNABLE PARAMETERS ---
TOTAL_SAMPLES = 1000
CLASSES = ["SAFE", "TOXIC", "SELF_HARM", "GIBBERISH"]
DISTRIBUTION = [0.80, 0.15, 0.04, 0.01] # Sums to 1.0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Metrics_Lesson_3")


class MultiClassEvaluator:
    """Implements F1 variants and Agreement math from scratch."""

    @staticmethod
    def calculate_per_class_f1(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        scores = {}
        for cls in CLASSES:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            scores[cls] = f1
        return scores

    @staticmethod
    def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
        """Arithmetic mean of per-class F1 scores. Treats rare classes as equals."""
        scores = MultiClassEvaluator.calculate_per_class_f1(y_true, y_pred)
        return sum(scores.values()) / len(scores)

    @staticmethod
    def micro_f1(y_true: List[str], y_pred: List[str]) -> float:
        """Global calculation of TP/FP/FN. Biased towards the majority class."""
        total_tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        # In multi-class, total FP and FN are equal to the total errors
        total_errors = sum(1 for t, p in zip(y_true, y_pred) if t != p)
        
        # In micro settings, Precision = Recall = F1 = Accuracy
        return total_tp / (total_tp + total_errors)

    @staticmethod
    def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
        """
        Calculates agreement adjusted for chance.
        Kappa = (Po - Pe) / (1 - Pe)
        Po: Observed agreement.
        Pe: Expected agreement by random chance.
        """
        total = len(labels_a)
        if total == 0: return 0.0

        # 1. Observed Agreement (Po)
        observed_matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
        po = observed_matches / total

        # 2. Expected Agreement (Pe)
        # How often would they agree if they just guessed based on label frequency?
        count_a = Counter(labels_a)
        count_b = Counter(labels_b)
        
        pe = 0.0
        for cls in CLASSES:
            prob_a = count_a[cls] / total
            prob_b = count_b[cls] / total
            pe += (prob_a * prob_b)
            
        # 3. Final Kappa
        if pe == 1: return 1.0 # Perfect agreement
        return (po - pe) / (1 - pe)


class DataSimulator:
    """Generates ground truth and simulated human/model labels."""

    @staticmethod
    def generate_ground_truth() -> List[str]:
        return random.choices(CLASSES, weights=DISTRIBUTION, k=TOTAL_SAMPLES)

    @staticmethod
    def simulate_model_results(y_true: List[str], type: str) -> List[str]:
        """
        'Lazy': Perfectly predicts 'SAFE', fails everything else.
        'Specialist': Lower accuracy on 'SAFE', but tries hard on rare classes.
        """
        y_pred = []
        for label in y_true:
            if type == "Lazy":
                y_pred.append("SAFE") # The majority-class bypass
            else:
                # 70% chance to be right, 30% chance to guess randomly
                y_pred.append(label if random.random() < 0.70 else random.choice(CLASSES))
        return y_pred

    @staticmethod
    def simulate_annotators(y_true: List[str]) -> Tuple[List[str], List[str]]:
        """Simulates two humans who agree a lot but might just be biased."""
        alice, bob = [], []
        for label in y_true:
            # Both humans are biased toward 'SAFE'
            # Alice is very careful
            alice.append(label if random.random() < 0.90 else "SAFE")
            # Bob is lazy and defaults to 'SAFE' even more
            bob.append(label if random.random() < 0.60 else "SAFE")
        return alice, bob


def run_lesson():
    logger.info("=== LESSON 3: MULTI-CLASS & HUMAN AGREEMENT ===\n")
    y_true = DataSimulator.generate_ground_truth()
    
    # --- DEMO 1: MICRO vs MACRO F1 ---
    logger.info("--- 1. THE CLASS IMBALANCE SHADOW (Micro vs Macro) ---")
    y_lazy = DataSimulator.simulate_model_results(y_true, "Lazy")
    y_spec = DataSimulator.simulate_model_results(y_true, "Specialist")
    
    for name, preds in [("Lazy Model (SAFE-only)", y_lazy), ("Balanced Specialist", y_spec)]:
        micro = MultiClassEvaluator.micro_f1(y_true, preds)
        macro = MultiClassEvaluator.macro_f1(y_true, preds)
        per_class = MultiClassEvaluator.calculate_per_class_f1(y_true, preds)
        
        logger.info(f"\nModel: {name}")
        logger.info(f"   Micro-F1: {micro:.3f} (Looks good, right?)")
        logger.info(f"   Macro-F1: {macro:.3f} (The class-average truth)")
        logger.info(f"   Rare Class (SELF_HARM) F1: {per_class['SELF_HARM']:.3f}")

    logger.info("\n[Analysis]: The Lazy model has a high Micro-F1 (~0.80) because it rides the coattails of the 'SAFE' majority. "
                "Macro-F1 exposes it by showing a low average score, highlighting that the model is failing on rare intents.")


    # --- DEMO 2: COHEN'S KAPPA ---
    logger.info("\n\n--- 2. AGREEMENT BY ACCIDENT (Cohen's Kappa) ---")
    alice, bob = DataSimulator.simulate_annotators(y_true)
    
    raw_agree = sum(1 for a, b in zip(alice, bob) if a == b) / TOTAL_SAMPLES
    kappa = MultiClassEvaluator.cohens_kappa(alice, bob)
    
    logger.info(f"Annotator A (Alice) vs Annotator B (Bob):")
    logger.info(f"   Raw Observed Agreement: {raw_agree*100:.1f}%")
    logger.info(f"   Cohen's Kappa Score:    {kappa:.3f}")
    
    logger.info("\n[Analysis]: Alice and Bob agree ~85% of the time, which sounds great! "
                "But Kappa is much lower (~0.5-0.6).")
    logger.info("This is because in a dataset that is 80% 'SAFE', any two people would agree "
                "on 'SAFE' 64% of the time (0.8 * 0.8) just by closing their eyes. "
                "Kappa removes that 'free' agreement and measures only genuine consensus.")


if __name__ == "__main__":
    run_lesson()

# ----------------------------------------------------------------------
# REFERENCE: Interpretation of Cohen's Kappa
# ----------------------------------------------------------------------
# < 0    : Poor (Agreement worse than random chance)
# 0-0.20 : Slight
# 0.21-0.40 : Fair
# 0.41-0.60 : Moderate
# 0.61-0.80 : Substantial (The "Gold Standard" for high-quality NLP data)
# 0.81-1.00 : Almost Perfect (Usually only seen in trivial tasks)
#
# scikit-learn equivalent:
# from sklearn.metrics import f1_score, cohen_kappa_score
# macro = f1_score(y_true, y_pred, average='macro')
# kappa = cohen_kappa_score(annotator_1, annotator_2)
# ----------------------------------------------------------------------