"""
Metrics Series: Lesson 1 - The Precision, Recall, and Accuracy Dilemma

The Challenge:
If you ask a junior data scientist how good their model is, they will say 
"95% Accurate!" But in the real world, Accuracy is only valid when classes 
are perfectly balanced AND the cost of being wrong is identical for both classes.

The Solution:
We decompose predictions into a Confusion Matrix (True Positives, False Positives, etc.).
We then use Precision, Recall, and the F-Beta score to measure performance 
based on specific business constraints.

The 3 Scenarios Demonstrated:
1. The Cat/Dog Classifier (Balanced - Accuracy works).
2. The Auto-Ban Spam Filter (Precision Focus - Don't ban innocent users).
3. The Cancer Screening (Recall Focus - Don't let sick people go undiagnosed).
"""


import random
import logging
from typing import List, Tuple

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Metrics_Lesson_1")


class ConfusionMatrix:
    """The mathematical foundation of all classification metrics."""
    
    def __init__(self, y_true: List[int], y_pred: List[int]):
        self.tp = 0  # True Positives (Hit)
        self.tn = 0  # True Negatives (Correct Reject)
        self.fp = 0  # False Positives (False Alarm / Type I Error)
        self.fn = 0  # False Negatives (Miss / Type II Error)

        for truth, pred in zip(y_true, y_pred):
            if truth == 1 and pred == 1: self.tp += 1
            elif truth == 0 and pred == 0: self.tn += 1
            elif truth == 0 and pred == 1: self.fp += 1
            elif truth == 1 and pred == 0: self.fn += 1

    def accuracy(self) -> float:
        """(TP + TN) / Total"""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def precision(self) -> float:
        """TP / (TP + FP). Quality of the alarms. 'When I fire, am I right?'"""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def recall(self) -> float:
        """TP / (TP + FN). Quantity of the targets caught. 'Did I catch them all?'"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def f1_score(self) -> float:
        """The Harmonic Mean of Precision and Recall. A balanced view."""
        p = self.precision()
        r = self.recall()
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    def f_beta_score(self, beta: float) -> float:
        """
        Weights Recall vs Precision. 
        Beta = 0.5: Precision is 2x more important than Recall.
        Beta = 2.0: Recall is 2x more important than Precision.
        """
        p = self.precision()
        r = self.recall()
        beta_sq = beta ** 2
        numerator = (1 + beta_sq) * (p * r)
        denominator = (beta_sq * p) + r
        return numerator / denominator if denominator > 0 else 0.0


class MockModelSimulator:
    """Generates ground truth and predictions tailored to specific business profiles."""
    
    @staticmethod
    def run_simulation(total_samples: int, prevalence: float, recall_rate: float, false_alarm_rate: float) -> Tuple[List[int], List[int]]:
        """
        prevalence: % of data that is Class 1 (True).
        recall_rate: % of Class 1 the model successfully catches (True Positive Rate).
        false_alarm_rate: % of Class 0 the model accidentally flags as Class 1 (False Positive Rate).
        """
        y_true = []
        y_pred = []
        
        for _ in range(total_samples):
            # Generate Reality
            is_class_1 = 1 if random.random() < prevalence else 0
            y_true.append(is_class_1)
            
            # Generate Prediction
            if is_class_1:
                # Does the model catch it?
                pred = 1 if random.random() < recall_rate else 0
            else:
                # Does the model false alarm?
                pred = 1 if random.random() < false_alarm_rate else 0
            y_pred.append(pred)
            
        return y_true, y_pred


def print_evaluation(name: str, y_true: List[int], y_pred: List[int], beta: float):
    """Helper to log the metrics cleanly."""
    cm = ConfusionMatrix(y_true, y_pred)
    
    logger.info(f"\n--- EVALUATING: {name.upper()} ---")
    logger.info(f"Confusion Matrix: TP:{cm.tp:<5} FN:{cm.fn:<5} | FP:{cm.fp:<5} TN:{cm.tn:<5}")
    logger.info("-" * 65)
    
    logger.info(f"Accuracy:  {cm.accuracy() * 100:>5.1f}%")
    logger.info(f"Precision: {cm.precision() * 100:>5.1f}%")
    logger.info(f"Recall:    {cm.recall() * 100:>5.1f}%")
    logger.info(f"F1 Score:  {cm.f1_score() * 100:>5.1f}%")
    logger.info(f"F-Beta({beta}): {cm.f_beta_score(beta) * 100:>5.1f}%")


def run_lesson():
    logger.info("=== LESSON 1: WHEN TO USE WHICH METRIC ===\n")

    # -------------------------------------------------------------------------
    # SCENARIO A: Balanced Data (The Cat/Dog Classifier)
    # -------------------------------------------------------------------------
    logger.info("SCENARIO A: The Balanced Problem (Cats vs Dogs)")
    logger.info("Classes are 50/50. False Positives and False Negatives are equally annoying.")
    
    # Model catches 90% of dogs, and false-alarms on 10% of cats
    truth_A, pred_A = MockModelSimulator.run_simulation(
        total_samples=10_000, prevalence=0.50, recall_rate=0.90, false_alarm_rate=0.10
    )
    print_evaluation("Balanced Model", truth_A, pred_A, beta=1.0)
    logger.info("[Takeaway]: Here, Accuracy, Precision, and Recall are all ~90%. Accuracy is a perfectly valid and honest metric for balanced datasets.")


    # -------------------------------------------------------------------------
    # SCENARIO B: Precision is King (The Spam Auto-Ban)
    # -------------------------------------------------------------------------
    logger.info("\n\nSCENARIO B: Precision Priority (YouTube Spam Auto-Bans)")
    logger.info("Spam is 10%. If we ban an innocent user (False Positive), we lose a customer forever.")
    logger.info("We tune the model to be EXTREMELY conservative. We only ban if we are 99.9% sure.")
    
    # Model only catches 40% of spam (Low Recall), but false alarms on only 0.1% of safe users (High Precision)
    truth_B, pred_B = MockModelSimulator.run_simulation(
        total_samples=10_000, prevalence=0.10, recall_rate=0.40, false_alarm_rate=0.001
    )
    
    # We use Beta=0.5 to tell the math that Precision is 2x more important than Recall
    print_evaluation("Conservative Spam Filter", truth_B, pred_B, beta=0.5)
    logger.info("[Takeaway]: Recall is terrible (40%), meaning 60% of spam gets through. BUT Precision is ~97%! When this model fires, it is right. The F-Beta(0.5) score reflects our business goal of safety first.")


    # -------------------------------------------------------------------------
    # SCENARIO C: Recall is King (Cancer Screening)
    # -------------------------------------------------------------------------
    logger.info("\n\nSCENARIO C: Recall Priority & The Base Rate Fallacy (Cancer Screening)")
    logger.info("Disease is 1% of patients. If we miss it (False Negative), they die.")
    logger.info("If we False Alarm (False Positive), they just get an extra biopsy. We MUST maximize Recall.")
    
    # The Trap: The Lazy Model
    truth_C_lazy, pred_C_lazy = MockModelSimulator.run_simulation(
        total_samples=10_000, prevalence=0.01, recall_rate=0.0, false_alarm_rate=0.0
    )
    print_evaluation("Lazy Model (Always guesses 'Healthy')", truth_C_lazy, pred_C_lazy, beta=2.0)
    logger.info("[Danger]: The lazy model has 99% Accuracy, but 0% Recall. Accuracy lies to us.")

    # The Solution: The Paranoid/Sensitive Model
    # Catches 99% of cancer, but has a 5% false alarm rate on healthy people.
    truth_C, pred_C = MockModelSimulator.run_simulation(
        total_samples=10_000, prevalence=0.01, recall_rate=0.99, false_alarm_rate=0.05
    )
    print_evaluation("Highly Sensitive Screener", truth_C, pred_C, beta=2.0)
    
    logger.info("[The Base Rate Fallacy]: Look at that low Precision (~16%)! Why?")
    logger.info("Because 99% of the 10,000 patients are healthy (9,900). A 5% false alarm rate creates ~495 False Positives.")
    logger.info("There are only 100 sick people. So our True Positives (99) are drowned out by False Positives (495).")
    logger.info("This is mathematically unavoidable in rare events. By using F-Beta(2.0), we prove to stakeholders that despite the low Precision, the model successfully achieved its goal of finding the sick patients.")


if __name__ == "__main__":
    run_lesson()

# ----------------------------------------------------------------------
# Industry Standard Implementation (scikit-learn):
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
# print("Precision:", precision_score(y_true, y_pred))
# print("F2 Score:", fbeta_score(y_true, y_pred, beta=2.0))
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# REFERENCE: WHAT IS A "GOOD" SCORE?
# ----------------------------------------------------------------------
# There is no universal "good" number. It depends on the Difficulty and Cost.
#
# 1. ACCURACY
#    - Good: > 90% (ONLY if classes are balanced 50/50).
#    - Bad:  99% (If the target class is only 1% of data).
#    - Rule: Accuracy must beat the "No Information Rate" (Majority Class %).
#
# 2. PRECISION (The "Trust" Metric)
#    - Target: > 99% for Automated Actions (Banning users, deleting files).
#    - Target: > 60% for "Suggestions" (Recommendations, Search results).
#
# 3. RECALL (The "Safety" Metric)
#    - Target: > 99% for Critical Safety (Cancer diag, Bomb detection).
#    - Target: > 80% for General Retrieval (Finding relevant documents).
#
# 4. F1-SCORE (The Balance)
#    - Target: > 70% is generally considered "Production Ready" for NLP.
#    - Target: > 90% is State-of-the-Art (SOTA) for easy tasks.
#
# 5. F-BETA (The Rare Event Reality)
#    - Scenario: Finding a needle in a haystack (1 in 10,000 prevalence).
#    - Reality: A score of ~40-50% is often a HUGE SUCCESS.
#    - Why? Because getting even 20% Precision on a 0.01% event is statistically 
#      difficult. Don't let a low F-Beta discourage you if the task is rare!
# ----------------------------------------------------------------------