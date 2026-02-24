"""
Data Sampling Series: Level 1 - Baselines & Stratification

Focus:
Creating mathematically sound datasets for Model Evaluation (Test Sets) 
or initial training baselines. We explore the silent dangers of Pure Random 
Sampling on imbalanced data, and how to fix it using Proportional Stratification 
and Minimum-Threshold Stratification.

Use Cases:
- Generating Golden Test Sets (where real-world distribution matters).
- Ensuring rare, high-risk edge cases (e.g., Fraud, Emergencies) are 
  statistically represented in the evaluation metrics.

Note:
In production, you would typically use stratification via:
`sklearn.model_selection.train_test_split(stratify=y)`.
Here, we implement it from scratch to demonstrate the exact failure states and math.
"""

import random
import logging
import math
from collections import defaultdict
from typing import List, Dict

# --- TUNABLE PARAMETERS ---
TOTAL_POPULATION = 100_000
SAMPLE_BUDGET = 250

# The real-world distribution of our data (Highly Imbalanced)
CLASS_DISTRIBUTION = {
    "greetings": 0.85,          # 85% of traffic is simple hellos
    "password_reset": 0.13,     # 13% is standard IT support
    "fraud_alert": 0.02,         # 2% is rare but critical emergencies
}

# For Threshold Stratification: We demand at least this many examples 
# of EVERY class to ensure statistical significance during evaluation.
MINIMUM_CLASS_REPRESENTATION = int(SAMPLE_BUDGET * 0.03)

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level1_Baselines")


class MockDataGenerator:
    """Generates a highly imbalanced dataset representing real-world traffic."""
    
    @staticmethod
    def generate(population_size: int, distribution: Dict[str, float]) -> List[Dict]:
        dataset = []
        for i in range(population_size):
            # Pick a class based on the defined probability distribution
            rand_val = random.random()
            cumulative = 0.0
            chosen_class = "unknown"
            
            for cls_name, prob in distribution.items():
                cumulative += prob
                if rand_val <= cumulative:
                    chosen_class = cls_name
                    break
                    
            dataset.append({
                "id": f"doc_{i}",
                "intent": chosen_class,
                "text": f"Simulated text for {chosen_class}..."
            })
        
        # Shuffle to simulate a raw, unorganized database dump
        random.shuffle(dataset)
        return dataset


class Sampler:
    """Core algorithms for Baseline Data Selection."""

    @staticmethod
    def pure_random(dataset: List[Dict], budget: int) -> List[Dict]:
        """
        Level 1A: Pure Random Sampling.
        Danger: Likely to completely miss minority classes in small budgets.
        """
        return random.sample(dataset, budget)

    @staticmethod
    def proportional_stratified(dataset: List[Dict], budget: int) -> List[Dict]:
        """
        Level 1B: Proportional Stratified Sampling.
        Ensures the sample exactly matches the population's percentage breakdown.
        """
        # 1. Group by class
        grouped_data = defaultdict(list)
        for item in dataset:
            grouped_data[item["intent"]].append(item)
            
        total_items = len(dataset)
        sampled_data = []
        
        # 2. Sample proportionally from each group
        for cls_name, items in grouped_data.items():
            # Calculate exact proportion
            proportion = len(items) / total_items
            target_count = round(budget * proportion)
            
            # Handle rounding edge cases (don't sample more than exists)
            actual_count = min(target_count, len(items))
            sampled_data.extend(random.sample(items, actual_count))
            
        # 3. Handle leftover budget due to rounding errors
        shortfall = budget - len(sampled_data)
        if shortfall > 0:
            # Randomly fill the rest from unpicked data
            picked_ids = {d["id"] for d in sampled_data}
            unpicked = [d for d in dataset if d["id"] not in picked_ids]
            sampled_data.extend(random.sample(unpicked, shortfall))
            
        return sampled_data

    @staticmethod
    def threshold_stratified(dataset: List[Dict], budget: int, min_count: int) -> List[Dict]:
        """
        Expert Technique: Threshold Stratification.
        If a class is 1% of the data, and budget is 100, proportional gives 1 item.
        1 item is not enough to calculate statistical accuracy. We must force a minimum.
        """
        grouped_data = defaultdict(list)
        for item in dataset:
            grouped_data[item["intent"]].append(item)
            
        sampled_data = []
        budget_remaining = budget
        
        # Step 1: Secure the minimum threshold for EVERY class
        for cls_name, items in grouped_data.items():
            allocation = min(min_count, len(items))
            sampled_data.extend(random.sample(items, allocation))
            budget_remaining -= allocation
            
            # Remove picked items so we don't pick them again
            grouped_data[cls_name] = [item for item in items if item not in sampled_data]

        if budget_remaining < 0:
            logger.warning("Budget too small to satisfy minimum thresholds!")
            return sampled_data[:budget]

        # Step 2: Distribute the remaining budget proportionally
        total_remaining_items = sum(len(items) for items in grouped_data.values())
        
        for cls_name, items in grouped_data.items():
            if total_remaining_items == 0: break
            
            proportion = len(items) / total_remaining_items
            target_count = round(budget_remaining * proportion)
            actual_count = min(target_count, len(items))
            
            sampled_data.extend(random.sample(items, actual_count))

        return sampled_data[:budget] # Cap to exact budget


def evaluate_sample(sample_name: str, sample: List[Dict]):
    """Helper to log the distribution of the selected sample."""
    counts = defaultdict(int)
    for item in sample:
        counts[item["intent"]] += 1
        
    logger.info(f"\n--- {sample_name.upper()} RESULTS ---")
    logger.info(f"Total Sample Size: {len(sample)}")
    
    for cls_name in CLASS_DISTRIBUTION.keys():
        count = counts.get(cls_name, 0)
        percentage = (count / len(sample)) * 100 if len(sample) > 0 else 0
        
        # Highlight danger if a critical class is missing or too low
        warning = ""
        if cls_name == "fraud_alert" and count < MINIMUM_CLASS_REPRESENTATION:
            warning = " ⚠️ DANGER: Insufficient data for evaluation!"
            
        logger.info(f"   {cls_name:<15}: {count:>3} items ({percentage:>5.1f}%){warning}")


def run_lesson():
    logger.info("=== LESSON 1: BASELINES & STRATIFICATION ===")
    logger.info(f"Generating Universe of {TOTAL_POPULATION:,} items...")
    
    dataset = MockDataGenerator.generate(TOTAL_POPULATION, CLASS_DISTRIBUTION)
    
    # 1. The Naive Approach
    logger.info("\n[Scenario 1]: The Naive Data Scientist")
    logger.info(f"Action: Selecting {SAMPLE_BUDGET} items using pure random.sample().")
    random_sample = Sampler.pure_random(dataset, SAMPLE_BUDGET)
    evaluate_sample("Pure Random Sampling", random_sample)
    logger.info("[Takeaway]: Pure random sampling is blind to distribution. Critical minority classes are often ignored entirely.")

    # 2. The Standard Approach
    logger.info("\n[Scenario 2]: The Standard Data Scientist")
    logger.info(f"Action: Selecting {SAMPLE_BUDGET} items using Proportional Stratification.")
    prop_sample = Sampler.proportional_stratified(dataset, SAMPLE_BUDGET)
    evaluate_sample("Proportional Stratified", prop_sample)
    logger.info("[Takeaway]: The distribution matches reality perfectly. However, 2% of a small budget is still a tiny number. We can't trust an accuracy score based on 3 or 4 examples.")

    # 3. The Expert Approach
    logger.info("\n[Scenario 3]: The Expert Language Engineer")
    logger.info(f"Action: Selecting {SAMPLE_BUDGET} items with a guaranteed MINIMUM of {MINIMUM_CLASS_REPRESENTATION} items per class.")
    thresh_sample = Sampler.threshold_stratified(dataset, SAMPLE_BUDGET, MINIMUM_CLASS_REPRESENTATION)
    evaluate_sample("Threshold Stratified", thresh_sample)
    logger.info(f"[Takeaway]: We deliberately distorted the real-world distribution to ensure statistical significance for minorities. We evaluate the model safely, and then apply 'Sample Weights' during final metric calculation to correct the distortion.")


if __name__ == "__main__":
    run_lesson()