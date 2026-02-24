"""
Data Sampling Series: Level 2 - The "Cold Start" (Heuristic Sampling)

Focus:
Solving the "Cold Start" problem. When you have 10 Million raw logs and ZERO 
labeled data, you cannot train a model or use Active Learning (Level 3). 
Instead, we use Programmatic Labeling (Weak Supervision) to filter the noise.

When is this approach "Enough"?
- Bootstrapping V1 Models: When you need 5,000 examples by Friday.
- Compliance & Triage: When looking for exact strings (e.g., SSNs or specific error codes).
- Data Routing: Sending highly-confident data straight to a database while routing 
  confusing data to human support agents.

When does it fail?
- Semantic Nuance: It cannot understand context.
- The "Unknown Unknowns": It only finds what you tell it to find. If users 
  invent a new slang term for "cancel", this system is blind to it.
"""

import logging
import re
from collections import Counter
from typing import List, Dict, Callable, Optional

# --- TUNABLE PARAMETERS ---
# Minimum agreement required among rules to bypass human review
CONSENSUS_THRESHOLD = 0.70  

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level2_WeakSupervision")


class LabelingFunctions:
    """
    Expert Technique: Independent Heuristics.
    Instead of one massive, brittle Regex, we write small, targeted functions.
    Each LF votes on the intent, or returns None if it has no opinion.
    """

    @staticmethod
    def lf_keyword_cancel(text: str) -> Optional[str]:
        """High precision, low recall rule for cancellations."""
        keywords = ["cancel", "terminate", "close account", "unsubscribe"]
        if any(word in text.lower() for word in keywords):
            return "CANCEL_INTENT"
        return None

    @staticmethod
    def lf_keyword_downgrade(text: str) -> Optional[str]:
        """Detects requests for cheaper tiers."""
        keywords = ["downgrade", "cheaper", "lower tier", "basic plan"]
        if any(word in text.lower() for word in keywords):
            return "DOWNGRADE_INTENT"
        return None

    @staticmethod
    def lf_regex_billing(text: str) -> Optional[str]:
        """Uses regex to catch complaints about money or invoices."""
        if re.search(r'\b(charge|bill|invoice|refund|$$)\b', text.lower()):
            return "BILLING_ISSUE"
        return None

    @staticmethod
    def lf_is_question(text: str) -> Optional[str]:
        """Syntactic heuristic looking for interrogatives."""
        if "?" in text or text.lower().startswith(("how", "why", "can i", "what")):
            return "QUESTION"
        return None


# Registry of our active rules
ACTIVE_LFS: List[Callable] = [
    LabelingFunctions.lf_keyword_cancel,
    LabelingFunctions.lf_keyword_downgrade,
    LabelingFunctions.lf_regex_billing,
    LabelingFunctions.lf_is_question
]


class WeakSupervisionEngine:
    """
    The orchestrator. Applies all rules, calculates consensus, and routes 
    data to either an Auto-Labeled database or a Human Review queue.
    """
    
    @staticmethod
    def process_dataset(dataset: List[Dict]) -> Dict[str, List[Dict]]:
        routed_data = {
            "auto_labeled": [],
            "human_review": [],
            "unknown_unknowns": []  # Data missed by all heuristics
        }

        for doc in dataset:
            text = doc["text"]
            
            # 1. Gather votes from all LFs
            votes = [lf(text) for lf in ACTIVE_LFS]
            
            # 2. Filter out abstentions (None)
            valid_votes = [v for v in votes if v is not None]

            # 3. Handle data that bypassed all rules
            if not valid_votes:
                routed_data["unknown_unknowns"].append(doc)
                continue

            # 4. Calculate Consensus
            vote_counts = Counter(valid_votes)
            top_label, top_count = vote_counts.most_common(1)[0]
            consensus_ratio = top_count / len(valid_votes)

            doc["lf_votes"] = dict(vote_counts)
            doc["consensus"] = consensus_ratio

            # 5. Routing Logic
            # If rules disagree, or confidence is too low, send to humans
            if len(vote_counts) > 1 and consensus_ratio < CONSENSUS_THRESHOLD:
                routed_data["human_review"].append(doc)
            
            # If rules strongly agree, save money and auto-label
            elif consensus_ratio >= CONSENSUS_THRESHOLD:
                doc["final_label"] = top_label
                routed_data["auto_labeled"].append(doc)
                
            else:
                routed_data["human_review"].append(doc)

        return routed_data


# --- TUTORIAL DEMONSTRATIONS ---

def run_lesson():
    logger.info("=== LESSON 2: WEAK SUPERVISION & DATA ROUTING ===\n")

    # A mock dataset representing real-world user utterances
    raw_dataset = [
        # The Obvious Wins
        {"id": 1, "text": "I want to cancel my account right now."},
        {"id": 2, "text": "Where is my refund for the last invoice?"},
        
        # The Conflicts (Overlapping Rules)
        {"id": 3, "text": "Can I cancel my account, or just downgrade?"}, 
        {"id": 4, "text": "Why is there a weird charge on my bill?"}, 
        
        # The Trap (Unknown Unknowns)
        {"id": 5, "text": "I love this product, it works great!"}, # True negative
        {"id": 6, "text": "I'm done with you guys, pull the plug on my sub."}, # OOV Cancellation!
        {"id": 7, "text": "Stop taking money out of my bank."} # OOV Billing!
    ]

    logger.info(f"Processing {len(raw_dataset)} raw logs through {len(ACTIVE_LFS)} Labeling Functions...\n")
    results = WeakSupervisionEngine.process_dataset(raw_dataset)

    # --- DEMO 1: The Triage Win ---
    logger.info("--- 1. THE TRIAGE WIN (Auto-Labeled) ---")
    logger.info(f"Saved annotation budget on {len(results['auto_labeled'])} obvious items.")
    for doc in results['auto_labeled']:
        logger.info(f"   [{doc['final_label']:<15}] : '{doc['text']}' (Agreement: {doc['consensus']*100:.0f}%)")

    # --- DEMO 2: The Conflict ---
    logger.info("\n--- 2. THE CONFLICT (Routed to Humans) ---")
    logger.info(f"Routed {len(results['human_review'])} complex items to human experts.")
    for doc in results['human_review']:
        logger.info(f"   Votes: {doc['lf_votes']} -> '{doc['text']}'")
    logger.info("   [Takeaway]: Rather than rules silently overwriting each other, we detect the conflict and leverage human nuance.")

    # --- DEMO 3: The Fatal Flaw ---
    logger.info("\n--- 3. THE UNKNOWN UNKNOWNS TRAP (Missed Data) ---")
    logger.info(f"Ignored {len(results['unknown_unknowns'])} items because no keywords matched.")
    for doc in results['unknown_unknowns']:
        logger.info(f"   Ignored: '{doc['text']}'")
        
    logger.info("\n[Critical Analysis]:")
    logger.info("Look at items 6 and 7. 'Pull the plug' is a cancellation. 'Stop taking money' is a billing issue.")
    logger.info("Because our users didn't use our exact dictionary words, the system completely ignored them.")
    logger.info("If we train a model ONLY on our auto-labeled data, the model will become an 'Echo Chamber' of our Regex.")
    logger.info("To fix this, we must eventually move toward Uncertainty or Semantic Clustering to find the intents we couldn't think of.")


if __name__ == "__main__":
    run_lesson()