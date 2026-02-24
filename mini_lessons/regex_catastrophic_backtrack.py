"""
Mini-Lesson: Regex Explosion & Catastrophic Backtracking

Context:
In Data Sampling Level 2 (Heuristics), we use Regular Expressions to route data.
If you apply a vulnerable Regex to a 10 Million row dataset, a single malformed 
string can cause the CPU to hang indefinitely, crashing the pipeline.

The Trap (Catastrophic Backtracking):
When a Regex uses nested greedy quantifiers (e.g., `(a+)+` or `(\w+\s?)*`), 
and the target string *almost* matches but fails at the very end, the engine 
tries every single permutation of the nested groups before giving up. 
This results in O(2^N) exponential time complexity. 

Because \w+ can match "word", but also "wor" and "d", the engine tries slicing 
the string into thousands of tiny pieces. This causes an exponential explosion.

The Fix:
- Unroll the loops (remove nested quantifiers).
- Make quantifiers possessive or use atomic groups.
"""

import re
import time
import sys
import threading
import logging
from typing import List, Tuple

# --- TUNABLE PARAMETERS ---
# Lengths of the malformed string (number of words). 
# We cap at 9 (approx 15-20 seconds on modern CPUs) to prevent hard freezes.
DANGER_LENGTHS = [4, 6, 7, 8, 9]

# A massive test length for the optimized regex to prove it scales safely
SAFE_TEST_LENGTH = 50_000 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("RegexMiniLesson")


class RegexDemonstrator:
    def __init__(self):
        # THE EVIL REGEX
        # The Flaw: The * is nested outside the \w+ and optional \s?
        self.evil_regex = re.compile(r'^(\w+\s?)*$')

        # THE SAFE REGEX
        # The Fix: Unroll the loop. Match ONE word. Then optionally match 
        # (space + word) multiple times. No nested ambiguous quantifiers.
        self.safe_regex = re.compile(r'^\w+(?:\s\w+)*\s?$')

    @staticmethod
    def measure_execution_time(pattern: re.Pattern, text: str) -> float:
        """Runs the regex, and returns time."""
        done_event = threading.Event()
        
        print(f"\r   Evaluating {'. ' * 3}", end="\r")

        # --- THE BLOCKING C-LEVEL REGEX CALL ---
        start_time = time.perf_counter()
        pattern.match(text)
        end_time = time.perf_counter()

        return end_time - start_time


def run_lesson():
    logger.info("=== MINI-LESSON: CATASTROPHIC BACKTRACKING ===\n")
    demo = RegexDemonstrator()

    # --- DEMO 1: The Happy Path ---
    logger.info("--- 1. THE HAPPY PATH ---")
    valid_string = "word " * 10 + "word"
    
    evil_time = demo.measure_execution_time(demo.evil_regex, valid_string)
    safe_time = demo.measure_execution_time(demo.safe_regex, valid_string)
    
    logger.info(f"Target: A perfectly valid string ('{valid_string[:20]}...')")
    logger.info(f"   Evil Regex Time: {evil_time:.6f} seconds")
    logger.info(f"   Safe Regex Time: {safe_time:.6f} seconds")
    logger.info("[Takeaway]: On valid data, the bad regex works instantly. This is why it slips into production.\n")

    # --- DEMO 2: The Explosion ---
    logger.info("--- 2. THE EXPLOSION (Exponential Time Complexity) ---")
    logger.info("Target: A string of words that ends with an INVALID character ('!').")
    logger.info("The engine must try to match it, fail, and backtrack...\n")
    
    explosion_results: List[Tuple[int, float]] = []

    for length in DANGER_LENGTHS:
        logger.info(f"Run for Length {length}...")
        malformed_string = ("word " * length) + "!"
        
        exec_time = demo.measure_execution_time(demo.evil_regex, malformed_string)
        
        logger.info(f"   -> Failed after {exec_time:.4f} seconds.")
        logger.info("-" * 45)
        
        explosion_results.append((length, exec_time))

    # --- SUMMARY TABLE ---
    logger.info("\n=== RESULTS SUMMARY ===")
    logger.info(f"{'String Length (Words)':<22} | {'Execution Time (Seconds)':<25}")
    logger.info("-" * 50)
    for length, exec_time in explosion_results:
        logger.info(f"{length:<22} | {exec_time:.5f} sec")

    logger.info("\n[Critical Analysis]:")
    logger.info("Notice how adding just 1 word multiplies the execution time massively.")
    logger.info("If we tested length 15, it would take hours. Length 20 would take centuries.")
    logger.info("A malicious user can weaponize this to cause a ReDoS (Regular Expression Denial of Service).\n")

    # --- DEMO 3: The Fix ---
    logger.info("--- 3. THE FIX (Unrolled Loops) ---")
    logger.info(f"We rewrote the Regex to remove the nested quantifiers: r'^\\w+(?:\\s\\w+)*\\s?$'")
    
    massive_malformed_string = ("word " * SAFE_TEST_LENGTH) + "!"
    logger.info(f"Testing the SAFE regex against a massive string of {SAFE_TEST_LENGTH:,} words...")
    
    safe_time_massive = demo.measure_execution_time(demo.safe_regex, massive_malformed_string)
    
    logger.info(f"   Safe Regex Time: {safe_time_massive:.6f} seconds")
    logger.info("\n[Takeaway]: The optimized Regex fails almost instantly, even on a string 10,000x larger.")
    logger.info("It sees the '!', realizes there are no ambiguous paths, and immediately returns False.\n")


if __name__ == "__main__":
    run_lesson()