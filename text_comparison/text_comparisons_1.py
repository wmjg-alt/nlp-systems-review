"""
Text Comparison Series: Level 1 - Exact Identity & Probabilistic Membership

Focus: 
Determining if two items are bit-for-bit identical, how to loosen that 
identity with Normalization, and how to scale identity checks to billions 
of records using Bloom Filters. Finally, we prove the mathematical limits 
of hashing via the Pigeonhole Principle.

Use Cases:
- Password authentication
- Web crawler deduplication (Have I visited this URL?)
- Data integrity checks (File checksums)
- Caching systems
"""

import hashlib
import logging
import random
import re
import string
from typing import List

# --- TUNABLE PARAMETERS ---
# Controls the strictness of basic identity checks
DEFAULT_ENCODING = "utf-8"

# Bloom Filter configurations
BLOOM_CAPACITY = 500        # Size of the bit array (Memory footprint)
BLOOM_HASH_COUNT = 3        # Number of hash functions to run per item

# Hash Collision Demonstration configurations
COLLISION_SPACE_SIZE = 256  # 8-bit space (0-255) to guarantee a collision

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level1_Identity")


class TextNormalizer:
    """
    Expert Technique: Preprocessing.
    Exact matching fails on invisible trailing spaces or unexpected capitalization.
    Normalization bridges the gap between 'Binary Identity' and 'Human Identity'.
    """

    @staticmethod
    def clean(text: str, strict: bool = False) -> str:
        """
        If strict=True, returns the exact raw text.
        If strict=False, removes casing, punctuation, and excess whitespace.
        """
        if strict:
            return text
        
        # 1. Lowercase for case-insensitivity
        text = text.lower()
        # 2. Remove all punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)
        # 3. Collapse multiple spaces into a single space and trim edges
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class IdentityRegistry:
    """
    Stores normalized fingerprints for O(1) identity checks.
    
    Why hash instead of storing strings?
    Storing a 10,000-word article takes ~50KB. 
    Storing its SHA-256 fingerprint takes exactly 64 characters (bytes).
    This keeps RAM usage predictable.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self._fingerprints = set()

    def register(self, text: str):
        """Normalizes, hashes, and stores the text."""
        cleaned = TextNormalizer.clean(text, self.strict)
        fingerprint = hashlib.sha256(cleaned.encode(DEFAULT_ENCODING)).hexdigest()
        self._fingerprints.add(fingerprint)

    def check_exists(self, text: str) -> bool:
        """Checks if the text's fingerprint is in the registry."""
        cleaned = TextNormalizer.clean(text, self.strict)
        fingerprint = hashlib.sha256(cleaned.encode(DEFAULT_ENCODING)).hexdigest()
        return fingerprint in self._fingerprints


class BloomFilter:
    """
    Expert Technique: Probabilistic Scaling.
    
    When a Set becomes too large for RAM (e.g., 10 Billion URLs), we use a Bloom Filter.
    Trade-off:
    - Zero False Negatives (If it says 'No', it is definitely not there).
    - Possible False Positives (If it says 'Yes', it MIGHT be there).
    """

    def __init__(self, capacity: int, hash_count: int):
        self.capacity = capacity
        self.hash_count = hash_count
        self.bit_array = [0] * capacity

    def _get_indices(self, text: str) -> List[int]:
        """
        Generates multiple array indices using salted MD5 hashes.
        Simulates having 'hash_count' independent hash functions.
        """
        indices = []
        for i in range(self.hash_count):
            # Salt the string with the current index 'i'
            seed = f"{i}_{text}".encode(DEFAULT_ENCODING)
            hash_digest = hashlib.md5(seed).hexdigest()
            # Convert hex to int and modulo by capacity to map to the array
            indices.append(int(hash_digest, 16) % self.capacity)
        return indices

    def add(self, text: str):
        """Sets the bits at all generated indices to 1."""
        for idx in self._get_indices(text):
            self.bit_array[idx] = 1

    def exists(self, text: str) -> bool:
        """
        Checks if ALL bits at the generated indices are 1.
        If even one bit is 0, the item was never added.
        """
        for idx in self._get_indices(text):
            if self.bit_array[idx] == 0:
                return False
        return True


class WeakHashCollisionDemo:
    """
    Expert Technique: The Pigeonhole Principle.
    To prove that hashes CAN collide (and why we use 256-bit hashes in production),
    we use a weak 8-bit hash. If we hash 257 distinct strings into 256 buckets, 
    math guarantees at least one collision.
    """

    def __init__(self, space_size: int):
        self.space_size = space_size
        self.hash_map = {}  # Maps hash_value -> List of original strings

    def weak_hash(self, text: str) -> int:
        """A terrible hash function: sum of ascii values modulo space_size."""
        return sum(ord(c) for c in text) % self.space_size

    def add_and_check_collision(self, text: str) -> bool:
        """Returns True if a collision occurred."""
        h = self.weak_hash(text)
        
        if h in self.hash_map:
            # Hash exists. Are the strings actually different?
            if text not in self.hash_map[h]:
                self.hash_map[h].append(text)
                return True # Verified Hash Collision!
            return False # We just added the exact same string twice
        else:
            self.hash_map[h] = [text]
            return False


# --- DEMONSTRATIONS & TUTORIALS ---

def demo_exact_and_normalized():
    logger.info("=== DEMO 1: EXACT MATCHING & NORMALIZATION ===")
    
    raw_1 = "The quick brown fox."
    raw_2 = "  the QUICK brown fox  "
    raw_3 = "The quick brown fox." # Exact match
    
    strict_registry = IdentityRegistry(strict=True)
    loose_registry = IdentityRegistry(strict=False)
    
    # Register the baseline string
    strict_registry.register(raw_1)
    loose_registry.register(raw_1)
    
    logger.info(f"Baseline String: '{raw_1}'")
    logger.info(f"Test String 1:   '{raw_3}' (Exact)")
    logger.info(f"Test String 2:   '{raw_2}' (Messy formatting)\n")
    
    logger.info("-> Testing Strict Identity (No Normalization):")
    logger.info(f"   Matches Test 1? {strict_registry.check_exists(raw_3)}")
    logger.info(f"   Matches Test 2? {strict_registry.check_exists(raw_2)}")
    
    logger.info("\n-> Testing Loose Identity (Normalized):")
    logger.info(f"   Matches Test 1? {loose_registry.check_exists(raw_3)}")
    logger.info(f"   Matches Test 2? {loose_registry.check_exists(raw_2)}")
    logger.info("\n[Takeaway]: Level 1 is unforgiving. Always normalize text before hashing unless you are auditing exact file integrity.\n")


def demo_bloom_filter_scale():
    logger.info("=== DEMO 2: BLOOM FILTERS & PROBABILISTIC SCALING ===")
    logger.info(f"Initializing Bloom Filter with {BLOOM_CAPACITY} bits and {BLOOM_HASH_COUNT} hashes per item.")
    
    bf = BloomFilter(capacity=BLOOM_CAPACITY, hash_count=BLOOM_HASH_COUNT)
    
    # Simulate adding valid data
    logger.info("Adding 50 valid URLs to the filter...")
    for i in range(50):
        bf.add(f"https://example.com/page_{i}")
        
    logger.info(f"Checking existing URL (page_42): {bf.exists('https://example.com/page_42')} (Expected: True)")
    logger.info(f"Checking unseen URL (page_99):   {bf.exists('https://example.com/page_99')} (Expected: False)")
    
    # Forcing a False Positive by saturating the filter
    logger.info("\n-> Forcing a False Positive...")
    logger.info(f"Saturating filter by adding 1,000 more items to a {BLOOM_CAPACITY}-bit array...")
    for i in range(1000):
        bf.add(f"noise_data_{i}")
        
    random_junk = "phantom_ghost_url_that_was_never_added"
    is_present = bf.exists(random_junk)
    
    logger.info(f"Checking unseen junk string: {is_present}")
    if is_present:
        logger.info("\n[Takeaway]: False Positive detected! Because the bit array filled up with 1s, the filter mathematically guessed 'Yes'. This is the RAM-saving trade-off of Bloom Filters.\n")


def demo_hash_collisions():
    logger.info("=== DEMO 3: THE PIGEONHOLE PRINCIPLE ===")
    logger.info(f"Using a weak 8-bit hash (Capacity: {COLLISION_SPACE_SIZE} buckets).")
    
    demo = WeakHashCollisionDemo(space_size=COLLISION_SPACE_SIZE)
    attempts = 0
    
    logger.info("Generating random strings until a hash collision occurs...")
    
    # We loop until we guarantee a collision (Space Size + 1)
    while attempts <= COLLISION_SPACE_SIZE + 1:
        # Generate a random 10-character string
        random_str = ''.join(random.choices(string.ascii_letters, k=10))
        attempts += 1
        
        if demo.add_and_check_collision(random_str):
            h = demo.weak_hash(random_str)
            colliding_strings = demo.hash_map[h]
            logger.info(f"💥 COLLISION FOUND on attempt {attempts}!")
            logger.info(f"Hash Bucket [{h}] represents multiple unique strings: {colliding_strings}")
            break
            
    logger.info("\n[Takeaway]: Hashes DO collide. We use SHA-256 in production because having 2^256 buckets makes a collision mathematically certain at infinite scale, but practically impossible in our lifetimes.\n")


if __name__ == "__main__":
    # Execute all learning modules
    demo_exact_and_normalized()
    demo_bloom_filter_scale()
    demo_hash_collisions()