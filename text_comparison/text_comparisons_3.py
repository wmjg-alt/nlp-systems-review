"""
Text Comparison Series: Level 3 - Syntactic Similarity & LSH

Focus: 
Comparing large documents (articles, books) where Edit Distance (Level 2) 
is computationally impossible (O(N^2) on massive strings).
We use Shingling to preserve word order, Jaccard to measure overlap, 
MinHash to compress the documents into O(1) vectors, and LSH to bucket 
similar documents to avoid comparing every document to every other document.

Use Cases:
- Deduplicating training data for Large Language Models (e.g., CommonCrawl)
- Plagiarism detection
- News article clustering (grouping syndicated articles)
"""

import hashlib
import logging
from pathlib import Path
from typing import Set, List, Dict, Tuple

# --- TUNABLE PARAMETERS ---
SHINGLE_SIZE = 3           # 'k' in k-grams. 3 words preserves phrase structure.
MINHASH_PERMUTATIONS = 100 # Size of the MinHash signature (Number of hash functions).
LSH_BANDS = 20             # Number of LSH bands (Must divide MINHASH_PERMUTATIONS evenly).
LSH_ROWS = 5               # Rows per band (BANDS * ROWS = PERMUTATIONS).

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level3_Syntactic")


class DocumentProcessor:
    """Handles text tokenization and N-Gram/Shingle generation."""
    
    @staticmethod
    def get_shingles(text: str, k: int = SHINGLE_SIZE) -> Set[str]:
        """
        Converts text into a set of k-word overlapping phrases.
        Why?
        If k=1 (Bag of Words): "Dog bites man" and "Man bites dog" are 100% identical.
        If k=3: "Dog bites man" vs "Man bites dog" share 0 shingles. Order is preserved.
        """
        # Simplistic tokenization (lowercase, split by whitespace)
        words = text.lower().replace('.', '').replace(',', '').split()
        shingles = set()
        
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i : i + k])
            shingles.add(shingle)
            
        return shingles


class SyntacticMetrics:
    """Core mathematical comparisons for sets of shingles."""
    
    @staticmethod
    def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """
        Intersection over Union. 
        Best for finding documents of roughly equal size that are nearly identical.
        """
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def containment(set_a: Set[str], set_b: Set[str]) -> float:
        """
        Intersection over the size of the *smaller* set.
        Best for finding Subsets (e.g., Plagiarism, or an article inside a daily digest).
        """
        intersection = len(set_a.intersection(set_b))
        min_len = min(len(set_a), len(set_b))
        return intersection / min_len if min_len > 0 else 0.0


class MinHash:
    """
    Compresses a massive set of strings into a tiny, fixed-size array of integers.
    Mathematical Guarantee: The probability that MinHash(A)[i] == MinHash(B)[i] 
    is exactly equal to the Jaccard Similarity of A and B.
    """
    
    def __init__(self, num_permutations: int = MINHASH_PERMUTATIONS):
        self.num_permutations = num_permutations

    def compute_signature(self, shingles: Set[str]) -> List[int]:
        """Generates the signature. Time Complexity: O(Shingles * Permutations)."""
        signature = []
        
        for i in range(self.num_permutations):
            min_hash_val = float('inf')
            
            for shingle in shingles:
                # We simulate multiple hash functions by salting with the index 'i'
                seed = f"{i}_{shingle}".encode('utf-8')
                # SHA-1 is used here for deterministic cross-platform hashing, 
                # though MurmurHash3 is faster in production.
                hash_val = int(hashlib.sha1(seed).hexdigest(), 16)
                
                if hash_val < min_hash_val:
                    min_hash_val = hash_val
                    
            signature.append(min_hash_val)
            
        return signature

    @staticmethod
    def estimate_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimates Jaccard similarity by counting matching integers in the signature."""
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)


class LSHIndex:
    """
    Locality Sensitive Hashing (LSH) via Banding.
    Solves the O(N^2) comparison problem. If we have 1 Million documents, we cannot 
    compare 1 Trillion signatures. LSH buckets similar signatures together in O(1) time.
    """
    
    def __init__(self, bands: int = LSH_BANDS, rows: int = LSH_ROWS):
        self.bands = bands
        self.rows = rows
        # A list of dictionaries. One dictionary (hash map) per band.
        self.buckets: List[Dict[int, List[str]]] = [{} for _ in range(bands)]
        
    def insert(self, doc_id: str, signature: List[int]):
        """Hashes segments (bands) of the signature. Drops the doc_id into buckets."""
        if len(signature) != self.bands * self.rows:
            raise ValueError("Signature length must equal bands * rows.")
            
        for band_idx in range(self.bands):
            # Extract the subset of the signature for this band
            start = band_idx * self.rows
            end = start + self.rows
            band_segment = tuple(signature[start:end])
            
            # Hash the tuple to get a bucket ID
            bucket_id = hash(band_segment)
            
            if bucket_id not in self.buckets[band_idx]:
                self.buckets[band_idx][bucket_id] = []
            self.buckets[band_idx][bucket_id].append(doc_id)

    def get_candidates(self) -> Set[Tuple[str, str]]:
        """
        Returns pairs of doc_ids that share AT LEAST ONE bucket.
        These are the 'Candidate Pairs' that we will actually verify with exact Jaccard.
        """
        candidates = set()
        for band_buckets in self.buckets:
            for bucket_id, doc_ids in band_buckets.items():
                if len(doc_ids) > 1:
                    # Generate all pairs within this bucket
                    for i in range(len(doc_ids)):
                        for j in range(i + 1, len(doc_ids)):
                            # Sort to avoid (A,B) and (B,A) duplicates
                            pair = tuple(sorted([doc_ids[i], doc_ids[j]]))
                            candidates.add(pair)
        return candidates


# --- TUTORIAL DEMONSTRATIONS ---

def load_docs() -> Dict[str, str]:
    docs_dir = Path("docs")
    if not docs_dir.exists():
        logger.error("Docs directory not found. Please run generate_level3_docs.py first.")
        exit(1)
        
    return {f.stem: f.read_text() for f in docs_dir.glob("*.txt")}


def demo_1_jaccard_vs_containment(docs: Dict[str, str]):
    logger.info("\n=== DEMO 1: JACCARD VS CONTAINMENT (THE SUBSET PROBLEM) ===")
    
    shingles_a = DocumentProcessor.get_shingles(docs["doc_A_original"])
    shingles_c = DocumentProcessor.get_shingles(docs["doc_C_subset"])
    shingles_d = DocumentProcessor.get_shingles(docs["doc_D_aggregate"])
    
    logger.info("Comparing Doc A (Original) to Doc C (Subset/First Sentence):")
    logger.info(f"   Jaccard Similarity: {SyntacticMetrics.jaccard_similarity(shingles_a, shingles_c):.3f}")
    logger.info(f"   Containment:        {SyntacticMetrics.containment(shingles_a, shingles_c):.3f}")
    
    logger.info("\nComparing Doc A (Original) to Doc D (Aggregate Digest):")
    logger.info(f"   Jaccard Similarity: {SyntacticMetrics.jaccard_similarity(shingles_a, shingles_d):.3f}")
    logger.info(f"   Containment:        {SyntacticMetrics.containment(shingles_a, shingles_d):.3f}")
    
    logger.info("\n[Takeaway]: Jaccard penalizes size differences heavily. If you are looking for plagiarism (a subset), you MUST use Containment. Jaccard will miss it.\n")


def demo_2_minhash_compression(docs: Dict[str, str]):
    logger.info("=== DEMO 2: MINHASH COMPRESSION ===")
    
    shingles_a = DocumentProcessor.get_shingles(docs["doc_A_original"])
    shingles_b = DocumentProcessor.get_shingles(docs["doc_B_minor_edits"])
    
    exact_jaccard = SyntacticMetrics.jaccard_similarity(shingles_a, shingles_b)
    
    logger.info("Computing MinHash signatures (Compressing text to 100 integers)...")
    mh = MinHash(num_permutations=MINHASH_PERMUTATIONS)
    sig_a = mh.compute_signature(shingles_a)
    sig_b = mh.compute_signature(shingles_b)
    
    est_jaccard = MinHash.estimate_jaccard(sig_a, sig_b)
    
    logger.info(f"   Exact Jaccard (Heavy Set Math): {exact_jaccard:.3f}")
    logger.info(f"   Estimated Jaccard (MinHash):    {est_jaccard:.3f}")
    logger.info("\n[Takeaway]: MinHash perfectly approximates Jaccard Similarity, but turns variable-length string comparisons into fast, fixed-size integer array comparisons.\n")


def demo_3_lsh_scaling(docs: Dict[str, str]):
    logger.info("=== DEMO 3: LSH & AVOIDING O(N^2) COMPARISONS ===")
    
    mh = MinHash(num_permutations=MINHASH_PERMUTATIONS)
    lsh = LSHIndex(bands=LSH_BANDS, rows=LSH_ROWS)
    
    logger.info(f"Hashing {len(docs)} documents into LSH Buckets (Bands: {LSH_BANDS}, Rows: {LSH_ROWS})...")
    
    # Process all documents
    for name, text in docs.items():
        shingles = DocumentProcessor.get_shingles(text)
        signature = mh.compute_signature(shingles)
        lsh.insert(name, signature)
        
    candidates = lsh.get_candidates()
    
    logger.info(f"\nLSH identified {len(candidates)} candidate pairs that warrant full comparison:")
    for pair in candidates:
        logger.info(f"   -> {pair[0]} & {pair[1]}")
        
    logger.info("\nNotice who is missing:")
    logger.info("   - Doc E (Unrelated) never matched with anything.")
    logger.info("   - Doc C (Subset) did not match Doc A because LSH optimizes for Jaccard (Similarity), not Containment.")
    
    logger.info("\n[Takeaway]: Out of 10 possible combinations (5 docs = 5*4/2), LSH only flagged the actual duplicates for review. At a scale of 1 Billion docs, this saves trillions of wasted comparisons.\n")


if __name__ == "__main__":
    test_docs = load_docs()
    demo_1_jaccard_vs_containment(test_docs)
    demo_2_minhash_compression(test_docs)
    demo_3_lsh_scaling(test_docs)