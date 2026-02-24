"""
Text Comparison Series: Level 5 - Semantic Search & Dense Embeddings

Focus:
Comparing the *Meaning* of text, rather than the spelling or keywords.
We map text into a dense vector space, calculate Cosine Similarity, 
solve the "Information Dilution" problem using Chunking, and combine 
Semantic and Lexical search using Reciprocal Rank Fusion (RRF).

Use Cases:
- Retrieval-Augmented Generation (RAG) for LLMs
- Question Answering systems ("How do I fix my screen?" -> "Display repair guide")
- Recommendation Engines
"""

import math
import logging
from typing import List, Dict, Tuple

# --- TUNABLE PARAMETERS ---
# Chunking configurations for long documents
CHUNK_SIZE_WORDS = 6
CHUNK_OVERLAP_WORDS = 2

# Reciprocal Rank Fusion constant (Standard industry value is 60)
RRF_K = 60  

# Configure logging for pedagogical output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Level5_Semantic")


class VectorMath:
    """Core mathematical operations for dense vectors (Embeddings)."""
    
    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    @staticmethod
    def magnitude(v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """
        Measures the angle between two vectors.
        1.0 = Identical direction (Same meaning)
        0.0 = Orthogonal (Unrelated)
        -1.0 = Opposite direction (Opposite meaning)
        """
        mag1 = VectorMath.magnitude(v1)
        mag2 = VectorMath.magnitude(v2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return VectorMath.dot_product(v1, v2) / (mag1 * mag2)

    @staticmethod
    def average_vectors(vectors: List[List[float]]) -> List[float]:
        """Averages a list of vectors into a single vector (Mean Pooling)."""
        if not vectors:
            return []
        dimensions = len(vectors[0])
        avg_vec = [0.0] * dimensions
        for v in vectors:
            for i in range(dimensions):
                avg_vec[i] += v[i]
        return [x / len(vectors) for x in avg_vec]


class MockTransformerEncoder:
    """
    Expert Technique: Dense Embeddings.
    In production, this is a Neural Network (BERT, Titan, OpenAI Ada) that 
    outputs 768+ dimensional floats. 
    
    For education, we use a 5-Dimensional Semantic Space:
    [Technology, Biology/Nature, Finance, Urgency/Negative, Exact_ID_Focus]
    """
    
    # Pre-calculated "Embeddings" for our tutorial
    KNOWLEDGE_BASE = {
        # Queries
        "computer broke":          [0.9, 0.0, 0.0, 0.8, 0.0],
        "how does a leaf eat":     [0.0, 0.9, 0.0, 0.1, 0.0],
        "apple iphone 15 pro max": [0.8, 0.0, 0.2, 0.0, 0.9],
        
        # Documents
        "laptop won't turn on":    [0.9, 0.0, 0.0, 0.7, 0.0], # High semantic match to "computer broke"
        "the broke guy bought a computer": [0.6, 0.0, 0.8, 0.4, 0.0], # Lexical trap
        "photosynthesis converts light into energy": [0.0, 0.9, 0.0, 0.0, 0.0],
        "stock market crashes today": [0.0, 0.0, 0.9, 0.9, 0.0],
        "apple iphone 14":         [0.8, 0.0, 0.2, 0.0, 0.4], # Semantically close, but wrong exact ID
    }

    @staticmethod
    def encode(text: str) -> List[float]:
        """Returns the dense vector for a string."""
        text = text.lower().strip()
        # Return known vectors, or a neutral fallback vector
        return MockTransformerEncoder.KNOWLEDGE_BASE.get(text, [0.1, 0.1, 0.1, 0.1, 0.1])


class DocumentChunker:
    """
    Expert Technique: Information Dilution Prevention.
    A single vector can only hold so much meaning. If we embed a 100-page 
    textbook into ONE vector, the specific chapters get diluted into a 
    generic 'average' vector. We must chunk the document.
    """
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
        """Splits text into overlapping sliding windows."""
        words = text.split()
        chunks = []
        if len(words) <= chunk_size:
            return [text]
            
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks


class HybridSearcher:
    """
    Expert Technique: Reciprocal Rank Fusion (RRF).
    Combines the exactness of Keyword Search (BM25) with the understanding 
    of Semantic Search (Vectors) without needing complex ML weighting.
    """
    
    @staticmethod
    def reciprocal_rank_fusion(lexical_ranks: List[str], semantic_ranks: List[str], k: int = RRF_K) -> List[Tuple[str, float]]:
        """
        RRF Formula: Score = 1 / (Rank + K).
        Documents that rank highly in BOTH lists get a massive score boost.
        """
        rrf_scores = {}
        
        # Process Lexical List
        for rank, doc_id in enumerate(lexical_ranks):
            # Rank is 0-indexed, so we use rank + 1
            score = 1.0 / ((rank + 1) + k)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
            
        # Process Semantic List
        for rank, doc_id in enumerate(semantic_ranks):
            score = 1.0 / ((rank + 1) + k)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
            
        # Sort by highest combined RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


# --- TUTORIAL DEMONSTRATIONS ---

def demo_1_the_lexical_gap():
    logger.info("=== DEMO 1: THE LEXICAL GAP (Why Semantic Wins) ===")
    
    query = "computer broke"
    doc_a = "laptop won't turn on"
    doc_b = "the broke guy bought a computer"
    
    logger.info(f"Query: '{query}'")
    logger.info(f"Doc A: '{doc_a}' (0 shared words)")
    logger.info(f"Doc B: '{doc_b}' (All query words shared)\n")
    
    # Simulate Lexical Search (Word Overlap)
    lexical_a = len(set(query.split()).intersection(set(doc_a.split())))
    lexical_b = len(set(query.split()).intersection(set(doc_b.split())))
    
    # Semantic Search
    vec_q = MockTransformerEncoder.encode(query)
    vec_a = MockTransformerEncoder.encode(doc_a)
    vec_b = MockTransformerEncoder.encode(doc_b)
    
    sem_a = VectorMath.cosine_similarity(vec_q, vec_a)
    sem_b = VectorMath.cosine_similarity(vec_q, vec_b)
    
    logger.info(f"Lexical Score (Word Match):  Doc A: {lexical_a} | Doc B: {lexical_b}   -> Lexical chooses B (Wrong!)")
    logger.info(f"Semantic Score (Cosine Sim): Doc A: {sem_a:.2f} | Doc B: {sem_b:.2f} -> Semantic chooses A (Correct!)")
    logger.info("\n[Takeaway]: Semantic Embeddings understand synonyms and intent. They map 'laptop' and 'computer' to the same geometric space.\n")


def demo_2_chunking_dilution():
    logger.info("=== DEMO 2: THE CHUNKING DILEMMA (Information Dilution) ===")
    
    query = "how does a leaf eat"
    
    # A bloated document that contains our target answer buried inside it
    bloated_text = "stock market crashes today . stock market crashes today . photosynthesis converts light into energy . stock market crashes today"
    
    logger.info(f"Query: '{query}'")
    logger.info(f"Bloated Document: '{bloated_text}'\n")
    
    # 1. Embed the whole document (Simulate by averaging the vectors of its sentences)
    sentences = [s.strip() for s in bloated_text.split('.') if s.strip()]
    sentence_vectors = [MockTransformerEncoder.encode(s) for s in sentences]
    
    whole_doc_vector = VectorMath.average_vectors(sentence_vectors)
    query_vector = MockTransformerEncoder.encode(query)
    
    whole_doc_sim = VectorMath.cosine_similarity(query_vector, whole_doc_vector)
    logger.info(f"1. Cosine Sim of Whole Document Vector: {whole_doc_sim:.2f}")
    
    # 2. Chunking the document
    logger.info("\n2. Applying Chunking Strategy...")
    best_chunk_sim = 0.0
    best_chunk_text = ""
    
    for sentence in sentences:
        chunk_vec = MockTransformerEncoder.encode(sentence)
        sim = VectorMath.cosine_similarity(query_vector, chunk_vec)
        logger.info(f"   Chunk: '{sentence[:20]}...' -> Sim: {sim:.2f}")
        
        if sim > best_chunk_sim:
            best_chunk_sim = sim
            best_chunk_text = sentence
            
    logger.info(f"\n[Takeaway]: If you embed a 500-page book, the specific chapter on Biology gets drowned out by the noise (Score: {whole_doc_sim:.2f}). By chunking, we isolate the exact semantic meaning (Score: {best_chunk_sim:.2f}).\n")


def demo_3_hybrid_search_rrf():
    logger.info("=== DEMO 3: HYBRID SEARCH (Reciprocal Rank Fusion) ===")
    
    query = "apple iphone 15 pro max"
    logger.info(f"Query: '{query}'")
    
    # Scenario: 
    # Lexical is great at Exact IDs (15 pro max).
    # Semantic understands "Apple" and "iPhone" generally, but gets fuzzy on version numbers.
    
    # Mock Ranks returned by our two independent search systems
    # Order represents Rank 1, Rank 2, Rank 3...
    lexical_rankings = [
        "apple iphone 15 pro max",   # Exact match wins Lexical
        "iphone 15 case",            # Lexical trap (matches 'iphone' and '15')
        "apple iphone 14"            # Misses '15', ranks lower
    ]
    
    semantic_rankings = [
        "apple iphone 14",           # Semantic loves this (it's a phone)
        "apple iphone 15 pro max",   # Semantic also loves this (it's a phone)
        "iphone 15 case"             # Semantic knows a case is an accessory, drops it
    ]
    
    logger.info("\nLexical Rankings (BM25):")
    for i, doc in enumerate(lexical_rankings): logger.info(f"  {i+1}. {doc}")
        
    logger.info("\nSemantic Rankings (Vector Cosine):")
    for i, doc in enumerate(semantic_rankings): logger.info(f"  {i+1}. {doc}")
        
    logger.info(f"\nApplying Reciprocal Rank Fusion (K={RRF_K})...")
    final_hybrid_rankings = HybridSearcher.reciprocal_rank_fusion(lexical_rankings, semantic_rankings)
    
    for rank, (doc_id, score) in enumerate(final_hybrid_rankings):
        logger.info(f"  {rank+1}. {doc_id:<25} (RRF Score: {score:.5f})")
        
    logger.info("\n[Takeaway]: The 'iPhone 15 Case' tricked Lexical. The 'iPhone 14' tricked Semantic. But the exact desired product ('iPhone 15 Pro Max') ranked highly in BOTH, so RRF mathematically forces it to the absolute top position.")

if __name__ == "__main__":
    demo_1_the_lexical_gap()
    demo_2_chunking_dilution()
    demo_3_hybrid_search_rrf()