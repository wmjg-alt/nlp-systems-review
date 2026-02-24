# Applied Data Engineering & NLP Systems

A practical, code-first study guide and architectural reference for building scalable data pipelines and Information Retrieval (IR) systems. 

This repository bridges the gap between theoretical algorithms and production-grade system design. Rather than relying on black-box libraries, these modules implement core algorithms from scratch to demonstrate their mathematical trade-offs, edge cases, and scaling limitations under extreme data constraints.

## Design Philosophy
* **Pedagogical Output**: Every script is designed to be run. Extensive logging explicitly demonstrates the "Why" and "How" of algorithm behavior, forcing edge cases and failures to highlight system boundaries.
* **Standard Library First**: Core logic is built using Python's standard library whenever possible to expose the underlying mechanics (e.g., building an Inverted Index or LSH Bucket system from scratch).
* **Modular Architecture**: Concepts are isolated into distinct domains and directories, making it easy to study individual paradigms.

---

## Repository Structure
```text
├── sharding/
│   ├── log_sharder.py              # Core Logic: Map-Reduce sharding algorithm
│   └── shard_log_generator.py      # Utility: Generates noisy log data with planted signals
└── text_comparison/
    ├── generate_level3_docs.py     # Utility: Builds a little test corpus
    ├── text_comparisons_1.py       # Level 1: Exact Identity & Bloom Filters
    ├── text_comparisons_2.py       # Level 2: Fuzzy Matching & Metric Search
    ├── text_comparisons_3.py       # Level 3: Syntactic Similarity & LSH
    ├── text_comparisons_4.py       # Level 4: Lexical Search (IR) & BM25
    └── text_comparisons_5.py       # Level 5: Semantic Vectors & Hybrid Search
```

---

## Module 1: Big Data Processing & Sharding (`/sharding`)
**Focus:** Processing massive datasets (e.g., 80TB log files) on strictly memory-constrained environments (e.g., 64GB RAM) without distributed frameworks like Spark.

* **Algorithmic Sharding:** (`log_sharder.py`) Implementing consistent hashing to partition data deterministically.
* **Map-Reduce Paradigms:** Demonstrating the critical importance of early heuristic filtering (Map phase) to prevent disk I/O bottlenecks.
* **Resource Safety:** Dynamic shard calculation based on available RAM and OS file-handle limits (`ulimit`).
* **Validation:** Use `shard_log_generator.py` to plant "bad actors" in a sea of noise, then verify the sharder's output against `ground_truth.txt`.

---

## Module 2: The Text Comparison Spectrum (`/text_comparison`)
**Focus:** A five-level progression of comparing text, scaling from byte-level cryptographic identity to multi-dimensional semantic meaning. 

### Level 1: Exact Identity & Probabilistic Scaling (`text_comparisons_1.py`)
* **Concepts:** SHA-256 Hashing, Normalization strictness, Hash Collisions (The Pigeonhole Principle).
* **Scale Architecture:** Implementing **Bloom Filters** to perform O(1) membership checks on billions of records with fixed memory footprints, demonstrating the False Positive trade-off.

### Level 2: Fuzzy Matching & Metric Search (`text_comparisons_2.py`)
* **Concepts:** Edit Distances (Levenshtein vs. Damerau-Levenshtein), Transposition handling, Prefix-biased similarity (Jaro-Winkler), and Morphological Stemming.
* **Scale Architecture:** Transitioning from O(N) linear scans to O(1) and O(log N) metric spaces using **SymSpell** (Symmetric Delete Indexing) and **BK-Trees**.

### Level 3: Syntactic Similarity (`text_comparisons_3.py`)
* **Concepts:** N-Gram Shingling, Jaccard Similarity, and the Subset/Plagiarism problem (Containment vs. Jaccard).
* **Scale Architecture:** Compressing massive documents into fixed-size integer signatures using **MinHash**, and completely avoiding O(N²) all-to-all comparisons using **Locality Sensitive Hashing (LSH)** via banding. *(Run `generate_level3_docs.py` first to populate the `docs/` test corpus).*

### Level 4: Lexical Search / Information Retrieval (`text_comparisons_4.py`)
* **Concepts:** Term Frequency vs. Inverse Document Frequency, Query Expansion (The Vocabulary Gap), and Keyword Saturation.
* **Scale Architecture:** Building an **Inverted Index**, upgrading to a **Positional Index** to support exact-phrase matching, and mathematically proving why **BM25** outperforms raw TF-IDF for length normalization.

### Level 5: Semantic Search / Dense Embeddings (`text_comparisons_5.py`)
* **Concepts:** Information Dilution, Chunking strategies, Dense Vectors, and Cosine Similarity.
* **Scale Architecture:** Mapping out a mock Vector Database (ANN/HNSW concepts) and solving the Semantic/Lexical divide by building a Hybrid Search pipeline fused with **Reciprocal Rank Fusion (RRF)**.

---

*More modules will be appended as this repository expands.*
