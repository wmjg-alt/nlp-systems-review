# Text Comparison & Document Similarity

**The Core Question:** "How do you compare your text?"

The method used to compare two strings / documents depends entirely on the domain, the length of the text, and the computational budget. Comparing a password requires bit-level rigidity; comparing a user's search intent requires multi-dimensional semantic mapping. This module explores the hierarchy of text comparison, from exact cryptographic identity to modern vector-based semantic search.

## How to use this module
Run the scripts in order. Each script includes basic and advanced implementations, highlighting the trade-offs between precision, speed, and scale.

```bash
# Note: Level 3 requires the docs directory to be populated first
python generate_level3_docs.py

python text_comparisons_1.py
# ... observe the Pigeonhole Principle and Bloom Filter False Positives.

python text_comparisons_3.py
# ... observe how MinHash approximates Jaccard Similarity.

python text_comparisons_5.py
# ... observe how Reciprocal Rank Fusion (RRF) merges search results.
```

---


## The Comparison Hierarchy

We explore the spectrum of similarity across 5 levels, moving from purely syntactic matches to abstract semantic understanding.

### Level 1: Exact Identity & Probabilistic Membership
*   **File:** `text_comparisons_1.py`
*   **Goal:** Determine if two items are bit-for-bit identical.
*   **Technique:** 
    * **Cryptographic Hashing & Bloom Filters**. We use SHA-256 for fingerprints and Bloom Filters for O(1) membership checks on massive datasets with a fixed memory footprint.
*   **Trap:** 
    * **Sensitivity**. Invisible whitespace or case differences result in a total mismatch. Requires robust normalization to be useful for human-generated text.
*   **Best For:** 
    * Passwords, file integrity (checksums), and massive URL "seen" lists for web crawlers.

```python
# Bit-level identity fingerprint
fingerprint = hashlib.sha256(text.encode()).hexdigest()
```

### Level 2: Fuzzy Matching & Metric Search
*   **File:** `text_comparisons_2.py`
*   **Goal:** Quantify similarity for short strings with typos or variations.
*   **Technique:** 
    * **Edit Distance (Levenshtein/Damerau) & BK-Trees**. We measure the "cost" of transforming one string to another and use metric trees to scale fuzzy lookups.
*   **Trap:** 
    * **Computational Complexity**. Standard Edit Distance is $O(N \times M)$, making it unusable for long documents or massive dictionary scans without indexing.
*   **Best For:** 
    * Search autocorrect, entity resolution (merging "Jon Smith" and "John Smith"), and OCR correction.

```python
# Measuring edit distance between two short strings
distance = levenshtein_distance(query, target)
```

### Level 3: Syntactic Similarity (Near-Duplicates)
*   **File:** `text_comparisons_3.py`
*   **Goal:** Detect copy-pasted or slightly edited long-form documents.
*   **Technique:** 
    * **MinHash & Locality Sensitive Hashing (LSH)**. We shingle text into N-Grams and compress documents into fixed-size integer signatures to avoid $O(N^2)$ all-to-all comparisons.
*   **Trap:** 
    * **The Subset Problem**. Jaccard Similarity fails when one document is much larger than the other (e.g., a quote inside a book). Requires **Containment** math to solve.
*   **Best For:** 
    * Plagiarism detection and deduplicating massive web-scraped training sets (e.g., CommonCrawl).

```python
# Creating a fixed-size signature that preserves Jaccard Similarity
signature = minhash_generator.compute(shingles)
```

### Level 4: Lexical Search (Information Retrieval)
*   **File:** `text_comparisons_4.py`
*   **Goal:** Rank documents based on keyword relevance to a short query.
*   **Technique:** 
    * **Inverted Indexing & BM25**. We map words to document locations and use BM25 to apply "mathematical brakes" to keyword spamming and document length bias.
*   **Trap:** 
    * **The Vocabulary Gap**. A search for "sneakers" will fail if the document only uses the word "shoes." Requires Query Expansion or Lemmatization.
*   **Best For:** 
    * E-commerce search engines and log debugging systems.

```python
# The industry standard for keyword relevance ranking
score = bm25_scorer.get_score(query_terms, doc_id)
```

### Level 5: Semantic Search (Dense Embeddings)
*   **File:** `text_comparisons_5.py`
*   **Goal:** Compare text based on intent and meaning rather than keywords.
*   **Technique:** 
    * **Dense Vectors & Hybrid Search (RRF)**. We map text into a multi-dimensional geometry and use **Reciprocal Rank Fusion** to combine the precision of BM25 with the intuition of AI.
*   **Trap:** 
    * **Information Dilution**. Embedding a massive document into a single vector "averages out" the meaning. Requires **Chunking** to maintain granularity.
*   **Best For:** 
    * Retrieval-Augmented Generation (RAG) and recommendation engines.

```python
# Semantic similarity via vector geometry
similarity = cosine_similarity(query_vector, document_vector)
```
