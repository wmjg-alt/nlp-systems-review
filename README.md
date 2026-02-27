# Applied NLP & Data Engineering Systems

A practical, code-first study guide and architectural reference for building scalable data pipelines and Information Retrieval (IR) systems. 

We hope to bridge the gap between theoretical algorithms and production-grade system design. Rather than relying on black-box libraries, these modules implement core algorithmic examples from scratch to demonstrate their mathematical TRADE-OFFS, EDGE CASES, and SCALING LIMITATIONS under extreme data constraints.

## Design Philosophy
* **Pedagogical Output**: Every script is instrumented with extensive logging to demonstrate the "Why" and "How" of algorithm behavior, forcing edge cases and failures to highlight system boundaries.
* **Standard Library First**: Logic is implemented in raw Python/NumPy whenever possible to expose the underlying mechanics (e.g., building an Inverted Index or LSH Bucket system from scratch).
* **Resource Awareness**: Every module considers memory constraints ($O(1)$ space) and computational complexity ($O(N)$ vs $O(N^2)$) as high priorities.

---

## Repository Structure
```text
├── text_comparison/    # Identity, Fuzzy, Syntactic, Lexical, and Semantic matching
├── data_sampling/      # Stratification, Weak Supervision, and Active Learning
├── data_analytics/     # ETL, Window Functions, and Strategic Visualization
├── mini_lessons/       # Specialized deep-dives (Regex ReDoS, Clustering)
├── metrics/            # (WIP): Evaluation frameworks (Precision/Recall, Kappa, etc.)
├── sharding/           # (WIP): Map-Reduce shards for Big Data
└── python_practice/    # (WIP): Performance engineering (GIL, Generators, more)
```

---

## Module 1: Big Data Architecture (`/sharding`)
**The Question:** "How do you process data larger than RAM?"

* **Consistent Hashing:** Implementing deterministic partitioning to group data without $O(N^2)$ overhead.
* **Heuristic Filtering:** Demonstrating the "Map Phase" trick of string-matching before JSON parsing to save hours of CPU time.
* **Use Case:** Forensic analysis of 80TB security logs to identify credential-stuffing IPs on a single 64GB machine.

---

## Module 2: The Text Comparison Spectrum (`/text_comparison`)
**The Question:** "How do you define and measure similarity?"

### Level 1: Exact Identity & Bloom Filters
* **Use Case:** Checking if a URL has been crawled before or verifying file integrity (checksums).
### Level 2: Fuzzy Matching & Metric Search
* **Use Case:** Search bar autocorrect ("iphne" ➔ "iphone") and resolving duplicate "John Smith" records in a CRM.
### Level 3: Syntactic Similarity & LSH
* **Use Case:** Detecting plagiarism in long-form articles or deduplicating massive web-scraped datasets (CommonCrawl).
### Level 4: Lexical Search (IR) & BM25
* **Use Case:** E-commerce product search and technical log retrieval where exact part numbers or error codes matter.
### Level 5: Semantic Search & Hybrid RRF
* **Use Case:** Building a Retrieval-Augmented Generation (RAG) system that understands intent ("computer broke" ➔ "laptop repair guide").

---

## Module 3: Corpus Curation & Data Sampling (`/data_sampling`)
**The Question:** "How do you select high-value data for human annotation?"

### Level 1: Stratified Baselines
* **Use Case:** Creating a statistically honest "Golden Test Set" that doesn't ignore rare 1% classes like "Fraud."
### Level 2: Weak Supervision (Heuristics)
* **Use Case:** "Cold-starting" a new feature (e.g., "Cancel Subscription") with 10 million raw logs and zero labels.
### Level 3: Uncertainty Sampling
* **Use Case:** Identifying the "Decision Boundary" to efficiently teach a model the difference between "Downgrade" and "Cancel."
### Level 4: Geometric Diversity (Core-Set)
* **Use Case:** Discovering "Unknown Unknowns" by sampling from every semantic neighborhood in a dataset.
### Level 5: Hybrid Active Learning
* **Use Case:** Production-grade sampling that picks "Confusing" data while automatically rejecting unlearnable "Noise."

---

## Module 4: Data Analytics & Strategic Insights (`/data_analytics`)
**The Question:** "Can you find the story in the data?"

* **Step 1: Synthetic Sabotage:** Programmatically injecting "Real-World Mess" (Nested JSON, Unicode noise, date chaos).
* **Step 2: The Data Janitor:** Professional ETL patterns using `pd.json_normalize` and schema coalescing.
* **Step 3: Window Functions:** Using `.transform()` for category-relative Z-scores to find "Hidden Gems."
* **Step 4: Applied Science:** Time-series resampling and "Rating Decay" detection.
* **Step 5: Visual Storytelling:** Using Faceting and KDE distributions to justify business decisions.

---

*More modules will come...*