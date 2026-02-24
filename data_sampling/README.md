# Data Sampling & Corpus Curation

**The Core Question:** "How do you select your data?"

In an era of billion-scale datasets, we cannot label everything. The difference between a mediocre AI and a state-of-the-art model often isn't the architecture—it's the curation of the training data. This module explores the architecture of **Active Learning** and **Data Selection**, moving from naive random sampling to geometric information maximization.

## How to use this module
Run the scripts in order. Each script generates a pedagogical output log that demonstrates the specific failure state of that level and how the next technique solves it.

```bash
python data_sampling_1.py
# ... observe how Random Sampling misses the "Fraud" class.

python data_sampling_3_uncertainty.py
# ... observe how Entropy Sampling obsessively selects "Noise".

python data_sampling_5_hybrid.py
# ... observe how Hybrid Sampling filters the noise and keeps the hard data.
```

---

## The Sampling Hierarchy

We explore this domain across 5 distinct levels of complexity. Each level solves the specific failure state of the level before it.

### Level 1: The Unbiased Baseline
*   **File:** `data_sampling_1.py`
*   **Goal:** Create a statistically accurate snapshot of the world.
*   **Technique:** 
    * **Stratified Sampling**. We ensure that minority classes (e.g., "Fraud" at 1%) are forcibly represented in the sample to guarantee statistical significance.
*   **Trap:** 
    * **Pure Random Sampling**. In highly imbalanced datasets, random sampling often captures *zero* examples of critical edge cases.
*   **Best For:** 
    * **Test Sets & Final Evaluation**. (Never use Active Learning for your Test Set; it introduces bias).

```python
# The industry standard implementation
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

### Level 2: The "Cold Start" (Heuristics)
*   **File:** `data_sampling_2.py`
*   **Goal:** Bootstrap a V1 model when we have **zero** labeled data.
*   **Technique:** 
    * **Weak Supervision**. We write Labeling Functions (heuristics/regex) to auto-label obvious data and route conflicting rules to human annotators.
*   **Trap:** 
    * **The Echo Chamber**. The model only learns what we told it to look for. It misses "Unknown Unknowns" (intents we didn't write rules for).
*   **Best For:** 
    * Initial training data when launching a new product.

```python
# Regex-based heuristics acting as noisy labels
heuristic = lambda x: 'CANCEL' if re.search(r'cancel|stop', x) else None
```

### Level 3: The Boundary Learner (Uncertainty)
*   **File:** `data_sampling_3.py`
*   **Goal:** Refine the decision boundary of an existing model.
*   **Technique:** 
    * **Margin Sampling / Entropy**. We query the model for data where it is "confused" (Probability ~0.5).
*   **Trap:** 
    * **The Noise Trap (Aleatoric Uncertainty)**. Models are highly uncertain about valid hard data, but they are *also* highly uncertain about pure garbage (random noise). Uncertainty sampling wastes budget labeling garbage.
*   **Best For:** 
    * Fine-tuning a robust model that already understands the domain.

```python
# Sampling the items closest to the decision boundary (0.5)
dataset.sort(key=lambda x: abs(x.predict_proba() - 0.5))
```

### Level 4: The Coverage Guarantee (Diversity)
*   **File:** `data_sampling_4.py`
*   **Goal:** Find the "Unknown Unknowns" that Level 2 and 3 missed.
*   **Technique:** 
    * **Geometric Sampling**. We map data to Vector Space (Embeddings) and use **Clustering** or **Core-Set Selection** (Greedy Farthest Point) to ensure we sample from every corner of the topic map.
*   **Trap:** 
    * **Redundancy**. Geometric sampling ensures coverage, but it might pick "Easy" data that the model already knows perfectly well, just because it's in a sparse region.
*   **Best For:** 
    * Ensuring total topic coverage and finding outliers.

```python
# Clustering vectors to find representative centroids
kmeans = KMeans(n_clusters=k).fit(embeddings)
centroids = get_closest_samples_to_centers(kmeans.cluster_centers_)
```

### Level 5: Hybrid Active Learning
*   **File:** `data_sampling_5.py`
*   **Goal:** Maximize Information Gain per Dollar.
*   **Technique:** 
    * **Density-Weighted Uncertainty**. We combine Level 3 (Uncertainty) and Level 4 (Density). We pick data that is **Confusing** (Uncertain) but also **Representative** (Dense/Clustered).
*   **Trap:** 
    * **Complexity**. Requires maintaining an embedding index and a live model inference pipeline simultaneously.
*   **Best For:** 
    * Production Active Learning pipelines.

```python
# Combining signals to reject isolated noise
hybrid_score = (uncertainty_score ** alpha) * (density_score ** beta)
```
```