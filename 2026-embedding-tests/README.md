# Embedding Analysis Tests (2026)

This directory contains experiments focused on evaluating and improving the semantic embedding pipeline used in the *Processus Universalis* analysis. The primary goal was to overcome the limitations of the original "Integrated Pipeline" by investigating how different aggregation methods affect the detection of procedural and thematic relationships.

### Research Steps and Methodology

1.  **From Global Averages to Local Alignment**
    The original analysis used **Mean-Pooling**, which reduces an entire document to a single centroid vector. While intuitive, this "average vibe" approach often masks the specific procedural steps that define alchemical traditions. We shifted focus to **alignment-based metrics** that compare documents at the sentence (atomic) level.

2.  **Investigated Aggregation Methods**
    *   **Mean-Pool (Baseline)**: Averaging all sentence embeddings.
    *   **Chamfer Distance**: Computes the average of maximum sentence-to-sentence similarities, allowing for "fuzzy" matching of process steps.
    *   **Top-K (10) Alignment**: Averages only the 10 strongest sentence-to-sentence pairings to isolate the core shared procedural logic.
    *   **EMD/OT (Optimal Transport)**: Uses bipartite matching to find the globally optimal alignment cost between text units.

3.  **Performance Benchmarking**
    We benchmarked these methods against expert-annotated ground truth (Jaccard distance of annotated features).

---

### Key Findings and Success Metrics

The experiments demonstrated that moving away from global averaging significantly improves the ability to recover expert-defined relationships:

| Method | Pearson *r* | Spearman *ρ* | NN Top-1 | NN Top-3 |
| :--- | :--- | :--- | :--- | :--- |
| **Mean-Pool (Baseline)** | 0.349 | 0.434 | 4/17 | 6/17 |
| **Chamfer Alignment** | **0.707** | **0.764** | 8/17 | 11/17 |
| **Top-K (10) Alignment** | **0.692** | **0.773** | 8/17 | 11/17 |

**Conclusion**: The **Top-K (10)** and **Chamfer** methods nearly doubled the correlation with expert truth compared to the baseline.

---

### Project Structure

```text
2026-embedding-tests/
├── embedding_analysis.py  # Core script for testing pooling and distance metrics
└── find_similar.py        # Utility to identify semantic neighbors in vector space
```
