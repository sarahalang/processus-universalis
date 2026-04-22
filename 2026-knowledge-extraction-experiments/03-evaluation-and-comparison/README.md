# 03. Evaluation and Comparison Phase

This phase benchmarks the automated extraction against expert annotations and analyzes the semantic and structural properties of the results.

---

## Analysis Report: Unsupervised Knowledge Extraction

### Methodology
The pipeline extracts atomic units (procedural steps and descriptive passages) from alchemical transcriptions using LLM-based extraction (Qwen 122B) and TextTiling segmentation. We investigated three extraction modes:
1. **Intent Only**: Standardized procedural/thematic intent.
2. **Raw Source**: Verbatim alchemical German snippets.
3. **Intent + Context**: A hybrid approach combining the intent with conditional theoretical justifications.

### Clustering & Comparison
We employed **Agglomerative Hierarchical Clustering** on semantic embeddings (Multi-lingual MiniLM) to group units. To validate the quality of these clusters, we:
- Computed document-level similarities using **Jaccard similarity of cluster membership**.
- Compared these results against an expert-annotated **Gold Standard** (derived from the Capstone matrix).
- Used **Spearman Rank Correlation ($\rho$)** to measure structural agreement between automated groupings and scholarly traditions.

---

## Results Summary
The "Intent + Context" hybrid mode significantly outperformed the baseline and previous Capstone methodologies. 

### Leaderboard (Optimal Configuration)
| Mode | Threshold | Clusters | Spearman $\rho$ | NN-1 | NN-3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent + Context** | 0.5 | 239 | **0.906** | 12/18 | 13/18 |
| **Intent + Context** | 0.6 | 108 | 0.832 | 12/18 | 13/18 |
| **Intent Only** | 0.4 | 428 | 0.844 | 10/18 | 12/18 |

---

## Interpretations
1. **Contextual Signal**: Including "context and theory" (the philosophical/alchemical rationale) provides critical semantic signals that significantly improve cluster cohesion.
2. **Structural Coherence**: The peak Spearman $\rho$ of 0.9l06 demonstrates that the automated extraction mirrors the expert structural analysis with high fidelity.
3. **Hierarchy of Abstraction**: The sensitivity analysis reveals that the corpus naturally organizes into roughly 80–150 core procedural steps, matching the granularity of classical alchemical classifications.

---

### Project Structure

```text
03-evaluation-and-comparison/
├── analyze_extraction_stats.py    # Statistical summary of extracted data
├── author_test.py                 # Stylistic fingerprint detection
├── calculate_correlation.py       # Metrics for text-expert agreement
├── compare_alignment.py           # Evaluation of segment alignment
├── compare_extraction_to_xml.py   # Maps LLM output back to original tags
├── compare_final.py               # Comparative report of pipeline performance
├── compare_keys.py                # Direct keyword matching evaluation
├── evaluate_extraction_modes.py   # Main benchmarking suite for modes/thresholds
├── generate_automated_matrix.py   # Reconstructs presence/absence matrix from LLM
├── generate_cluster_report.py     # Hierarchical cluster exploration tool
├── pilot_clustering.py            # Initial semantic grouping of atomic intents
├── pilot_richness_analysis.py     # Evaluation of extracted procedural details
├── show_comparison.py             # Visual reporting of results
└── verify_extraction_v2.py        # Plausibility checks for chemical steps
```
