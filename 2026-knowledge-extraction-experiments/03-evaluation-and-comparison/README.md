# 03. Evaluation and Comparison Phase

This phase benchmarks the automated extraction against expert annotations and analyzes the semantic and structural properties of the results.

---

## Analysis Report: Unsupervised Knowledge Extraction

### Methodology
The pipeline extracts atomic units (procedural steps and descriptive passages) from alchemical transcriptions using LLM-based extraction (Qwen 122B) and TextTiling segmentation. 

### Predictive Validation
We tested the predictability of the expert-defined manuscript groups (I, II, and III) using our unsupervised clusters as features.
- **Method**: Random Forest and Linear SVM classifiers using Leave-One-Out Cross-Validation (LOOCV).
- **Goal**: Determine if the extracted "conceptual fingerprints" are sufficient to reconstruct historical groupings.

---

## Results Summary

### Predictive Power
The Random Forest classifier achieved a **70.6% accuracy** in reconstructing the expert manuscript groups, outperforming the SVM (52.9%). 

| Document | True Group | Predicted | Match |
| :--- | :--- | :--- | :--- |
| g1a1 | II | II | ✓ |
| g1a15 | II | I | ✗ |
| g1a16 | II | II | ✓ |
| g1a5 | II | II | ✓ |
| g1a6 | II | I | ✗ |
| g2a12 | III | III | ✓ |
| g2a13 | III | I | ✗ |
| g2a2 | III | III | ✓ |
| g2a3 | III | III | ✓ |
| g2a4 | III | III | ✓ |
| g2a7 | III | III | ✓ |
| g2a8 | III | III | ✓ |
| g3a21 | I | I | ✓ |
| g3a22 | I | III | ✗ |
| g3a25 | I | II | ✗ |
| g3a26 | I | I | ✓ |
| g3a9 | I | I | ✓ |

### Top Ranked Features (Random Forest)
The following concepts showed the highest Gini importance in the classification task:
1. Calcined Earth Mixing (0.039)
2. Vapor Fire Putrefaction (0.037)
3. Extended Heating Process (0.035)
4. Black Stage Heating (0.031)
5. Lye Evaporation Process (0.030)

### Structural Insights
- **Conceptual Distribution**: The corpus follows a power-law distribution where the majority of concepts appear in very few documents, while a small set of concepts is shared across the corpus.
- **Hybridization**: Larger conceptual clusters tend to be more "Hybrid" (mixing procedural and descriptive units), suggesting that theory and laboratory instruction are semantically intertwined at the cluster level.

---

### Project Structure

```text
03-evaluation-and-comparison/
├── analyze_cluster_composition.py   # Statistical audit: Lab vs Theory purity
├── analyze_cluster_distribution.py  # Consensus analysis: Universal vs Idiosyncratic concepts
├── analyze_group_predictability.py  # Classifier benchmarking and feature ranking
├── analyze_group_signatures.py      # Diagnostic marker extraction (Correlation analysis)
├── evaluate_extraction_modes.py     # Benchmarking extraction modes/thresholds
├── generate_cluster_report.py       # Hierarchical cluster exploration tool
└── ...
```
