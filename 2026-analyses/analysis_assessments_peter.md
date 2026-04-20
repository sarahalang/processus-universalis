# Assessment of Computational Analysis: Processus Universalis (2026)

Problem Statement

The analysis of modular textual traditions in historical manuscripts requires a methodology that extends beyond traditional text-reuse studies. It must account for:
- Historical context and authorial intent.
- Non-linear transmission patterns where recipes are recombined across sources.
- Modular composition that deviates from strictly linear stemmatology.

---

Overall Impression

The claude code analysis produces a high volume of output but lacks a clear, unified methodology. Because each report partially copies others and uses inconsistent cross-references, it remains difficult to see how the documents relate to each other or how the technical steps fit together. A central problem is that the analysis tries to force methods with completely different intents into a uniform comparison against expert categories. This leads to arbitrary statistical optimizations and obscure recommendations, like the 3% stylometry weight, which suggest the overall research goal was not fully understood. Ultimately, the most informative results are the descriptive visualizations in the documentation rather than the forced performance metrics.

---

Method-Specific Critiques

*Capstone Analysis*
The main problem is that it tries to compare methods with different intents—such as stylometry and text reuse—in a uniform way. By forcing similarity through expert categories as the sole ground truth, it ranks methods like Quadratic Delta and 4-gram overlap using the same Spearman ρ metric, which masks the unique signals each method is designed to capture.

*Cascading Analysis*
This approach fails to yield meaningful results because it lacks reasoning about what combinations would actually be meaningful. Furthermore, it also tries to only optimize the correlation constructed through the expert annotation, which does not reflect the overall research question.

*Embedding Analysis*
The analysis constructs semantic poles that are not used in the overall comparison (in the capstone analysis), which is inconsistent. The construction of the poles also seems somewhat arbitrary; also the example sentences used to construct a representative embedding vector are not clearly distinguishable. Furthermore, the use of mean pooling for document aggregation and the arbitrary use of early-half embeddings lack clear justification.

*Chemistry Language Analysis*

The categories used for word list construction need to be judged by experts as they do not appear directly aligned with one another. While the classification of vocabulary into practical and theoretical pools is a starting point, the specific purpose and definition of the color stages category remain unclear, and its broader relationship to the chemical process is not sufficiently established.

---

Methodological Overview

As a consequence of the critique, this table tries to clarify the scope and methodological intent of the considered methods:

| Method | Relational Scope | Normalization | Prerequisites | Assumptions | Interpretability | Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 4-gram Jaccard Overlap | Pairwise (Global) | Lowercase / Phonetic | Word-based plaintext | Sequence overlap implies relation | High (intuitive) | Linguistic comparison |
| Quadratic Delta (Stylometry) | Pairwise (Global) | Z-score / MFW | Word frequency distributions | Frequency reflects stylistic signal | Medium (abstract) | Document-level comparison |
| Text-Matcher (LCS) | Pairwise (Local) | Fuzzy matching | Word-based plaintext | Length of match implies similarity | Very high (visual) | Verbatim text reuse detection |
| Embeddings | Pairwise (Local) | Vector representations | Pretrained transformer model | Semantic similarity implies relation | Low (black box) | Semantic relatedness |
| FLAME | Pairwise (Global) | Rule-based | Word-based plaintext | Similar sequences imply relation | Very high (visual) | Text comparison |
| Phylogenetic Networks (SplitsTree) | Corpus-wide (Global) | Feature matrix | Document-level feature matrix | Documents share evolutionary signals | Low (highly abstract) | Relationship modeling |
