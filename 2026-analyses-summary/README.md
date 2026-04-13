## Problem Statement

**What methods are suited to analyzing modular textual traditions in historical manuscripts?**

This problem goes beyond classical text-reuse studies. It requires accounting for:

- **Historical context**
- **Authorial or scribal intent**
- **Non-linear transmission patterns**

Unlike traditional **stemmatology**, these manuscripts do not evolve in a strictly linear fashion. Instead, they exhibit **modular composition**, recombination, and reuse across sources.

---

## Relationship Modeling and Similarity Measures

The following table compares different computational methods (already considered) for modeling relationships between texts:

| Method                                | Relational Scope        | Normalization          | Prerequisites                 | Assumptions                          | Interpretability        | Suitability                     |
|--------------------------------------|--------------------------|-------------------------|------------------------------|--------------------------------------|--------------------------|----------------------------------|
| **4-gram Jaccard Overlap**           | Pairwise (Global)        | Lowercase / Phonetic    | Word-based plaintext          | Sequence overlap implies relation     | High (intuitive)         | Linguistic comparison            |
| **Quadratic Delta (Stylometry)**     | Pairwise (Global)        | Z-score / MFW           | Word frequency distributions  | Frequency reflects stylistic signal   | Medium (abstract)        | Document-level comparison        |
| **Text-Matcher (LCS)**               | Pairwise (Local)         | Fuzzy matching          | Word-based plaintext          | Length of match implies similarity    | Very high (visual)       | Verbatim text reuse detection    |
| **Embeddings**                       | Pairwise (Local)         | Vector representations  | Pretrained transformer model  | Semantic similarity implies relation  | Low (black box)          | Semantic relatedness             |
| **FLAME**                            | Pairwise (Global)        | Rule-based              | Word-based plaintext          | Similar sequences imply relation      | Very high (visual)       | Text comparison                  |
| **Phylogenetic Networks (SplitsTree)** | Corpus-wide (Global)   | Feature matrix          | Document-level feature matrix | Documents share evolutionary signals  | Low (highly abstract)    | Relationship modeling            |

### Note

A single, unified evaluation framework may not be appropriate, as these methods serve **different analytical purposes** (e.g., stylistic vs. semantic vs. structural comparison).

---

## Feature Engineering

Relevant feature types (taken from: https://arxiv.org/pdf/2510.27045) for modeling textual relationships include:

- **Overlap features**
  - e.g., n-gram similarity, shared sequences

- **Vector features**
  - e.g., embeddings from transformer models

- **Stylistic features**
  - Textual cadence
  - Sentence structure
  - Entity distribution
  - Function word usage patterns
  - Particularly useful for **poetry and drama**

- **Metadata features**
  - e.g., date, origin, manuscript context

- **Sequence labeling**
  - e.g., identifying quoted or reused text segments

