# 03. Evaluation and Comparison Phase

This phase benchmarks the automated extraction against expert annotations and analyzes the semantic and structural properties of the results.

---

## Analysis Results Overview

The extraction and evaluation pipeline has yielded high-fidelity results. **Note: All statistical results in sections 2 through 6 are based on the optimal clustering configuration: Hybrid (Intent + Context) mode at a Threshold of 0.5.**

### 1. Extraction Mode & Threshold Benchmarking (`01_evaluate_modes.py`)
We compared three ways of representing alchemical knowledge for the clustering algorithm:
- **Raw Source Only**: Verbatim Early Modern German snippets.
- **Intent Only**: Standardized English summaries of the action/concept (`normalized_intent`).
- **Intent + Context (Hybrid)**: A combination of the English summary and the qualitative conditions/justifications (`context_and_theory`).

The pipeline was evaluated by comparing derived Jaccard similarity matrices against the fine-grained expert XML ground truth.

| Mode | Threshold | Spearman $\rho$ | NN-1 | NN-3 | Clusters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent + Context** | 0.5 | **0.906** | 12/18 | 13/18 | 239 |
| **Intent + Context** | 0.4 | 0.856 | 13/18 | 17/18 | 449 |
| **Intent Only** | 0.4 | 0.844 | 10/18 | 12/18 | 428 |
| **Intent + Context** | 0.6 | 0.832 | 12/18 | 13/18 | 108 |
| **Intent + Context** | 0.3 | 0.827 | 12/18 | 17/18 | 704 |
| **Intent Only** | 0.5 | 0.810 | 11/18 | 14/18 | 230 |
| **Raw Source Only** | 0.4 | 0.795 | 12/18 | 14/18 | 320 |
| **Raw Source Only** | 0.3 | 0.794 | 12/18 | 15/18 | 615 |
| **Intent Only** | 0.6 | 0.777 | 12/18 | 15/18 | 121 |
| **Raw Source Only** | 0.5 | 0.760 | 6/18 | 11/18 | 137 |

### 2. Tradition Predictability (`02_analyze_predictability.py`)
Using Random Forest classification with Leave-One-Out Cross-Validation (LOOCV), we tested how well unsupervised clusters predict scholarly Groups I-III.

**Accuracy: 70.6%** (Random Forest) vs. 52.9% (SVM)

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

### 3. Knowledge Purity vs. Hybridity (`03_analyze_composition.py`)
Audit of the 239 conceptual clusters (Threshold 0.5) to determine the balance between technical laboratory actions and theoretical context.

| Category | Count | Description |
| :--- | :--- | :--- |
| **Pure Procedural** | 105 | 100% Laboratory Actions |
| **Pure Descriptive** | 72 | 100% Theoretical/Conceptual |
| **Balanced Mixed** | 30 | Integrated Theory & Practice |
| **Mostly Proc/Desc** | 32 | Dominant focus (>75%) |

### 4. Consensus & Rarity Distribution (`04_analyze_distribution.py`)
Analysis of the "Alchemical Common Core" vs. idiosyncratic "Unique Flourishes."

- **Unique Fingerprints**: 89 concepts appear in exactly one document.
- **Genetic Links**: 52 concepts shared by exactly two documents.
- **Universal Core**: 1 concept appears in all 18 documents.
- **Consensus**: Only ~9% of concepts (22/239) are shared by >50% of the corpus.

### 5. Diagnostic Group Signatures (`05_analyze_signatures.py`)
Point-Biserial correlation identifies clusters most diagnostic for specific traditions:
- **Group I Signature**: Solar Precipitation, Crucible Melting, Medical Efficacy.
- **Group II Signature**: Earth Layer Compression, Impurity Discard, Controlled Putrefaction Heat.
- **Group III Signature**: Silver Plate Testing, Black Stage Heating, Gold Menstruum Mixture.

### 6. Interactive Cluster Synthesis (`06_generate_cluster_report.py`)
Qualitative exploration of the cluster hierarchy is available in the interactive HTML report.

**Report Location**: `2026-knowledge-extraction-experiments/data/cluster_report.html`

---

## Core Evaluation Pipeline

These scripts provide the validated, reproducible methodology for evaluating the knowledge extraction results. 

| Script | Purpose | Implementation Logic |
| :--- | :--- | :--- |
| `01_evaluate_modes.py` | Benchmarks extraction pipeline | Performs sensitivity sweep (0.3-0.7) comparing derived Jaccard-cluster distance matrices against fine-grained XML feature annotations using Spearman Correlation ($\rho$). |
| `02_analyze_predictability.py` | Validates tradition mapping | Implements Random Forest & SVM classifiers with Leave-One-Out Cross-Validation (LOOCV) to test if cluster-based conceptual fingerprints predict scholarly Groups I-III. |
| `03_analyze_composition.py` | Audits Purity vs. Hybridity | Calculates procedural-to-descriptive ratios per cluster to quantify the balance between laboratory operations and theoretical conditions. |
| `04_analyze_distribution.py` | Maps Alchemical Consensus | Computes document frequency per cluster to distinguish between Universal Core concepts and Idiosyncratic manuscript-specific flourishes. |
| `05_analyze_signatures.py` | Diagnostic Group Signatures | Uses Point-Biserial correlation to identify which specific clusters act as diagnostic gatekeepers for distinguishing between alchemical traditions. |
| `06_generate_cluster_report.py` | Qualitative Synthesis (Tool) | Compiles a hierarchical, interactive HTML report that maps every atomic unit to its cluster; essential for interpreting the conceptual labels assigned by the LLM. |

---

## Project Structure

```text
03-evaluation-and-comparison/
├── archive/                      # Obsolete pilot tests and diagnostic scripts
├── core-evaluation/              # Validated, repeatable analysis pipeline
│   ├── 01_evaluate_modes.py
│   ├── 02_analyze_predictability.py
│   ├── 03_analyze_composition.py
│   ├── 04_analyze_distribution.py
│   ├── 05_analyze_signatures.py
│   └── 06_generate_cluster_report.py
└── README.md
```
