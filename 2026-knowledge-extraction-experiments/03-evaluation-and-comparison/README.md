# 03. Evaluation and Comparison Phase

This phase benchmarks the automated extraction against expert annotations and analyzes the semantic and structural properties of the results.

---

## Analysis Results Overview

The extraction and evaluation pipeline has yielded high-fidelity results. **Note: Unless otherwise stated, all statistical results in sections 2 through 6 are based on the Hybrid (Intent + Context) mode at a Threshold of 0.5.**

### 1. Extraction Mode & Threshold Benchmarking (`01_evaluate_modes.py`)
We compared three ways of representing alchemical knowledge for the clustering algorithm:
- **Raw Source Only**: Verbatim Early Modern German snippets.
- **Intent Only**: Standardized English summaries of the action/concept (`normalized_intent`).
- **Hybrid (Intent + Context)**: A combination of the English summary and the qualitative conditions/justifications (`context_and_theory`).

| Mode | Threshold | Rho(Raw) | Rho(Filt) | NN1(R) | NN3(R) | C(Raw) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hybrid** | 0.4 | **0.888** | 0.886 | 12/18 | 17/18 | 464 |
| **Hybrid** | 0.5 | 0.885 | 0.870 | 12/18 | 15/18 | 257 |
| **Intent Only** | 0.3 | 0.883 | 0.845 | 11/18 | 11/18 | 650 |
| **Intent Only** | 0.4 | 0.863 | 0.831 | 7/18 | 10/18 | 413 |
| **Intent Only** | 0.5 | 0.842 | 0.822 | 9/18 | 12/18 | 218 |
| **Hybrid** | 0.6 | 0.840 | 0.786 | 10/18 | 11/18 | 129 |
| **Raw Source** | 0.4 | 0.794 | 0.783 | 13/18 | 16/18 | 306 |
| **Hybrid** | 0.3 | 0.788 | 0.729 | 13/18 | 16/18 | 747 |
| **Raw Source** | 0.5 | 0.778 | 0.784 | 10/18 | 11/18 | 132 |
| **Intent Only** | 0.6 | 0.773 | 0.768 | 8/18 | 11/18 | 115 |

### 2. Tradition Predictability (`02_analyze_predictability.py`)
This experiment tests whether the conceptual clusters successfully capture the "Phylogenetic Signal" of the alchemical tradition.

**Accuracy: 76.5%** (Random Forest) vs. 70.6% (SVM)

| Document | True Group | Predicted | Match |
| :--- | :--- | :--- | :--- |
| g1a1 | II | II | ✓ |
| g1a15 | II | I | ✗ |
| g1a16 | II | II | ✓ |
| g1a5 | II | II | ✓ |
| g1a6 | II | I | ✗ |
| g2a12 | III | III | ✓ |
| g2a13 | III | III | ✓ |
| g2a2 | III | III | ✓ |
| g2a3 | III | III | ✓ |
| g2a4 | III | III | ✓ |
| g2a7 | III | III | ✓ |
| g2a8 | III | III | ✓ |
| g3a21 | I | I | ✓ |
| g3a22 | I | III | ✗ |
| g3a25 | I | III | ✗ |
| g3a26 | I | I | ✓ |
| g3a9 | I | I | ✓ |

### 3. Knowledge Purity vs. Hybridity (`03_analyze_composition.py`)
Audit of the 257 conceptual clusters (Threshold 0.5) to determine the balance between technical laboratory actions and theoretical context.

| Category | Count | Description |
| :--- | :--- | :--- |
| **Pure Procedural** | 109 | 100% Laboratory Actions |
| **Pure Descriptive** | 77 | 100% Theoretical/Conceptual |
| **Balanced Mixed** | 41 | Integrated Theory & Practice |
| **Mostly Proc/Desc** | 30 | Dominant focus (>75%) |

### 4. Consensus & Rarity Distribution (`04_analyze_distribution.py`)
- **Universal Core**: *Salt Purification and Crystallization* appears in **100%** of documents.
- **Common Shared**: *Spirit Distillation Setup* and *Fixed Gold Preparation* appear in **83.3%** of documents.
- **Rarity**: 92 concepts are unique to a single document, while 58 are shared by only two.
- **Complexity**: On average, each manuscript contains **47.67** unique conceptual steps.

### 5. Diagnostic Group Signatures (`05_analyze_signatures.py`)
Highest Point-Biserial correlations per tradition:
- **Group I**: Zodiacal Earth Sourcing, Sunlight Precipitation, Silver Dissolution.
- **Group II**: Straw and Earth Layering (corr=1.0), Athanor Putrefaction Phase.
- **Group III**: Athanor Structure Theory, Coloration Phase Initiation, Earth Celestial Impregnation.

### 6. Interactive Cluster Synthesis (`06_generate_cluster_report.py`)
**Report Location**: `2026-knowledge-extraction-experiments/data/cluster_report.html`

---

## Core Evaluation Pipeline

**Note on Dependencies:**
- **Script 00** (`00_label_clusters.py`) must be run first to generate conceptual labels.
- **Scripts 01 and 06** operate directly on the raw extraction results (`atomic_extraction_results.json`).
- **Scripts 02, 03, 04, and 05** depend on the labeled data (`labeled_segments.json`).

| Script | Purpose | Implementation Logic |
| :--- | :--- | :--- |
| `00_label_clusters.py` | Semantic Labeling | Clusters atomic units and uses LLM batch-summarization for concept naming. |
| `01_evaluate_modes.py` | Benchmarks extraction | Sensitivity sweep (0.3-0.7) comparing derived Jaccard matrices against expert XML ground truth. |
| `02_analyze_predictability.py` | Validates tradition | Random Forest & SVM classification (LOOCV) using conceptual fingerprints. |
| `03_analyze_composition.py` | Audits Purity | Calculates procedural-to-descriptive ratios per cluster. |
| `04_analyze_distribution.py` | Maps Consensus | Computes document frequency to find Universal Core vs. Unique flourishes. |
| `05_analyze_signatures.py` | Group Signatures | Correlates clusters with expert groups to find diagnostic markers. |
| `06_generate_cluster_report.py` | Qualitative Tool | Hierarchical HTML report mapping atomic units to clusters. |

---

## Project Structure

```text
03-evaluation-and-comparison/
├── archive/                      # Obsolete pilot tests and diagnostic scripts
├── core-evaluation/              # Validated analysis pipeline
│   ├── 00_label_clusters.py
│   ├── 01_evaluate_modes.py
│   ├── 02_analyze_predictability.py
│   ├── 03_analyze_composition.py
│   ├── 04_analyze_distribution.py
│   ├── 05_analyze_signatures.py
│   └── 06_generate_cluster_report.py
└── README.md
```
