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

| Mode            | Threshold | Rho(Raw)  | Rho(Filt) | NN1(R) | NN3(R) | C(Raw) |
| :-------------- | :-------- | :-------- | :-------- | :----- | :----- | :----- |
| **Hybrid**      | 0.5       | **0.885** | 0.868     | 13/17  | 14/17  | 252    |
| **Hybrid**      | 0.4       | **0.878** | 0.882     | 13/17  | 16/17  | 448    |
| **Intent Only** | 0.3       | 0.864     | 0.824     | 10/17  | 11/17  | 634    |
| **Intent Only** | 0.4       | 0.864     | 0.835     | 9/17   | 11/17  | 401    |
| **Hybrid**      | 0.6       | 0.839     | 0.794     | 11/17  | 13/17  | 123    |
| **Intent Only** | 0.5       | 0.809     | 0.801     | 9/17   | 10/17  | 213    |
| **Raw Source**  | 0.5       | 0.803     | 0.795     | 9/17   | 11/17  | 128    |
| **Raw Source**  | 0.4       | 0.782     | 0.784     | 12/17  | 15/17  | 293    |
| **Intent Only** | 0.6       | 0.762     | 0.752     | 6/17   | 10/17  | 118    |
| **Hybrid**      | 0.3       | 0.761     | 0.708     | 12/17  | 14/17  | 717    |

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
Point-Biserial correlation identifies clusters that are most statistically diagnostic for specific traditions. 

- **Group I**: Zodiacal Earth Sourcing (0.566), Sunlight Precipitation (0.566), Silver Dissolution Capability (0.566), Extended Cooking Cycle (0.533), Dissolution Ratio Specifics (0.432).
- **Group II**: Straw and Earth Layering (1.000), Athanor Putrefaction Phase (0.859), Philosophical Water Union (0.859), Product Mixing and Storage (0.859), Gold Mercury Mixing (0.803).
- **Group III**: Athanor Structure Theory (0.757), Coloration Phase Initiation (0.703), Earth Celestial Impregnation (0.661), Second Spirit Purification (0.650), Red Grain Significance (0.639).

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
