# Knowledge Extraction Experiments (2026)

This project implements an automated pipeline for transforming raw historical alchemical transcriptions into structured, machine-readable knowledge. The goal is to reconstruct the "Phylogenetic Tree" of the alchemical tradition purely from the logic of procedural steps, validated against the manual expert ground truth.

### Overall Intent
The analysis aims to demonstrate that Large Language Models (LLMs) and semantic clustering can capture the underlying procedural and theoretical logic of alchemical manuscripts with near-human accuracy (Spearman $\rho = 0.906$). This allows for the discovery of shared traditions and "diagnostic markers" without relying on pre-defined keywords or manual segmentation.

---

## Pipeline Phases

The workflow is divided into four conceptual stages:

### [01. Segmentation](./01-segmentation/)
This phase focuses on how to logically divide continuous recipe text into discrete procedural steps. We use unsupervised **TextTiling** with sentence-level alignment to identify logical transition points in the text based on shifts in lexical cohesion, creating coherent procedural segments independent of XML tags.

### [02. Knowledge Extraction](./02-knowledge-extraction/)
Once segmented, individual process steps are processed using Large Language Models (LLMs). We use a dual-mode extraction engine to transform segments into **Atomic Knowledge Units**, distinguishing between *Procedural Steps* (physical actions) and *Descriptive Passages* (theoretical context).

### [03. Evaluation and Comparison](./03-evaluation-and-comparison/)
This phase benchmarks the automated results against human-expert "ground truth." It includes statistical validation (Spearman correlation), audits of conceptual "purity," and the use of **Random Forest classifiers** to identify diagnostic markers of specific manuscript traditions.

### [04. Conceptual Trajectory](./04-conceptual-trajectory/)
The final phase focuses on labeling the discovered alchemical concepts and visualizing their flow across the corpus. It assigns semantic labels to discovered clusters and provides an **Interactive Annotated Reader** to trace the flow of concepts across the entire corpus.
