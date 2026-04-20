# Knowledge Extraction Experiments (2026)

This project implements an automated pipeline for transforming raw historical alchemical transcriptions into structured, machine-readable knowledge. The workflow is divided into three conceptual stages:

### 01. Segmentation
This phase focuses on how to logically divide continuous recipe text into discrete procedural steps. The primary challenge is finding a "natural" way to segment the text without relying on manual expert annotations or keywords.

### 02. Knowledge Extraction
Once segmented, individual process steps are processed using Large Language Models (LLMs). The goal is to identify and extract entities, actions, and materials into a structured "Subject-Verb-Entity-Context" (SVEK) format.

### 03. Evaluation and Comparison
The final phase benchmarks the automated results against human-expert "ground truth." This includes statistical comparisons of extracted keys, semantic embedding analysis of the procedural steps, and validation of chemical plausibility.
