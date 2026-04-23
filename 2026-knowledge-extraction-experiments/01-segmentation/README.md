# 01. Segmentation Phase

The *sammlung_aller_texte.xml* does not contain any natural paragraph markers. All structural breaks are XML tag artifacts. To achieve a truly unsupervised result, we use a layered approach that detects lexical cohesion shifts and aligns them with grammatical boundaries.

### Implementation Detail: TextTiling with Sentence-Level Alignment

We use an unsupervised TextTiling algorithm that detects lexical valleys—points where the cosine similarity of term-frequency vectors between adjacent sliding windows drops significantly ($\mu - 0.5\sigma$). 

#### Pipeline Steps:
1.  **Lexical Cohesion Detection**: A sliding-window analysis on a normalized (lowercase, content-filtered) term stream detects "raw" transition points at the word level where similarity drops below the threshold.
2.  **Grammatical Alignment**: Raw transition points are "snapped" to the nearest sentence-final punctuation (., !, ?).
3.  **Abbreviation Filtering**: Common alchemical abbreviations are explicitly ignored to ensure segments do not break mid-sentence.
4.  **Reconstruction**: Segments are reconstructed from the original raw token stream, preserving punctuation and casing, and exported as a structured CSV.

### Execution Workflow
The scripts should be executed in the following order:

1. `01_run_segmentation.py` - Performs lexical cohesion analysis, snaps to grammatical boundaries, and exports the final, structured segment CSV.
2. `02_validate_segmentation.py` - Performs quality control (length distribution and coherence checks).

```text
01-segmentation/
├── archive/
├── 01_run_segmentation.py
└── 02_validate_segmentation.py
```
