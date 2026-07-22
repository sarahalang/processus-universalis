# Archive: Segmentation Phase Diagnostics

This directory contains scripts that were used to investigate the structural properties of the *sammlung_aller_texte.xml* corpus and justify the need for unsupervised segmentation.

### Archive Contents

```text
archive/
├── xml_to_segment_table.py                  # Baseline XML-to-CSV flattening utility
├── unsupervised_segmentation.py             # Earlier iteration of the lexical engine
├── check_nl.py                              # Verified absence of double newlines
├── find_indented_paragraphs.py              # Investigated indentation artifacts
├── count_whitespace_segments.py             # Verified whitespace splitting limitations
├── heading_analysis.py                      # Analyzed XML headers as markers
├── segmentation_comparison_texttiling_vs_anchors.py # Comparative anchor method evaluation
├── structural_segmentation.py               # XML-tag based segmentation (discarded)
└── test_segmentation.py                     # Unit tests for core functions
```

### Descriptions

* **xml_to_segment_table.py**: A baseline XML-to-CSV flattening utility; superseded by the unsupervised segmentation pipeline.
* **unsupervised_segmentation.py**: An earlier iteration of the lexical segmentation engine; superseded by the current, unified `01_run_segmentation.py`.
* **check_nl.py**: Searched for double newlines (`\n\n`) within `<div` tags to determine if the text contained natural paragraph markers. Found zero, confirming that structural breaks were purely tag-dependent.
* **find_indented_paragraphs.py**: Investigated whether indentation shifts (e.g., from 8 to 1 space) in the text aligned with logical breaks, proving they were artifacts of XML tag nesting rather than intentional paragraphing.
* **count_whitespace_segments.py**: Verified that standard whitespace-based splitting resulted in only one giant segment per document, justifying the move to a more advanced lexical cohesion algorithm.
* **heading_analysis.py**: Analyzed whether headers could serve as structural markers. Concluded that relying on headers was too circular, as they were expert-annotated.
* **segmentation_comparison_texttiling_vs_anchors.py**: Provided a comparative evaluation that justified the choice of TextTiling over Auto-Anchor methods for the final pipeline.
* **structural_segmentation.py**: An alternative segmenter based on XML tag markers; superseded by the more robust unsupervised TextTiling method.
* **test_segmentation.py**: Provided unit tests for core segmentation functions (e.g., grammatical boundary detection, abbreviation filtering) to ensure logical correctness during development.
