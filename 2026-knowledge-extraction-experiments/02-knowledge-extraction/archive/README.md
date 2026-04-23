# Archive: Knowledge Extraction Diagnostics

This directory contains legacy and experimental extraction methods that were tested during the development of the pipeline but are not part of the active production workflow.

### Archive Contents

```text
archive/
├── extract_opac_knowledge.py      # Early LLM engine using a coarse-grained SVEK-style schema
└── ie_reconstruction.py           # Rule-based entity-alignment using phonetic matching
```

### Descriptions

* **extract_opac_knowledge.py**: An earlier iteration of LLM extraction that focused on "Main Operations" in German. It used a simpler schema (OPAC) and was superseded by the more granular, dual-mode atomic extraction.
* **ie_reconstruction.py**: A non-LLM alternative that attempted to reconstruct procedural logic by matching keywords using the Cologne Phonetic algorithm. This approach proved less flexible than semantic LLM-based extraction.
