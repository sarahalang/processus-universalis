# 02. Knowledge Extraction Phase

This phase handles the transformation of segmented Early Modern German text into structured, machine-readable knowledge units using a Large Language Model (Qwen 122B).

---

## The Dual-Mode Extraction Strategy

To capture the complexity of alchemical manuscripts, the extraction engine distinguishes between two fundamental types of knowledge:

1.  **Procedural Steps**: Precise laboratory actions (e.g., *distilling*, *calcining*, *leaching*). These are extracted at a granular level—one unit per distinct instruction—to preserve the exact technical sequence.
2.  **Descriptive Passages**: Theoretical claims, philosophical justifications, or qualitative observations (e.g., *celestial timing*, *theoretical mercury*, *expected colors*). These are consolidated into coherent thematic blocks to provide semantic context without fragmenting the argument.

---

## Technical Implementation

### Extraction Engine (`01_extract_atomic_knowledge.py`)
The engine is built for high-volume processing and handles the variability of long-running API calls with three key features:

*   **Robust JSON Recovery**: Includes a `repair_json` utility that automatically strips preambles, handles markdown wrappers, and fixes common truncation errors (like missing closing braces) to maximize data recovery from interrupted streams.
*   **Incremental State Persistence**: The script saves results after every successfully processed segment, allowing for seamless resumes after API timeouts or local restarts.
*   **Structured Intent Mapping**: Every unit includes a `normalized_intent` in English (for cross-lingual clustering) while preserving the `raw_source` in the original German (for philological validation).

### Data Schema
The extracted JSON contains:
- `unit_type`: Classification as procedural or descriptive.
- `normalized_intent`: Standardized summary of the action/concept.
- `details`: Entities like `operation` (verb), `materials`, and `apparatus`.
- `context_and_theory`: Qualitative conditions or background knowledge.
- `state_dependencies`: Logical prerequisites for the step.
- `raw_source`: The verbatim German snippet used for the extraction.

---

### Project Structure

```text
02-knowledge-extraction/
├── archive/                      # Legacy and experimental methods (see archive/README.md)
└── 01_extract_atomic_knowledge.py # Main engine for dual-mode atomic step extraction
```
