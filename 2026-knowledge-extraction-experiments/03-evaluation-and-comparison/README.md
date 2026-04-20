# 03. Evaluation and Comparison Phase

This phase benchmarks the automated extraction against expert annotations and analyzes the semantic properties of the results.

---

### Project Structure

```text
03-evaluation-and-comparison/
├── analyze_extraction_stats.py    # Statistical summary of extracted data
├── author_test.py                 # Stylistic fingerprint detection
├── calculate_correlation.py       # Metrics for text-expert agreement
├── compare_alignment.py           # Evaluation of segment alignment
├── compare_extraction_to_xml.py   # Maps LLM output back to original tags
├── compare_final.py               # Comparative report of pipeline performance
├── compare_keys.py                # Direct keyword matching evaluation
├── embedding_analysis.py          # Semantic space analysis of segments
├── find_similar.py                # Finds procedural patterns in embedding space
├── generate_automated_matrix.py   # Reconstructs presence/absence matrix from LLM
├── show_comparison.py             # Visual reporting of results
└── verify_extraction_v2.py        # Plausibility checks for chemical steps
```
