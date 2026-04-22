# 01. Segmentation Phase

The *sammlung_aller_texte.xml* does not contain any natural paragraph markers like double newlines or indentation shifts in the document text. This means all the structural breaks we see are actually created by the XML tags themselves. 

We checked this with a few different scripts:
- **check_nl.py**: searched for any double newlines (\n\n) inside the div elements but found zero across all 18 documents.
- **find_indented_paragraphs.py**: looked for cases where the text indentation shifts from the standard 8 spaces to 1 space. This confirmed that the shift only happens when a new XML tag starts.
- **count_whitespace_segments.py**: tried to split the text while ignoring all tags, which resulted in only 1 segment per document.

In short, using the XML tags for segmentation is circular because it relies on the expert's manual tagging. To get a truly unsupervised result, we need a method that works without using these tags or the expert keywords.

### Unsupervised Segmentation Implementation Detail: TextTiling with Sentence-Level Alignment

To establish a segmentation that is independent of expert annotations, we use a layered approach that detects shifts in lexical cohesion and aligns them with the grammatical structure of the text.

#### 1. Measuring Lexical Cohesion
The text is treated as a distribution of terms. We slide two adjacent windows ($W_1$ and $W_2$) of size $k$ through the document and calculate the **Cosine Similarity** of their term-frequency vectors at each position. This provides a continuous measure of how the vocabulary is shifting:

$$sim(W_1, W_2) = \frac{W_1 \cdot W_2}{\|W_1\| \|W_2\|} = \frac{\sum_{i=1}^{n} f_{1,i} f_{2,i}}{\sqrt{\sum_{i=1}^{n} f_{1,i}^2} \sqrt{\sum_{i=1}^{n} f_{2,i}^2}}$$

Where $f_{1,i}$ is the frequency of term $i$ in the first window. A significant drop in similarity indicates a "lexical valley," where the vocabulary of the preceding section significantly diverges from the following one, signaling a transition in the recipe's procedural focus.

#### 2. The Multi-Stage Segmentation Pipeline
The implementation (`export_texttiling_segments.py`) follows a three-stage workflow to ensure that the resulting segments are both logically and grammatically coherent:

- **Structural Preprocessing and Feature Extraction**: The text is extracted from the XML while excluding expert-tagged headers to avoid circularity. We then generate two parallel representations: a raw token stream (to preserve punctuation and casing for the final output) and a normalized term stream (lowercase, content-filtered) used for the similarity calculations. During this stage, we also map all grammatical boundary points (sentence-final punctuation) to their respective token indices.
- **Lexical Change Point Detection**: A sliding-window analysis is performed on the normalized term stream. The algorithm identifies "raw" transition points at the word level wherever the local Cosine Similarity falls significantly below the corpus-wide mean ($\mu - 0.5\sigma$).
- **Grammatical Alignment and Reconstruction**: To prevent the fragmentation of individual instructions, each raw transition point is "snapped" to the nearest previously identified grammatical boundary. To distinguish true sentence endings from abbreviations (e.g., *lb.*, *thl.*, *NB.*), the algorithm requires that a boundary token be followed by an uppercase letter and not be present in a predefined list of common alchemical abbreviations. The final segments are then reconstructed from the raw token stream and exported into a structured format.

---

### Method Comparison Analysis
A comparative analysis was performed between **TextTiling** (global lexical cohesion) and **Auto-Anchors** (technical term shift detection). 
- **TextTiling** acts as a fine-grained detector, splitting text based on broader vocabulary shifts. It tends to produce a higher number of segments and is more sensitive to local stylistic variations.
- **Auto-Anchors** functions as a procedural stage detector, triggering boundaries only when significant technical vocabulary changes. This method yields fewer, more stable segments that align closely with high-level procedural units.
- **Quality Check**: We evaluate the segment length distribution to identify potential over-segmentation (too many segments < 20 words) or under-segmentation (massive chunks > 300 words).


```text
01-segmentation/
├── check_nl.py                    # Verifies absence of double newlines (\n\n)
├── find_indented_paragraphs.py    # Confirms indentation shifts align with tags
├── count_whitespace_segments.py   # Shows tag-less splitting yields 1 segment/doc
├── heading_analysis.py            # Analyzes headers as structural markers
├── xml_to_segment_table.py        # Flattens XML into a CSV for processing
├── segmentation_comparison_texttiling_vs_anchors.py # Compares TextTiling vs Auto-Anchor methods
├── export_texttiling_segments.py  # Exports TextTiling segments to CSV format
├── unsupervised_segmentation.py   # Main engine using lexical shift detection
├── validate_segmentation.py       # Metrics for segment coherence
└── test_segmentation.py           # Unit tests for segmentation logic
```
