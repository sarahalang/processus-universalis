# 01. Segmentation Phase

The *sammlung_aller_texte.xml* does not contain any natural paragraph markers like double newlines or indentation shifts in the document text. This means all the structural breaks we see are actually created by the XML tags themselves. 

We checked this with a few different scripts:
- **check_nl.py**: searched for any double newlines (\n\n) inside the div elements but found zero across all 18 documents.
- **find_indented_paragraphs.py**: looked for cases where the text indentation shifts from the standard 8 spaces to 1 space. This confirmed that the shift only happens when a new XML tag starts.
- **count_whitespace_segments.py**: tried to split the text while ignoring all tags, which resulted in only 1 segment per document.

In short, using the XML tags for segmentation is circular because it relies on the expert's manual tagging. To get a truly unsupervised result, we need a method that works without using these tags or the expert keywords.

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
├── unsupervised_segmentation.py   # Main engine using lexical shift detection
├── validate_segmentation.py       # Metrics for segment coherence
└── test_segmentation.py           # Unit tests for segmentation logic
```
