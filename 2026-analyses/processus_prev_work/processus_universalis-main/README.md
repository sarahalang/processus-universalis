# Semantic Analysis for Processus Universalis Documents
In this repository, we aim to analyze the relationships among the [Processus Universalis text documents](./ProcessusUniversalis_relevant-files-for-2025/). Our goal is to identify instances of text reuse and to investigate the historical originality of these documents.

We implement two main approaches:
- **Text similarity with [FLAME](https://github.com/kreeedit/FLAME)** to check if there are instances of text reuse.
- **Phylogenetic analysis** of the [XML corpus](./ProcessusUniversalis_relevant-files-for-2025/sammlung_aller_texte.xml) to extract semantic features and investigate deeper relationships among the documents. We [visualize these relationships](./results/splitstree/neighbornet-20251208.png) using [SplitsTree v4](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/algorithms-in-bioinformatics/software/splitstree/) to group similar documents and explore their originality.

## Usage and Pipeline

### Installation
Please install dependencies from `requirements.txt` (Recommended to build a Python virtual environment first)
```cmd
pip install -r requirements.txt
```
For **Text similarity with FLAME**, please install `FLAME` as follows:
```cmd
git clone https://github.com/kreeedit/FLAME
cd FLAME
pip install -r requirements.txt
```
For **Phylogenetic analysis**, please install SplitsTree v4 from [here](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/algorithms-in-bioinformatics/software/splitstree/).

### Pipeline
Our source data are stored in [`ProcessusUniversalis_relevant-files-for-2025`](./ProcessusUniversalis_relevant-files-for-2025/). Data generated during processing are saved in [`processed_data`](./processed_data/), and final results are stored in [`results`](./results/)
- **Text similarity with FLAME**: Run `python3 text_sim_flame.py` to generate results in [`results/flame`](./results/flame/)
- **Phylogenetic analysis**: Run `python3 phylogenetics.py` to produce the NEXUS file (`.nex`) and related outputs in [`results/splitstree`](./results/splitstree/).

## Methodology in Details

### Text similarity with FLAME

To investigate text-reuse and semantic relationships among the Processus Universalis documents, we employed the [FLAME](https://github.com/kreeedit/FLAME) toolkit for text similarity analysis. Our initial experiments revealed a key challenge: despite the presence of highly similar or even nearly identical phrases across documents, the overall similarity scores produced by FLAME were often not lower than expected. This is likely due to the historical orthographic variation and minor spelling differences, e.g., "*sendivogij*" versus "*sendivogy*." As a result, direct document-to-document similarity metrics failed to capture the nuanced patterns of textual borrowing and reuse that are of interest to both historians and philologists.

To address this, we shifted our focus from global similarity scores to the local alignments that FLAME identifies, specifically the "bridge words" highlighted in the HTML comparison reports. These bridge words are the terms or short phrases that occur between matched passages and are thus strong candidates for being semantically equivalent or functionally synonymous, even if their surface forms differ. By extracting these bridge words, we can systematically analyze the correspondences that underlie broader patterns of textual transmission.

Our methodology proceeds as follows. We decompose the bridge phrases between two documents pairwisely, then compute the Jaro-Winkler (JW) similarity between each pair of bridge words, both in their original form and in their phonetic encoding. For example, the pair "*sendivogij*" and "*sendivogy*" yields a JW similarity of 0.90, and their phonetic codes ("*SNTFJJ*", "*SNTFJ*") yield a JW similarity of 0.94. This dual approach allows us to identify pairs of words that are either orthographically or phonetically similar, capturing both direct and indirect forms of textual reuse.

During bridge words extraction, we encountered additional patterns. First, some pairs were written as single digits but were actually identical in meaning, such as "*3*" and "*drei*." Second, we observed word combinations with prepositions, like "*zu wissen*" and "*zuwißen*." We addressed these cases to build a more comprehensive bridge-word dictionary.

After calculating similarities for all bridge-word pairs, those exceed a chosen threshold are grouped into synonym clusters. From these clusters, we construct a dictionary mapping each variant to a canonical form. This dictionary is then used to "translate" the entire corpus, replacing synonyms with their canonical equivalents. The rationale for this translation step is to reduce the noise introduced by historical spelling variation, thereby enabling more accurate downstream similarity analysis and facilitating the identification of genuine textual relationships.

All intermediate and final results, including similarity tables, bridge word dictionaries, and translated documents, are archived in [`processed_data/flame`](./processed_data/flame/) and [`results/flame`](./results/flame/) for reproducibility. By focusing on bridge words and leveraging both orthographic and phonetic similarity, our approach overcomes the limitations of naive string matching and provides a more robust framework for tracing the complex web of textual reuse in early modern alchemical literature.

Despite the benefits of this method, it has some drawbacks. Filtering similarities of bridge words with a threshold can be too deterministic: a high threshold is stricter but may miss many synonym pairs, while a low threshold can capture more pairs but risk misjudging some. We ultimately chose 0.85 as our threshold. Additionally, while translating synonymous terms improves similarity analysis, it may obscure the historical uniqueness of certain words that were used only during specific periods.

<!-- First, filtering similarities of bridge words with a **threshold** might be too deterministic: a high threshold is stricter but can lose many synonym pairs; a low threshold can obtain many pairs but misjudge some of them. On the end, we choose **0.85** to be our threshold. Additionally, the use of words, although sementically identical, has its own historical timestamp. There are terms that were used only for a certain historical period. By translating them, it might wipe out its historical uniqueness. -->

### Phylogenetic analysis

To complement our text similarity investigations and gain deeper insight into the semantic structure and historical relationships among the Processus Universalis documents, we conducted a phylogenetic analysis based on the annotated XML corpus. The motivation for this approach stems from the need to move beyond surface-level textual similarity and instead inspect the evolution and transmission of ideas, recipes, and terminology with the inspiration from the *phylogenetic analysis* in biology.
<!-- within the corpus—much as one would trace the lineage of biological species. -->

Our workflow begins with the extraction of structured character data from the XML file, which encodes each document’s metadata and semantic features. Using our custom **PhyloXMLParser**, we systematically parse the XML to reconstruct a table of characters for each document. This includes both binary features (such as the presence or absence of specific keywords) and multi-valued features (such as recipe variants or locations), with careful normalization to account for descriptive variation, particularly in numerical expressions. For example, "*1a. 40–50 Tage im Wasserdampf*" and "*1a. 40–45 Tage im Wasserdampf*" are treated as representing the same recipe.

Once the character matrix is constructed, we encode it in the NEXUS (`.nex`) format, a standard in phylogenetic analysis. Each document is treated as a "taxon" and each semantic feature as a "character," resulting in a binary matrix that captures the presence, absence, or variant of each feature across the corpus. This matrix, along with a mapping file that records the meaning of each binary column, is then processed using [SplitsTree v4](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/algorithms-in-bioinformatics/software/splitstree/) to generate phylogenetic networks and trees.

The rationale for this method is twofold. First, it provides a principled way to visualize and quantify the relationships between documents, revealing clusters, lineages, and points of divergence that may correspond to historical transmission, adaptation, or innovation. Second, by encoding semantic content rather than mere textual similarity, we are able to detect deeper connections such as shared recipes or conceptual borrowings.

All results, including character tables, mapping files, binary matrices, and phylogenetic trees, are archived in [`processed_data/splitstree`](./processed_data/splitstree/) and [`results/splitstree`](./results/splitstree/). This approach, inspired by both computational linguistics and historical scholarship, enables us to trace the evolution of alchemical knowledge within the Processus Universalis corpus, offering new perspectives on originality, influence, and the dynamics of textual transmission.

Currently, our feature processing normalizes only numerical variation and does not yet address spelling variation (e.g., "*Tage*" versus "*Tagen*"), which remains a target for future work. Another important direction is the automatic extraction of character features, as the current XML character annotations were organized manually.
