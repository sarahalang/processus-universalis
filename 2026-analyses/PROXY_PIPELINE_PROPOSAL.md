# An Automated Proxy for Expert Annotations

## Discovering Phylogenetic Characters from Text Alone

---

## The problem

The current phylogenetic workflow for the *Processus Universalis* corpus requires expert chemists to read each recipe and annotate 30 categories of chemical content. These annotations are expanded into a binary character matrix (~250 columns) and exported as a NEXUS file for phylogenetic analysis in SplitsTree.

This works well — but only because experts spent years building the annotation schema and months annotating 18 texts. For a new, unstudied corpus of recipe texts, we would not know:

- What the relevant categories are
- What values each category can take
- What terms to look for

The question is: **can we build a pipeline that discovers its own characters from the text itself, producing a NEXUS-compatible matrix that approximates the expert one?**

The previous version of this proposal made the mistake of using the expert vocabulary as search terms — defeating the purpose. This revised version discovers features entirely from the text.

---

## What we know works (from our evaluation)

| What we measured | Best result | What it means |
|-----------------|------------|---------------|
| Pairwise distance correlation | r = 0.73 (Quadratic Delta 300) | Automated distances capture ~73% of expert distance structure |
| Rank-order correlation | ρ = 0.78 (text 4-grams) | Rank ordering of pairs largely matches experts |
| Nearest-neighbour identification | 12/17 (text 4-grams) | Direct text overlap is the strongest signal for closest-relative detection |
| Tree topology | cophenetic r = 0.71 (Quadratic Delta) | Stylometric trees recover ~71% of expert tree structure |
| Named entity categories | r = 0.52–0.59 | Specific terms (earth types, salts, equipment) are the most text-predictable |
| Complex procedure categories | r < 0.10 | Multi-step procedures are NOT recoverable from text surface |
| FLAME bridge words | Jaro-Winkler ≥ 0.85 | Spelling variants can be clustered into canonical forms |

The critical lesson: **specific recurring terms and phrases are recoverable; abstract procedural equivalences are not.** A proxy must be honest about this boundary.

---

## The core idea: let the texts tell you what matters

Expert annotators decide in advance what to look for (30 categories, ~250 values). An automated proxy cannot do this. Instead, it must:

1. **Find what the texts share** — recurring terms, phrases, and passages across the corpus
2. **Cluster the shared material** — group spelling variants and synonyms into canonical forms
3. **Turn each cluster into a character** — does this text contain this recurring element? 1/0/?
4. **Add positional structure** — where in the text does each element appear?
5. **Add stylometric evidence** — an independent signal from function-word distributions

The result: a binary character matrix where each column represents a *discovered* recurring element, not a predetermined annotation category. The characters are emergent — they come from what the texts actually share with each other.

---

## Pipeline: step by step

### Step 1: Spelling normalisation and synonym discovery

**What it does:**

Before comparing texts, reduce the noise from Early Modern German spelling variation. Two sub-steps:

**1a. Phonetic normalisation (Cologne encoding)**

Apply the Kölner Phonetik encoding to every word in every text. This maps spelling variants to the same phonetic code:
- "saltz" / "salz" → same code
- "sendivogij" / "sendivogy" → same code
- "theil" / "teil" → same code

We already tested this and found a small but consistent improvement (Pearson r: 0.569 → 0.585; cophenetic r: 0.170 → 0.305).

**1b. Bridge-word synonym detection (FLAME approach)**

Adapted from the existing FLAME pipeline in the project repository:

- Run pairwise text comparison using 5-gram alignments (as in `params_flame.yaml`: `ngram: 5`)
- Extract "bridge words" — terms that appear between aligned passages but differ in surface form
- Compute Jaro-Winkler similarity between bridge-word pairs, both in original form and phonetic encoding
- Cluster pairs above threshold (0.85) into synonym groups
- Build a normalisation dictionary mapping each variant to a canonical form

This is exactly what the existing FLAME methodology does (see README: "bridge words are the terms or short phrases that occur between matched passages and are thus strong candidates for being semantically equivalent"). The difference is that we use the resulting normalisation dictionary not as an end in itself, but as preprocessing for character discovery.

**What the user sees:**

A synonym dictionary, fully inspectable:
```
Canonical form      Variants found
───────────────────────────────────────────
salz                saltz, saltze, salzes, saltzes
erde                erden, erdreich, erdte
destilliren         destillir, distillir, distilliren, destilliere
phiole              phiol, fiole, viole
salpeter            salpeters, salpether, sallpeter
```

Each entry can be reviewed: are these really the same word? The user can correct errors before proceeding.

**Why this step matters:**

Without normalisation, "saltz" in E2 and "salz" in E34 would be treated as different terms, producing two separate characters instead of one. The FLAME bridge-word approach is specifically designed for this problem — it finds spelling correspondences from context, not from a predefined dictionary.

---

### Step 2: Discover recurring content elements

**What it does:**

After normalisation, find the specific terms and phrases that recur across multiple texts. These become the candidate characters.

**2a. Recurring multi-word phrases (shared n-grams)**

For each pair of texts, find all shared word 4-grams (after normalisation). Group overlapping 4-grams into contiguous shared passages.

We already know this works: E34–E35 share 451 four-grams forming extended passages about specific procedures. These shared passages are not random — they reflect genuine textual descent.

But instead of using shared passages only for pairwise similarity (as we did before), we now treat them as **evidence for characters**:
- A phrase that appears in 3+ texts is a *recurring content element*
- Each such element becomes a candidate character: "does text X contain this phrase?"

**2b. Recurring single terms (corpus-wide content vocabulary)**

Beyond multi-word phrases, identify single content words that:
- Appear in at least 3 texts (not corpus-universal boilerplate)
- Do NOT appear in all texts (not function words — those are handled by stylometry)
- Are nouns, not function words (basic POS filtering using word-frequency heuristics: words appearing in 3–14 of 17 texts, excluding the top 200 most frequent words which are mostly function words)

Each such term becomes a candidate character.

**What the user sees:**

A list of discovered recurring elements, ranked by how many texts contain them:

```
Recurring element          Texts containing it    Type
──────────────────────────────────────────────────────────
"erde ausbreiten"          11/17                  phrase
"phiole hermetice"          9/17                  phrase
"kolben"                   14/17                  term
"rubinkorn"                 8/17                  term
"salpeter"                 12/17                  term
"athanor"                   6/17                  term
"sal volatile"              9/17                  phrase
"im wasserdampf"            7/17                  phrase
"3 kugeln"                  6/17                  phrase
"fette erde"                8/17                  phrase
...
```

Each element links back to the specific passages where it was found, in each text. The user can inspect: "show me every occurrence of 'rubinkorn' across all texts."

**Why this is different from keyword step detection:**

The keyword detector we tested earlier used a predefined list of keywords mapped to predefined categories. It achieved F1 = 0.748 but had terrible group separation (1.12×) because the keywords were too broad.

This step does the opposite: it discovers terms *from the corpus itself*, with no predefined categories. A term becomes a character only if the corpus evidence supports it — it recurs across multiple texts, and its distribution across texts is informative (not present in all, not present in only one).

---

### Step 3: Positional encoding — where in the text does each element appear?

**What it does:**

For each discovered element, record not just *whether* it appears but *where* in the text it appears, as a normalised position (0.0 = beginning, 1.0 = end).

**Why this matters:**

The expert annotations have implicit sequential structure — the 30 categories follow the recipe's procedural order (preface → earth → extraction → recombination → philosopher's stone). Without knowing these categories, we can approximate this structure by attending to *position*.

If "erde" appears in the first quarter of most texts and "rubinkorn" appears in the last quarter, this positional pattern is informative even without knowing what the "earth phase" or "stone phase" are.

**How it becomes characters:**

Each recurring element is split into positional variants:
- "erde (early)" — appears in first half of text
- "erde (late)" — appears in second half of text

This doubles the character count but adds structural information. Two texts that both mention "erde" early are more similar than two texts where one mentions it early and the other late — the early mention likely describes earth selection (the actual procedure), while a late mention might be metaphorical or retrospective.

**What the user sees:**

A positional heatmap showing where each recurring element tends to appear across the corpus:

```
Element          E2    E3    E11   E16   E17   ...
────────────────────────────────────────────────
erde             0.12  0.15  0.08  0.11  0.14  ...  ← consistently early
salpeter         0.25  0.30  —     0.22  0.28  ...  ← early-to-middle
phiole           0.55  0.60  0.48  0.52  0.55  ...  ← middle
rubinkorn        0.78  0.82  —     0.75  0.80  ...  ← consistently late
```

If an element has a consistent positional pattern across texts, the positional split is meaningful. If its position varies wildly, the split is unreliable and should be flagged.

---

### Step 4: Stylometric profiling

**What it does:**

Compute Quadratic Delta at MFW 300 (best overall performer at r = 0.73 and cophenetic r = 0.685) and produce a distance matrix and dendrogram.

**How it relates to the character matrix:**

The stylometric tree provides an *independent check* on the character-based tree. It also contributes a small number of additional characters:

- **Cluster membership:** Run hierarchical clustering on the stylometric distance matrix and cut at a level that produces 3–5 groups. Each group becomes a character: "belongs to stylometric cluster A" = 1/0.
- **Distinctive function word patterns:** If a subset of texts shares a distinctive spelling of "und" vs "undt" (the top discriminating feature we found), this becomes a character.

These are qualitatively different from the content-based characters of Steps 2–3: they reflect *scribal tradition* rather than *recipe content*. The phylogenetic analysis benefits from having both types.

**What the user sees:**

A dendrogram with cluster labels, plus a report of the most discriminating function words:

```
Most discriminating features (Quadratic Delta, 300 MFW):
  Feature 1: "und" vs "undt" — separates cluster A from cluster B
  Feature 2: "cz" (chemical symbol marker) — high in cluster C
  Feature 3: "die" frequency — distinguishes subgroups within cluster A
```

---

### Step 5: Assemble the binary character matrix

**What it does:**

Combine all discovered characters into a single binary matrix and export as NEXUS.

**Character types in the matrix:**

| Source | What it represents | Example character | How assigned |
|--------|-------------------|-------------------|-------------|
| Step 2a | Shared phrase | "phiole hermetice sigilliren" | 1 if text contains this phrase (after normalisation) |
| Step 2b | Recurring term | "rubinkorn" | 1 if text contains this term |
| Step 3 | Positional variant | "erde (early)" | 1 if term appears in first half |
| Step 4 | Stylometric cluster | "cluster A member" | 1 if text belongs to this cluster |
| Step 4 | Spelling convention | "uses 'undt' not 'und'" | 1 if text predominantly uses this form |
| Step 1 | Synonym group membership | "uses 'saltz' variant" | 1 if text uses this specific variant (not just canonical) |

**NEXUS output:**

The matrix is written in standard NEXUS format, identical to what `phylogenetics.py` produces. It can be loaded directly into SplitsTree.

```nexus
#NEXUS
Begin data;
  Dimensions ntax=17 nchar=312;
  Format datatype=standard symbols="01" gap=- missing=?;
Matrix
E2-g1a9       010110100110...
E3-g1a21      011110100100...
E11-g1a26     000010010010...
...
;
End;
```

**Alongside the NEXUS file**, two companion files:

1. **Character mapping** (same format as the expert `characters_mapping.csv`):
```
character;label_0;label_1
erde_early;absent;term "erde" found in first half of text
erde_late;absent;term "erde" found in second half of text
phiole_hermetice;absent;phrase "phiole hermetice" found (after normalisation)
cluster_A;not member;member of stylometric cluster A
...
```

2. **Evidence file** linking each 1 to the source passage:
```
E2, "erde_early" = 1:
  Line 14: "soll mann etliche zentner solcher guten erden ausgraben"

E34, "phiole_hermetice" = 1:
  Line 87: "die phiole hermetice sigilliren und in den athanor setzen"
```

---

## Why this pipeline makes sense (grounded in what we measured)

### It uses what works best

| Pipeline step | What it exploits | Evidence from our evaluation |
|--------------|-----------------|------------------------------|
| Synonym normalisation | Phonetic encoding + FLAME bridge words | Phonetic encoding improved cophenetic r from 0.170 → 0.305 |
| Shared phrase discovery | Word 4-grams | Best NN agreement (71%), best Spearman ρ (0.78) |
| Recurring term discovery | Corpus-wide content vocabulary | Named-entity categories are the most text-predictable (r = 0.52–0.59) |
| Positional encoding | Phase structure | Text-annotation r declines from 0.56 (early) to 0.48 (late); position matters |
| Stylometric profiling | Function word distributions | Best tree topology (cophenetic r = 0.71); "und/undt" is the top discriminator |

### It avoids what doesn't work

| What we avoid | Why | Evidence |
|--------------|-----|----------|
| Predefined keyword categories | Circular; assumes knowledge that wouldn't exist for new corpus | — |
| Keyword step detection as characters | Poor group separation (1.12×); too coarse | Figure R, pipeline results |
| Single-method-produces-one-number | Unexplainable, unverifiable | User requirement |
| Claiming to detect procedural equivalences | r < 0.10 for complex procedures; can't be done from text surface | Figure M |

### It honestly acknowledges its limits

The pipeline will NOT replicate the expert annotation for:
- **Complex procedural categories** (Weiterverarbeitung, Zusammenfügung) — these describe multi-step operations that scribes routinely paraphrased. No text-surface method can detect that two differently-worded passages describe the same chemistry.
- **Rare features** appearing in only 1–2 texts — by definition, these cannot be "discovered" as recurring elements.
- **Meaning-level equivalences** — when two texts describe the same procedure in completely different words (E11 and E22 share zero 4-grams despite being expert-identified nearest neighbours), no text reuse method can connect them.

These gaps mean the proxy will produce a sparser matrix than the expert one. But a sparse, honest matrix (with `?` for uncertain characters and evidence links for every `1`) is more useful than a dense, unjustified one.

---

## Validation (using the corpus where we have expert annotations)

Because we have the expert NEXUS matrix for this corpus, we can measure exactly how good the proxy is before applying it to a new corpus.

### Test 1: Character recovery rate

For each expert character (e.g., "Art der Erde: fette schwarze Erde"), check whether the proxy discovered a corresponding character. Two types of match:
- **Exact match:** proxy discovered a character that maps 1:1 to the expert character
- **Partial match:** proxy discovered a broader or narrower character (e.g., just "erde" vs the specific "fette schwarze erde")

Report: how many of the ~250 expert characters were recovered, partially recovered, or missed entirely?

### Test 2: Distance matrix correlation

Compute Jaccard distances from the proxy matrix and compare against expert distances. Targets (from our best automated results): Pearson r ≥ 0.73, Spearman ρ ≥ 0.76, NN agreement ≥ 10/17.

### Test 3: Tree topology

Run both matrices through SplitsTree (or Ward's linkage). Compare with cophenetic correlation. Target: ≥ 0.71.

### Test 4: Ablation

Remove each character type (phrases, terms, positional variants, stylometric clusters) and re-run. Which type contributes most?

### Test 5: Expert review

Present the evidence files to domain experts. For each proxy character, ask: "Is this a meaningful distinction for understanding recipe transmission?" This qualitative check prevents the pipeline from discovering statistically recurring but semantically meaningless features (e.g., "uses the word 'und' more than 50 times" — recurring but uninformative).

---

## How this connects to the existing codebase

| Existing code | What it provides | How the proxy pipeline uses it |
|--------------|-----------------|-------------------------------|
| `phylogenetics.py` | NEXUS export format, binary matrix construction, character mapping | The proxy pipeline outputs in the same format, so the same SplitsTree workflow applies directly |
| FLAME (`params_flame.yaml`) | Bridge-word synonym detection, Jaro-Winkler similarity, phonetic comparison | Step 1b uses the same methodology for spelling normalisation |
| Cologne phonetic encoding (from `text_reuse_analysis.py`) | Phonetic normalisation for Early Modern German | Step 1a uses this as a first-pass normalisation |
| `text_reuse_analysis.py` | Word 4-gram detection, Jaccard similarity | Step 2a uses the same shared-passage detection |
| `delta_comparison.py` | Quadratic Delta implementation | Step 4 uses this for stylometric profiling |
| `processus_data.json` | Expert annotations (for validation only) | Used in validation tests, never in the pipeline itself |

The pipeline is not built from scratch — it assembles existing, tested components into a new workflow.

---

## What makes this different from just running stylometry

One might ask: if Quadratic Delta already achieves r = 0.73 and cophenetic r = 0.71, why build a more complex pipeline?

Three reasons:

**1. Explainability.** A stylometric distance of 0.847 between E2 and E17 tells you they are somewhat dissimilar. It does not tell you *why*. The proxy matrix tells you: "E2 has 'erde ausbreiten' and E17 does not; E17 has 'athanor' and E2 does not; both have 'salpeter' but at different positions." Each character is a specific, inspectable claim about what the text contains.

**2. Compatibility with phylogenetic methods.** Stylometry produces a distance matrix. Phylogenetic methods like parsimony, neighbor-joining, and Bayesian inference require a *character matrix*. Distance-based trees and character-based trees can tell different stories — and the comparison between them is itself informative (as we showed throughout our analysis).

**3. Cumulability.** When experts later annotate some of the texts, their annotations can be added as additional characters in the same NEXUS matrix, alongside the automatically discovered ones. The two types of evidence combine naturally. A pure distance number cannot be combined with expert knowledge in this way.

---

## Summary

The pipeline discovers phylogenetic characters from text by:

1. **Normalising spelling** via phonetic encoding and FLAME-style bridge-word detection
2. **Finding what recurs** — shared phrases and terms across the corpus (not predefined by experts)
3. **Adding where it recurs** — positional encoding approximates procedural phases
4. **Cross-checking with stylometry** — an independent signal from function-word distributions
5. **Outputting a NEXUS character matrix** — same format as the expert pipeline, directly usable in SplitsTree

Every character traces back to a specific text passage. Every uncertain assignment is flagged. The user can inspect, correct, and extend the matrix. And because we have expert annotations for this corpus, we can validate the proxy rigorously before applying it to a new one.

The proxy will not replicate the 30% of expert information that comes from chemical domain knowledge — the meaning-level equivalences that only an expert can see. But it will capture the 70% that lives in the text itself, structured in a way that is verifiable, explainable, and compatible with established phylogenetic methods.

---

*Revised proposal for an automated proxy pipeline. Based on results from the method evaluation (`METHOD_EVALUATION.md`), the existing phylogenetic code (`phylogenetics.py`), and the FLAME text-reuse methodology. To be implemented as `proxy_pipeline.py`.*
