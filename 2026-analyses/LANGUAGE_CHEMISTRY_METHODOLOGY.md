# Language vs Chemistry Divergence: Full Methodology

This document explains *exactly* what was measured, what data was used, how conclusions were reached, and — critically — what steps were taken to avoid merely confirming a prior expectation. It accompanies the graphics in `LANGUAGE_CHEMISTRY_DIVERGENCE.md` (Figures LL–RR) and the script `language_chemistry_divergence.py`.

---

## Table of Contents

1. [The Research Question](#1-the-research-question)
2. [The Data](#2-the-data)
3. [The Method: Vocabulary Classification](#3-the-method-vocabulary-classification)
4. [The Word Lists: What Was Included and Why](#4-the-word-lists-what-was-included-and-why)
5. [Ambiguous Cases: What Was Deliberately Excluded](#5-ambiguous-cases-what-was-deliberately-excluded)
6. [How Position Was Measured](#6-how-position-was-measured)
7. [The Transition Point Algorithm](#7-the-transition-point-algorithm)
8. [Bias Checks: How We Tested for Confirmation Bias](#8-bias-checks-how-we-tested-for-confirmation-bias)
9. [What the Results Actually Show (and Don't Show)](#9-what-the-results-actually-show-and-dont-show)
10. [Limitations and Caveats](#10-limitations-and-caveats)
11. [Glossary for Non-Specialists](#11-glossary-for-non-specialists)

---

## 1. The Research Question

The starting hypothesis was: *alchemical recipe texts begin with executable chemistry and end with theoretical/philosophical content that cannot be practically carried out.* Specifically, the late sections of these recipes describe the *opus magnum* — the stages of the philosopher's stone — using language rooted in alchemical tradition (color sequences, multiplication claims, transmutation of base metals) rather than in practical laboratory procedures.

The question is whether this shift can be *measured* in the text itself, or whether it is only an impression formed by expert reading. And crucially: is the shift really there in the data, or would we find it regardless because of how we set up the measurement?

### In plain language

These are 17 versions of an alchemical recipe from the 17th–18th century. The question is: do they start out describing real chemistry (dissolve this, heat that, filter the result) and then shift towards philosophical claims (the stone will turn lead into gold, cure all diseases, multiply its power infinitely)? And can we detect this shift by counting words, rather than relying on a human reader's interpretation?

---

## 2. The Data

**Source**: 17 transcribed recipe texts in Early New High German and Latin, stored as lowercase `.txt` files in `processus/processus_prev_work/processus_universalis-main/ProcessusUniversalis_relevant-files-for-2025/txt-files-lowercase_processus/`.

**Text lengths** range from 256 words (E3, the shortest) to 3,288 words (E16, the longest). Mean length: 1,564 words.

**Pre-processing**: Each text file is read as-is (already lowercased by the project). Words are extracted by matching the regex `[a-zäöüß\-]+`, which captures German and Latin vocabulary including umlauts and hyphenated compounds. No lemmatisation, stemming, or spelling normalisation is applied — each surface form is matched exactly against the word lists. This is a deliberate choice: normalisation could introduce ambiguity (merging practical and theoretical uses of the same root), and the word lists already include common spelling variants by hand.

**What's in the text files**: The files contain the recipe text preceded by metadata lines (manuscript identifier, title). These opening lines are included in the word count, which means the very first few percent of each text contain metadata rather than recipe content. This slightly inflates non-recipe vocabulary at position 0%, but the effect is small (metadata is typically 10–20 words in a 1,000+ word text).

---

## 3. The Method: Vocabulary Classification

The core method is **word-level categorical classification**: every word in every text is checked against four manually curated word lists. If the word appears in a list, it receives that category's label. If it appears in no list, it is left unclassified.

The four categories are:

| Category | Purpose | List size | Corpus hits |
|----------|---------|-----------|-------------|
| **Practical chemistry** | Laboratory operations, equipment, substances, measurements | 128 forms (105 found in corpus) | 638 total occurrences |
| **Theoretical/philosophical** | Alchemical philosophy, cosmology, transmutation claims, religious | 104 forms (87 found) | 447 total occurrences |
| **Procedural markers** | Imperative action verbs (nimm, thu, setz, gib, laß...) | ~45 forms | Used in some figures, not in the core practical/theoretical comparison |
| **Color stages** | The alchemical colour sequence (schwartze, weisse, gelbe, grüne, rothe) | ~20 forms | Used for Figure PP specifically |

### Critical fact: classification coverage is very low

**Only 4.1% of all words** in the corpus receive a practical or theoretical label. The remaining 95.9% are unclassified. This means the analysis is based on a narrow signal — 1,085 classified word occurrences out of 26,581 total words.

This is both a limitation and a safeguard:
- **Limitation**: the analysis can only see what the word lists capture. If important practical or theoretical terms are missing, the analysis is blind to them.
- **Safeguard**: because so few words are classified, the word lists cannot accidentally dominate the measurement. The 95.9% of unclassified words act as "noise" — they dilute any signal, meaning any pattern that *does* emerge must be driven by consistent, repeated use of the classified terms.

---

## 4. The Word Lists: What Was Included and Why

### Practical chemistry list (128 forms)

The practical list was constructed by asking: **"Could a person follow this instruction in a laboratory?"** A word qualifies if it refers to:

- **Specific equipment** that you could hold or build: *retorte* (retort), *kolben* (flask), *alembic/alembico/alembicum* (alembic), *tiegel* (crucible), *recipiente* (receiver), *vorlage* (receiver vessel), *ofen* (furnace), *capelle* (cupellation dish), *mörser* (mortar)
- **Specific substances** that you could obtain and measure: *salpeter* (saltpeter/niter), *vitriol*, *antimon/antimonium* (antimony), *phlegma* (watery distillate), *lauge* (lye), *asche* (ash), *regenwaßer* (rainwater)
- **Specific operations** with clear procedures: *destilliren* (distill), *filtriren* (filter), *calciniren* (calcine), *sublimiren* (sublimate), *solviren* (dissolve), *coaguliren* (coagulate), *evaporiren* (evaporate), *rectificiren* (rectify), *sieden* (boil), *schmelzen* (melt), *glüen* (glow/calcine)
- **Measurement units**: *pfund* (pound), *lb*, *loth* (a weight unit ~14-17g), *gran* (grain), *maas* (a volume measure)
- **Practical sealing/preparation terms**: *verlutirt/lutirt* (sealed with lute), *herüber* (passing over, in distillation)

Each term was included with its common spelling variants, since the texts span multiple scribes and centuries.

### Theoretical/philosophical list (104 forms)

The theoretical list was constructed by asking: **"Does this word refer to a concept, belief, or claim rather than an observable action?"** A word qualifies if it refers to:

- **Alchemical tradition**: *philosophorum* (of the philosophers), *tinctur/tinctura* (tincture — the transformative substance), *lapis* (stone, as in philosopher's stone), *quintum esse* (fifth essence)
- **Cosmological concepts**: *himmlisch* (heavenly), *gestirn* (stars, as cosmic influence), *einflüsse/influenzen* (cosmic influences), *elementen* (elements in the philosophical sense), *centrum* (centre of nature), *fundamentum* (foundation of being)
- **Religious/devotional language**: *amen*, *gloria*, *laus* (praise), *deo* (to God), *soli* (only, in *soli deo gloria*)
- **Transmutation claims**: *multiplication/multiplicatio* (increasing the stone's power), *fermentatio/fermentation* (combining tincture with gold), *projection* (throwing tincture onto base metal), *verwandeln* (transform), *tingiren* (tinge/transmute)
- **Universal power claims**: *universale/universalis* (universal), *kranckheiten/krankheiten* (all diseases), *gesundheit* (health), *reichtumb* (wealth), *königlich* (royal)
- **Superlatives expressing perfection**: *allerhöchste* (highest of all), *allerfeinste* (finest of all), *allerköstlichste* (most precious of all)
- **Alchemical stage names** (the traditional sequence): *putrefaction/putrefactio*, *nigredo*, *albedo*, *citrinitas*, *rubedo*
- **Metaphorical language**: *wiedergeburth* (rebirth), *verjüngen* (rejuvenate), *auferstehen* (resurrect), *lebendiges* (living, as in "living gold")

### How the lists were built

The lists were constructed **before looking at the positional distribution** — that is, I did not look at where words fell in the texts and then assign them to categories accordingly. The classification criterion was semantic (is this word practical or theoretical in meaning?) not positional (does this word appear early or late?).

However, the lists were built *after* reading several of the recipe texts, which means they are informed by the content. This is unavoidable — you cannot classify vocabulary without understanding the texts — but it does mean the analyst's expectations could influence which words were selected. This concern is addressed directly in Section 8 (Bias Checks).

---

## 5. Ambiguous Cases: What Was Deliberately Excluded

Several high-frequency words were **deliberately left unclassified** because they appear in both practical and theoretical contexts and cannot be disambiguated by surface form alone:

| Word | Practical use | Theoretical use | Decision |
|------|--------------|-----------------|----------|
| *gold/silber/kupfer/blei/zinn* | Specific metals being processed | Metals being "transmuted" by the stone | **Excluded** — appears throughout |
| *feuer* (fire) | Heating source, furnace operation | Element, philosophical principle | **Excluded** — too ambiguous |
| *wasser* (water) | Solvent, rainwater for extraction | Element, "water of life", philosophical | **Excluded** — too ambiguous |
| *saltz/sal* (salt) | Specific substance being extracted | *Sal philosophicum*, philosophical principle | **Excluded** — too ambiguous |
| *erden/erde* (earth) | Material being dug up and processed | *Terra virginea*, philosophical concept | **Excluded** — too ambiguous |
| *geist/spiritus* (spirit) | Spirit of niter (a real acid) | *Spiritus mundi* (spirit of the world) | **Excluded** — too ambiguous |
| *athanor* | A specific furnace design | A term loaded with philosophical meaning | **Excluded** — appears in both phases |

This exclusion list is important for the integrity of the analysis. These words are among the most frequent in the corpus. Including them in either list would dramatically increase classification coverage but at the cost of accuracy — *gold* classified as "practical" would artificially inflate practical density in the late sections where the projection step uses gold, while classifying it as "theoretical" would do the opposite.

Additionally, some terms in the lists have **debatable placement**:

- **`ofen` in practical**: The word means "furnace" — practical equipment — but the athanor is the specific furnace used in the philosophical opus magnum, so its late-text appearances serve the theoretical section.
- **`putrefaction` in theoretical**: Putrefaction is a real chemical process (decomposition), but in these texts it specifically refers to the nigredo stage of the opus magnum, a philosophical concept. Placing it in the theoretical list is a judgment call.
- **`materia` in theoretical**: Could mean simply "material" (practical) or *prima materia* (philosophical). In this corpus it almost always appears in the phrase *prima materia* or in philosophical contexts.
- **`universale` in theoretical**: Always part of *menstruum universale* — a name for the prepared solvent. The name itself is a philosophical claim (that the solvent is "universal"), but the substance is also used practically. Classified as theoretical because the claim of universality is aspirational.
- **`verwandeln` (transform) in theoretical**: Could describe an ordinary chemical transformation, but in these texts it overwhelmingly refers to transmuting base metals into gold.
- **`esse` in theoretical**: Part of *quintum esse* (fifth essence), but the Latin word *esse* also means "to be" and can appear in other contexts.

These debatable cases affect about 15–20 classified word occurrences. Moving them between lists would not change the overall pattern significantly (see Diagnostic 6 in Section 8).

---

## 6. How Position Was Measured

Every word in a text has a **position** expressed as a fraction from 0.0 (first word) to 1.0 (last word). This normalises texts of different lengths to a common scale.

Two complementary approaches were used:

### Sliding window analysis (Figures LL, NN, QQ)
A window of fixed size (the larger of 50 words or 1/N of the text, where N is typically 20–40) slides across the text in steps. At each position, the fraction of words in the window belonging to each category is computed. This produces a smooth density curve.

**Window size matters**: A small window (50 words) gives noisy but localised signals; a large window (1/10 of text) gives smoother curves but can blur boundaries. The default of 1/20 to 1/25 of text length was chosen as a compromise. The qualitative conclusions are robust to window sizes from 1/10 to 1/40.

### Quintile segmentation (Figures MM, OO)
Each text is divided into five equal segments (0–20%, 20–40%, 40–60%, 60–80%, 80–100%), and word counts per category are computed for each segment. This is cruder than sliding windows but easier to compare across texts and immune to smoothing artefacts.

---

## 7. The Transition Point Algorithm

The "transition point" reported for each text is defined as the position where theoretical vocabulary *sustainably* overtakes practical vocabulary. The algorithm works as follows:

1. Compute the practical-minus-theoretical word count in each sliding window across the text
2. Starting from position 30% (to skip the cosmological preamble that opens most recipes), look for the **last** point where this balance crosses from positive (practical dominant) to negative (theoretical dominant)
3. If no such crossing exists after 30%, check whether the final 20% of the text is nonetheless theoretically dominant (theo > 1.5 × practical), and if so, walk backwards from the end to find where the shift begins

The 30% skip is important: many recipes open with a philosophical introduction (*"Es ist erstlichen zu wissen, das die erde aller dinge saahmen..."* — "It is first to know that the earth contains the seeds of all things...") before the practical recipe begins. Without the skip, the transition point would register at 5–10% for most texts, which would be misleading — it would be detecting the *end* of the preamble, not the *beginning* of the theoretical closing.

### Why "last" crossing, not "first"?

Some texts have brief theoretical interludes in the middle (e.g., a parenthetical remark about the philosophical significance of a step). The *last* crossing captures the sustained final shift, not a momentary digression.

### In plain language

The transition point answers: "If you were reading the recipe from start to finish, at what point would you notice that the text has stopped giving you practical instructions and started talking about philosophy?" The algorithm approximates what a reader would perceive — not the first theoretical word, but the point where the theoretical language starts to dominate and stays dominant until the end.

---

## 8. Bias Checks: How We Tested for Confirmation Bias

The central concern is **confirmation bias**: the hypothesis predicts that theoretical language appears late, so the analyst might (consciously or not) construct word lists that place "late words" in the theoretical category and "early words" in the practical category. If so, the positional pattern would be an artefact of list construction rather than a feature of the texts.

Seven diagnostic tests were run to address this concern:

### Diagnostic 1: Classification coverage

**Result**: Only 4.1% of all words are classified. This means the analysis rests on a narrow, specific signal rather than a broad categorisation of the text. If the word lists were constructed to capture many words at specific positions, coverage would be much higher. The low coverage indicates that the lists are restrictive — they only include words with clear, unambiguous meanings.

**Implication**: The low coverage also means the analysis has limited statistical power. Any pattern it finds must be driven by repeated, consistent placement of categorised terms, not by one or two words in convenient positions.

### Diagnostic 2: Which list terms actually appear?

**Result**: 105 of 128 practical forms and 87 of 104 theoretical forms were found in the corpus. The "missing" forms are mostly spelling variants that don't happen to occur in these particular manuscripts (e.g., *calciniert* — a specific inflection of "calcine" — vs *calcinir*, which does occur).

**Implication**: The lists are not over-padded with phantom terms. Most listed forms are genuinely present.

### Diagnostic 3: Ambiguous terms (discussed in Section 5)

A detailed accounting of terms that could reasonably belong in either list. The most important exclusions (gold, water, fire, salt, earth, spirit) are all high-frequency words that would dwarf the current signal if included. Their exclusion is conservative — it reduces the analysis's power but protects its validity.

### Diagnostic 4: Permutation test (the most important check)

**Method**: For each text, compute the mean position of all practical words and the mean position of all theoretical words. The observed difference (theoretical mean − practical mean) measures whether theoretical words genuinely appear later. Then, randomly shuffle the category labels 10,000 times and recompute the difference each time. The p-value is the fraction of random shuffles that produce a difference as large as or larger than the observed one.

**Results**:

| Text | Practical mean pos | Theoretical mean pos | Difference | p-value | Significant? |
|------|-------------------|---------------------|------------|---------|-------------|
| E22 | 0.457 | 0.840 | +0.384 | <0.0001 | *** |
| E17 | 0.206 | 0.729 | +0.522 | <0.0001 | *** |
| E27 | 0.238 | 0.653 | +0.416 | <0.0001 | *** |
| E45 | 0.273 | 0.534 | +0.261 | 0.0003 | *** |
| E16 | 0.513 | 0.650 | +0.137 | 0.0004 | *** |
| E32b | 0.457 | 0.619 | +0.162 | 0.0024 | ** |
| E38 | 0.344 | 0.557 | +0.214 | 0.0131 | * |
| E44 | 0.394 | 0.523 | +0.129 | 0.0338 | * |
| E11 | 0.575 | 0.699 | +0.123 | 0.1275 | n.s. |
| E35 | 0.440 | 0.486 | +0.046 | 0.2150 | n.s. |
| E19 | 0.389 | 0.485 | +0.096 | 0.1017 | n.s. |
| E37 | 0.409 | 0.476 | +0.067 | 0.1925 | n.s. |
| E34 | 0.454 | 0.456 | +0.002 | 0.4893 | n.s. |
| E2 | 0.656 | 0.728 | +0.072 | 0.3862 | n.s. |
| **E39** | **0.388** | **0.346** | **−0.042** | **0.7600** | **opposite** |
| **E42** | **0.557** | **0.325** | **−0.233** | **0.9987** | **opposite** |

**Interpretation**:

- **7 of 16 testable texts** show the pattern significantly (p < 0.05): theoretical words appear later than practical words by more than chance would predict.
- **5 of those** are highly significant (p < 0.001).
- **7 texts** show the expected direction but do not reach significance — mostly because they have too few classified words for the test to have power.
- **2 texts** (E39 and E42, both Gruppe III) show the **opposite** pattern: practical words appear *later* than theoretical words.

**This is important**: the permutation test does NOT uniformly support the hypothesis. E39 and E42 are genuine counterexamples where practical chemistry vocabulary extends into the late sections. This may be because these particular manuscript versions include more detailed procedural instructions for the multiplication step. The honest conclusion is that the pattern holds for most texts but not all.

### Diagnostic 5: Swap test

**Method**: If the pattern were an artefact of how the lists were built (e.g., the "theoretical" list just happens to contain late-appearing words regardless of their meaning), then swapping the list labels should produce the same pattern — the relabelled "theoretical" words (actually practical) would still appear late.

**Result**: In 14 of 16 testable texts, practical words have a *lower* mean position than theoretical words. Swapping the labels would therefore *reverse* the pattern for those 14 texts — proving the positional separation is tied to the *content* of the categories, not an artefact.

### Diagnostic 6: Sensitivity to "easy" terms

**Method**: Remove from the theoretical list all terms that are *obviously* late by their nature — the religious closing terms (*amen, gloria, laus, deo, soli*) and the explicit transmutation-step names (*multiplicatio, fermentatio, projection*). These 12 forms are "easy" because they almost certainly appear at the end. Then check whether the remaining theoretical terms still appear later than practical terms.

**Result**: After removal, 11 of 15 testable texts still show theoretical words appearing later. However, 4 texts (E35, E34, E39, E42) now flip — their remaining theoretical vocabulary (cosmological terms, philosophical concepts) appears *earlier* than practical vocabulary. This shows that for some texts, the late-section pattern is partly driven by the religious and transmutation terms. Without them, the cosmological vocabulary is concentrated in the *opening* preamble, not the closing.

**Implication**: The pattern has two components — an early cosmological/philosophical preamble and a late transmutation/religious closing. The "U-shape" visible in Figure MM is real and is not solely driven by the closing section.

### Diagnostic 7: Excluding the preamble

**Method**: Remove the first 20% of each text (which contains the cosmological introduction) and compare the theory/practice ratio in the body (20–75%) versus the closing (75–100%).

**Result**: **Every single text** (17 of 17) shows a higher theory-to-practice ratio in the closing than in the body. This is the most robust finding — even E39 and E42, which showed the opposite pattern in the permutation test, have more theoretical vocabulary in their final quarter than in their middle section.

| Finding | Texts supporting | Texts opposing |
|---------|-----------------|----------------|
| Theoretical words appear later overall | 14/16 | 2/16 (E39, E42) |
| Statistically significant late shift | 7/16 | 2/16 |
| Late shift survives removing "easy" terms | 11/15 | 4/15 |
| Closing section more theoretical than body | **17/17** | **0/17** |

---

## 9. What the Results Actually Show (and Don't Show)

### What IS supported by the data

1. **The closing section (last 25%) of every text has a higher ratio of theoretical-to-practical vocabulary than the body.** This holds for all 17 texts without exception, including the two counterexamples from the permutation test. This is the strongest and most defensible finding.

2. **For the majority of texts (14/16), theoretical vocabulary has a later mean position than practical vocabulary.** For 7 texts this is statistically significant by permutation test.

3. **The pattern has a U-shape**: theoretical vocabulary is elevated at both the beginning (cosmological preamble) and the end (transmutation claims), with a practical-dominant middle section. This is visible in Figure MM and confirmed by Diagnostic 6.

4. **Color-stage terms cluster in the last 30% of texts.** This is shown in Figure PP and is unsurprising — the color stages describe the opus magnum, which comes after all the preparatory chemistry.

5. **Gruppe III texts maintain practical vocabulary later into the text** than Gruppe I or II. Their transition points are higher (84–87% vs 62–79%) and two of them (E39, E42) show the opposite pattern entirely. This may reflect the fact that Gruppe III texts are the most procedurally detailed versions.

6. **Multiplication and projection claims appear almost exclusively after 80% text position.** This is shown in Figure QQ (bottom-right).

### What is NOT supported (or only weakly supported)

1. **Not every text shows a clear transition point.** Some texts (E34, E37, E19) show only a mild shift that does not reach significance. The transition is a tendency, not a universal law.

2. **The pattern is not equally strong across all manuscript groups.** Gruppe II shows it most clearly (E17, E27, E22 are highly significant); Gruppe III shows it least (E39 and E42 counter it). This could mean Gruppe III texts are more consistently practical throughout, or it could mean the word lists are better tuned to Gruppe II vocabulary.

3. **We cannot prove that the late sections are "chemically implausible."** The word lists measure vocabulary, not chemistry. A word classified as "practical" (like *ofen* or *aschen*) in the late section might describe a step that is chemically impossible — but the vocabulary analysis does not assess chemical feasibility, only word choice. The term "chemical plausibility index" in Figure QQ is a metaphor for the vocabulary ratio, not a claim about actual chemistry.

4. **The analysis cannot determine whether authors *believed* their closing claims.** The shift from practical to theoretical language could reflect: (a) authors knowingly adding traditional formulas they didn't believe, (b) authors faithfully copying a tradition they did believe, or (c) a real shift in what the author *experienced* versus what they *hoped*. The text analysis cannot distinguish these.

### In plain language

The data shows that the recipe texts consistently use more "laboratory action" words in their middle sections and more "philosophical tradition" words at their ends. This pattern is real — it's not just an impression — but it's clearer in some texts than others. Two texts (E39, E42) actually have the opposite pattern overall, though even they have more philosophy in their final quarter. The strongest statement we can make is: *the closing quarter of every text in the corpus has a higher concentration of transmutation claims and philosophical vocabulary relative to practical chemistry vocabulary than the main body of the recipe.*

---

## 10. Limitations and Caveats

### The word lists are manually curated

The biggest methodological vulnerability is that the word lists were constructed by a human analyst who already knew the hypothesis. Although the classifications were based on semantic criteria (practical meaning vs theoretical meaning) rather than positional criteria (early vs late), the analyst's knowledge of the texts could unconsciously influence which words were selected. The bias checks in Section 8 mitigate but cannot fully eliminate this concern.

A stronger approach would be to have **independent annotators** classify the word lists without knowing the hypothesis, or to derive the categories from an external resource (e.g., a historical chemistry dictionary vs a philosophy of alchemy dictionary). Neither was available for this analysis.

### Low coverage means the analysis is fragile

At 4.1% coverage, the analysis is based on about 64 classified words per text on average. For the shortest texts (E2: 7 classified words, E3: 8), the analysis is effectively meaningless at the individual text level — there simply isn't enough data. The results for these texts should be interpreted with extreme caution.

### Spelling variation is handled manually, not systematically

The word lists include hand-picked spelling variants (e.g., *destilliren*, *destillirt*, *distilliren*, *destillire*...). If a variant was missed, its occurrences are invisible to the analysis. This is a systematic source of under-counting that could disproportionately affect one category if, for example, practical terms have more spelling variation than theoretical terms (or vice versa). No systematic spelling normalisation (e.g., the Cologne phonetic encoding used in the proxy pipeline) was applied here to avoid introducing the ambiguity problems that normalisation creates.

### The 30% skip in transition detection is a judgment call

The transition algorithm skips the first 30% of the text to avoid detecting the end of the opening preamble as the "transition point." This threshold (30%) was chosen based on observation of the texts — most preambles end within the first 15–20%, and 30% provides a generous buffer. But this means the algorithm is structurally unable to detect a transition before the 30% mark, even if one existed. This choice was made to answer the specific research question (where does theory *take over* at the end) rather than the more general question (where does the text shift between modes at any point).

### The analysis treats all words equally

Every classified word counts the same regardless of its context. The word *ofen* contributes equally whether it refers to "put it in the furnace to calcine for 12 hours" (practical) or "place the phial in the athanor" (the philosophical furnace). A context-sensitive analysis (using n-grams or dependency parsing) would be more accurate but also far more complex and harder to validate.

### Position normalisation hides structural differences

By normalising position to [0, 1], the analysis treats a 250-word text the same as a 3,000-word text. But in reality, short texts may skip entire sections (no preamble, no detailed multiplication instructions), while long texts may elaborate extensively on both practical and theoretical sections. The normalisation makes the texts comparable but obscures these structural differences.

---

## 11. Glossary for Non-Specialists

| Term | Meaning |
|------|---------|
| **Classification coverage** | The percentage of words in the text that match a word list. Higher coverage means the analysis "sees" more of the text, but risks including ambiguous words. |
| **Sliding window** | A fixed-size chunk of text that moves across the document from start to finish. At each position, we count how many words in the chunk belong to each category. This produces a smooth curve showing how the vocabulary composition changes across the text. |
| **Permutation test** | A statistical test that asks: "Could this pattern have arisen by chance?" We randomly shuffle the category labels (practical/theoretical) 10,000 times and check how often the random shuffle produces a pattern as extreme as what we actually observed. If it rarely does (p < 0.05), the pattern is unlikely to be accidental. |
| **p-value** | The probability of seeing a result this extreme if there were truly no pattern. A p-value of 0.001 means a 0.1% chance — very unlikely to be random. A p-value of 0.48 means a 48% chance — could easily be random. |
| **Transition point** | The position in the text where theoretical vocabulary starts to dominate over practical vocabulary and stays dominant until the end. |
| **Quintile** | One-fifth of the text. The first quintile is the opening 20%, the last quintile is the closing 20%. |
| **Jaccard similarity** | A measure of how much two sets overlap. If two texts share many of the same words, their Jaccard similarity is high. If they use mostly different words, it's low. |
| **Confirmation bias** | The tendency to find what you're looking for because of how you set up the experiment, not because it's truly there. For example, if you classify "multiplication" as theoretical because you expect it to appear late, and then find it appears late, you might be measuring your own classification rather than a real pattern. |
| **Nigredo, albedo, citrinitas, rubedo** | The four stages of the *opus magnum* in alchemical tradition: blackening (putrefaction), whitening, yellowing, and reddening. These are observed as colour changes when material is heated in a sealed vessel over weeks or months. Whether these colour changes occur in practice depends on the specific materials — some are chemically plausible, others are not. |
| **Opus magnum** | The "great work" — the final phase of alchemical practice where the prepared materials are sealed in a vessel and heated through the colour stages to produce the philosopher's stone. |
| **Athanor** | A slow-burning furnace designed to maintain constant heat over weeks. In these texts, it has a specific design with three nested spheres (*Kugeln*). The term is both practical (a real piece of equipment that could be built) and philosophical (loaded with symbolic meaning). |
| **Menstruum universale** | "Universal solvent" — the prepared mixture of salts and spirits that the recipe produces. The term *universale* is a philosophical claim that it can dissolve all metals and gemstones. |
| **Multiplication** | The claim that the philosopher's stone can be repeatedly strengthened: first 1 part transmutes 10, then 100, then 1000, and so on "without end." This is the most explicitly non-chemical claim in the recipes. |

---

## Summary of Methodological Integrity

| Question | Answer |
|----------|--------|
| Was the classification done before or after seeing the positional distribution? | Before — categories were assigned by semantic meaning, not by position |
| Could the word lists be biased? | Yes, inevitably, since they were constructed by an analyst who knew the hypothesis. Bias checks mitigate but cannot fully eliminate this |
| Is the pattern statistically significant? | For 7/16 texts individually, yes. For the closing-section comparison, 17/17 texts show the pattern. |
| Are there counterexamples? | Yes — E39 and E42 show the opposite overall pattern (practical later than theoretical) |
| Does the pattern survive removing "obviously late" terms? | For 11/15 texts, yes. For 4 texts, removing religious and transmutation terms flips the result |
| What's the strongest claim we can make? | The closing quarter of every text has more theoretical vocabulary relative to practical vocabulary than the body |
| What's the weakest link? | Low classification coverage (4.1%) and manual list construction |
