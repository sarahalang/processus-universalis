"""
Proxy Pipeline: Discovering Phylogenetic Characters from Text Alone

This pipeline builds a NEXUS-compatible binary character matrix WITHOUT
using expert annotations. Characters are discovered from the text itself:

  Step 1: Spelling normalisation (Cologne phonetic + bridge-word synonyms)
  Step 2: Discover recurring content elements (shared phrases + terms)
  Step 3: Positional encoding (where in the text each element appears)
  Step 4: Stylometric profiling (Quadratic Delta clusters + spelling features)
  Step 5: Assemble NEXUS matrix + mapping + evidence files

Validation: compare against expert annotation matrix (not used in pipeline).
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('processus-universalis-graphics')

# ══════════════════════════════════════════════════════════════
# LOAD DATA (text files only — no expert annotations used in pipeline)
# ══════════════════════════════════════════════════════════════

TXT_DIR = Path('processus_prev_work/processus_universalis-main/'
               'ProcessusUniversalis_relevant-files-for-2025/'
               'txt-files-lowercase_processus')

plain_texts = {}
for f in sorted(TXT_DIR.iterdir()):
    if f.suffix == '.txt':
        m = re.search(r'E(\d+[ab]?)', f.name)
        if m:
            plain_texts[f'E{m.group(1)}'] = f.read_text(encoding='utf-8', errors='replace').strip()

text_names = sorted(plain_texts.keys())
n = len(text_names)
print(f"Loaded {n} texts: {', '.join(text_names)}")

# ══════════════════════════════════════════════════════════════
# STEP 1a: Cologne phonetic encoding
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 1a: Cologne phonetic normalisation")
print("="*70)

def cologne_phonetic(word):
    """Cologne phonetic encoding (Kölner Phonetik)."""
    word = word.lower().strip()
    if not word:
        return ''
    code = []
    prev_code = ''
    for i, ch in enumerate(word):
        before = word[i-1] if i > 0 else ''
        after = word[i+1] if i < len(word) - 1 else ''
        c = ''
        if ch in 'aeiouäöüjyàáâãåèéêëìíîïòóôõùúûýÿ':
            c = '0'
        elif ch == 'h':
            c = ''
        elif ch in 'bp':
            c = '1'
        elif ch in 'dt':
            c = '8' if after in 'csz' else '2'
        elif ch in 'fvw':
            c = '3'
        elif ch in 'gkq':
            c = '4'
        elif ch == 'c':
            c = '4' if after in 'ahkoqux' else '8'
        elif ch == 'x':
            c = '8' if before in 'ckq' else '48'
        elif ch == 'l':
            c = '5'
        elif ch in 'mn':
            c = '6'
        elif ch == 'r':
            c = '7'
        elif ch in 'szßẞ':
            c = '8'
        if c and c != prev_code:
            code.append(c)
            prev_code = c[-1] if c else ''
        elif c:
            prev_code = c[-1] if c else ''
    result = ''.join(code)
    if result:
        result = result[0] + result[1:].replace('0', '')
    return result


def tokenize(text):
    """Split text into lowercase words."""
    return re.findall(r'[a-zäöüß]+', text.lower())


# Build phonetic mapping for corpus vocabulary
corpus_vocab = set()
text_tokens = {}
for name in text_names:
    tokens = tokenize(plain_texts[name])
    text_tokens[name] = tokens
    corpus_vocab.update(tokens)

phon_map = {}  # word → phonetic code
phon_groups = defaultdict(set)  # phonetic code → set of surface forms
for word in corpus_vocab:
    if len(word) > 1:
        code = cologne_phonetic(word)
        if code:
            phon_map[word] = code
            phon_groups[code].add(word)

# Report variant groups with >1 member
variant_groups = {code: words for code, words in phon_groups.items()
                  if len(words) > 1}
print(f"Vocabulary: {len(corpus_vocab)} unique words")
print(f"Phonetic groups with >1 spelling variant: {len(variant_groups)}")
print(f"\nTop variant groups (most variants):")
for code, words in sorted(variant_groups.items(), key=lambda x: -len(x[1]))[:15]:
    print(f"  {code}: {', '.join(sorted(words)[:6])}"
          f"{'...' if len(words) > 6 else ''} ({len(words)} variants)")


# ══════════════════════════════════════════════════════════════
# STEP 1b: Bridge-word synonym detection (FLAME-inspired)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 1b: Bridge-word synonym detection")
print("="*70)

def jaro_winkler(s1, s2, p=0.1):
    """Jaro-Winkler similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0
    transpositions = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(len2, i + max_dist + 1)
        for j in range(start, end):
            if match2[j] or s1[i] != s2[j]:
                continue
            match1[i] = match2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not match1[i]:
            continue
        while not match2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3
    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


# Find bridge words: words that appear in similar contexts across texts
# but have different surface forms. We compare words at the same position
# in shared n-gram contexts.
def find_bridge_words(text_tokens, text_names, min_jw=0.85):
    """
    Find likely synonym pairs by comparing words that appear in
    similar local contexts across different texts.

    Method: for each pair of texts, find shared 3-grams (after phonetic
    normalisation). Words adjacent to these shared contexts that differ
    in surface form but are phonetically similar are candidate synonyms.
    """
    synonym_pairs = defaultdict(float)  # (word1, word2) → max jw score

    for i, name_a in enumerate(text_names):
        for j, name_b in enumerate(text_names):
            if i >= j:
                continue
            tokens_a = text_tokens[name_a]
            tokens_b = text_tokens[name_b]

            # Build phonetic 3-gram index for text B
            phon_b_idx = defaultdict(list)  # phonetic 3-gram → positions
            for pos in range(len(tokens_b) - 2):
                pg = tuple(phon_map.get(tokens_b[pos+k], tokens_b[pos+k])
                           for k in range(3))
                phon_b_idx[pg].append(pos)

            # For each phonetic 3-gram in text A, find matches in B
            for pos_a in range(len(tokens_a) - 2):
                pg_a = tuple(phon_map.get(tokens_a[pos_a+k], tokens_a[pos_a+k])
                             for k in range(3))
                if pg_a not in phon_b_idx:
                    continue
                for pos_b in phon_b_idx[pg_a]:
                    # Check words immediately before/after the shared context
                    for offset in [-1, 3]:  # word before or after the 3-gram
                        pa = pos_a + offset
                        pb = pos_b + offset
                        if pa < 0 or pa >= len(tokens_a):
                            continue
                        if pb < 0 or pb >= len(tokens_b):
                            continue
                        wa = tokens_a[pa]
                        wb = tokens_b[pb]
                        if wa == wb or len(wa) < 3 or len(wb) < 3:
                            continue
                        # Check if they are phonetically similar
                        jw_orig = jaro_winkler(wa, wb)
                        ca = phon_map.get(wa, wa)
                        cb = phon_map.get(wb, wb)
                        jw_phon = jaro_winkler(ca, cb)
                        best_jw = max(jw_orig, jw_phon)
                        if best_jw >= min_jw:
                            pair = tuple(sorted([wa, wb]))
                            if best_jw > synonym_pairs[pair]:
                                synonym_pairs[pair] = best_jw

    return synonym_pairs


print("Finding bridge-word synonyms (this may take a moment)...")
bridge_pairs = find_bridge_words(text_tokens, text_names, min_jw=0.90)
print(f"Found {len(bridge_pairs)} candidate synonym pairs (JW ≥ 0.90)")

# Build synonym clusters with STRICT constraints to prevent over-merging:
# 1. Only merge phonetic groups with codes ≥ 3 chars (short codes are ambiguous)
# 2. Only merge bridge pairs if they ALSO share a phonetic code
# 3. Skip top-50 most frequent words (function words)
# 4. Cap cluster size at 8

word_freq = Counter()
for name in text_names:
    word_freq.update(text_tokens[name])

top50_func = set(w for w, _ in word_freq.most_common(50))

parent = {}
cluster_size = defaultdict(lambda: 1)

def find(x):
    while parent.get(x, x) != x:
        parent[x] = parent.get(parent[x], parent[x])
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        new_size = cluster_size[ra] + cluster_size[rb]
        if new_size > 8:  # cap cluster size
            return False
        parent[ra] = rb
        cluster_size[rb] = new_size
        return True
    return False

# Merge phonetic groups (only codes with ≥ 3 chars, skip function words)
for code, words in phon_groups.items():
    if len(code) < 3:
        continue  # too ambiguous
    content_words = [w for w in words if w not in top50_func and len(w) >= 3]
    if len(content_words) > 1:
        # Only merge pairs with high surface similarity too
        for i, w1 in enumerate(content_words):
            for w2 in content_words[i+1:]:
                if jaro_winkler(w1, w2) >= 0.80:
                    union(w1, w2)

# Merge bridge-word pairs (only if they also share a phonetic code)
for (w1, w2), jw in bridge_pairs.items():
    if w1 in top50_func or w2 in top50_func:
        continue
    c1 = phon_map.get(w1, '')
    c2 = phon_map.get(w2, '')
    if c1 and c2 and c1 == c2:
        union(w1, w2)

clusters = defaultdict(set)
for word in corpus_vocab:
    if len(word) > 1:
        clusters[find(word)].add(word)

canon_map = {}  # surface form → canonical form
canonical_forms = {}
for root, members in clusters.items():
    if len(members) > 1:
        canon = max(members, key=lambda w: word_freq.get(w, 0))
        canonical_forms[canon] = members
        for w in members:
            canon_map[w] = canon

print(f"Synonym clusters (>1 member): {len(canonical_forms)}")
print(f"\nExample synonym clusters:")
shown = 0
for canon, members in sorted(canonical_forms.items(),
                               key=lambda x: -len(x[1])):
    if len(members) >= 2 and shown < 15:
        print(f"  {canon} ← {', '.join(sorted(members - {canon})[:8])}")
        shown += 1


def normalise_token(word):
    """Map a word to its canonical form."""
    return canon_map.get(word, word)


def normalise_tokens(tokens):
    """Normalise a token list."""
    return [normalise_token(w) for w in tokens]


# Build normalised token lists
norm_tokens = {name: normalise_tokens(text_tokens[name]) for name in text_names}

print(f"\nNormalisation reduces vocabulary from {len(corpus_vocab)} to "
      f"{len(set(w for toks in norm_tokens.values() for w in toks))} canonical forms")


# ══════════════════════════════════════════════════════════════
# STEP 2a: Discover recurring phrases (shared n-grams)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 2a: Discover recurring phrases")
print("="*70)

# Build 4-grams from normalised tokens for each text
def get_ngrams_with_pos(tokens, ng=4):
    """Return dict: ngram_tuple → list of positions."""
    result = defaultdict(list)
    for i in range(len(tokens) - ng + 1):
        gram = tuple(tokens[i:i+ng])
        result[gram].append(i)
    return result

text_ngrams = {}
for name in text_names:
    text_ngrams[name] = get_ngrams_with_pos(norm_tokens[name], ng=4)

# Find 4-grams that appear in 3+ texts
ngram_presence = defaultdict(set)  # ngram → set of text names
for name in text_names:
    for gram in text_ngrams[name]:
        ngram_presence[gram].add(name)

# Filter: require at least ONE content word (not in top-200 most frequent)
top200_norm = set(w for w, _ in Counter(
    t for toks in norm_tokens.values() for t in toks).most_common(200))

def has_content_word(gram):
    return any(w not in top200_norm for w in gram)

recurring_phrases = {gram: texts for gram, texts in ngram_presence.items()
                     if 3 <= len(texts) <= n - 1  # not universal, not unique
                     and has_content_word(gram)}

print(f"Total distinct 4-grams across corpus: {len(ngram_presence)}")
print(f"Recurring in 3+ texts (but not all): {len(recurring_phrases)}")

# Group overlapping 4-grams into contiguous phrases
# A "phrase" is a maximal sequence of overlapping 4-grams shared by the
# same set of texts. For character purposes, we use individual 4-grams
# but report grouped passages for evidence.

# For the character matrix, we need to reduce the number of characters.
# Strategy: cluster 4-grams that always co-occur (appear in exactly the
# same texts) into a single character represented by the longest passage.

ngram_by_textset = defaultdict(list)
for gram, texts in recurring_phrases.items():
    key = frozenset(texts)
    ngram_by_textset[key].append(gram)

# Each group of co-occurring 4-grams becomes one character
phrase_characters = []
for text_set, grams in ngram_by_textset.items():
    # Find the most distinctive gram (shortest text set = most informative)
    # Use the first 4-gram (alphabetically) as the representative
    rep = min(grams, key=lambda g: ' '.join(g))
    phrase_characters.append({
        'type': 'phrase',
        'label': ' '.join(rep),
        'texts': text_set,
        'all_grams': grams,
        'n_texts': len(text_set),
    })

# Sort by number of texts (most specific first)
phrase_characters.sort(key=lambda c: (c['n_texts'], c['label']))

print(f"Phrase characters (groups of co-occurring 4-grams): {len(phrase_characters)}")
print(f"\nMost specific phrases (fewest texts):")
for c in phrase_characters[:10]:
    print(f"  \"{c['label']}\" — in {c['n_texts']} texts: "
          f"{', '.join(sorted(c['texts']))}")
print(f"\nMost widespread phrases:")
for c in phrase_characters[-5:]:
    print(f"  \"{c['label']}\" — in {c['n_texts']} texts")


# ══════════════════════════════════════════════════════════════
# STEP 2b: Discover recurring content terms
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 2b: Discover recurring content terms")
print("="*70)

# Count which normalised terms appear in which texts
term_presence = defaultdict(set)
for name in text_names:
    for tok in set(norm_tokens[name]):
        term_presence[tok].add(name)

# Corpus-wide frequency for filtering
all_norm_tokens = []
for name in text_names:
    all_norm_tokens.extend(norm_tokens[name])
corpus_freq = Counter(all_norm_tokens)

# Select content terms: appear in 3–14 texts (not universal, not unique)
# Exclude the 200 most frequent words (function words / boilerplate)
top200 = set(w for w, _ in corpus_freq.most_common(200))

content_terms = {}
for term, texts in term_presence.items():
    if 3 <= len(texts) <= n - 2:  # informative range
        if term not in top200:  # not a function word
            if len(term) >= 3:  # not too short
                content_terms[term] = texts

# Build term characters
term_characters = []
for term, texts in sorted(content_terms.items()):
    term_characters.append({
        'type': 'term',
        'label': term,
        'texts': texts,
        'n_texts': len(texts),
    })

term_characters.sort(key=lambda c: (c['n_texts'], c['label']))

print(f"Content terms (in 3–{n-2} texts, not top-200): {len(term_characters)}")
print(f"\nMost specific terms (fewest texts):")
for c in term_characters[:10]:
    print(f"  \"{c['label']}\" — in {c['n_texts']} texts: "
          f"{', '.join(sorted(c['texts']))}")
print(f"\nMost widespread terms:")
for c in sorted(term_characters, key=lambda c: -c['n_texts'])[:10]:
    print(f"  \"{c['label']}\" — in {c['n_texts']} texts")


# ══════════════════════════════════════════════════════════════
# STEP 3: Positional encoding
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 3: Positional encoding")
print("="*70)

# For each term/phrase, record its normalised position in each text
def get_term_positions(tokens, term):
    """Find all positions of a term, return as fractions of text length."""
    positions = []
    for i, tok in enumerate(tokens):
        if tok == term:
            positions.append(i / max(len(tokens), 1))
    return positions


def get_phrase_positions(tokens, phrase_tuple):
    """Find position of a 4-gram phrase, return as fraction of text length."""
    positions = []
    for i in range(len(tokens) - len(phrase_tuple) + 1):
        if tuple(tokens[i:i+len(phrase_tuple)]) == phrase_tuple:
            positions.append(i / max(len(tokens), 1))
    return positions


# Compute median position for each term across all texts where it appears
term_median_pos = {}
for tc in term_characters:
    positions = []
    for name in tc['texts']:
        positions.extend(get_term_positions(norm_tokens[name], tc['label']))
    if positions:
        term_median_pos[tc['label']] = np.median(positions)

# Split terms into early/late based on median position
# Only split if the term has a consistent positional pattern
# (standard deviation < 0.25 across occurrences)
positional_characters = []
for tc in term_characters:
    all_positions = []
    text_positions = {}
    for name in tc['texts']:
        pos = get_term_positions(norm_tokens[name], tc['label'])
        if pos:
            all_positions.extend(pos)
            text_positions[name] = np.mean(pos)

    if not all_positions:
        continue

    median_pos = np.median(all_positions)
    std_pos = np.std(all_positions)

    # Only create positional split if position is consistent
    if std_pos < 0.25:
        is_early = median_pos < 0.4
        is_late = median_pos > 0.6
        if is_early or is_late:
            pos_label = "early" if is_early else "late"
            positional_characters.append({
                'type': 'position',
                'label': f"{tc['label']} ({pos_label})",
                'base_term': tc['label'],
                'position': pos_label,
                'median_pos': median_pos,
                'std_pos': std_pos,
                'texts': set(name for name, p in text_positions.items()
                            if (p < 0.5) == is_early),
                'n_texts': 0,  # filled below
            })
            positional_characters[-1]['n_texts'] = len(positional_characters[-1]['texts'])

# Filter: only keep positional characters in 3+ texts
positional_characters = [c for c in positional_characters
                         if 3 <= c['n_texts'] <= n - 2]

print(f"Terms with consistent position (std < 0.25): {len(positional_characters)}")
print(f"\nEarly-positioned terms (median < 0.4):")
for c in sorted([c for c in positional_characters if c['position'] == 'early'],
                 key=lambda c: c['median_pos'])[:8]:
    print(f"  \"{c['label']}\" — median pos {c['median_pos']:.2f}, "
          f"std {c['std_pos']:.2f}, in {c['n_texts']} texts")
print(f"\nLate-positioned terms (median > 0.6):")
for c in sorted([c for c in positional_characters if c['position'] == 'late'],
                 key=lambda c: -c['median_pos'])[:8]:
    print(f"  \"{c['label']}\" — median pos {c['median_pos']:.2f}, "
          f"std {c['std_pos']:.2f}, in {c['n_texts']} texts")


# ══════════════════════════════════════════════════════════════
# STEP 4: Stylometric profiling (Quadratic Delta)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 4: Stylometric profiling")
print("="*70)

# MFW features
all_tokens_flat = []
for name in text_names:
    all_tokens_flat.extend(text_tokens[name])  # use original tokens for MFW
vocab_counts = Counter(all_tokens_flat)

MFW = 300
mfw_list = [w for w, _ in vocab_counts.most_common(MFW)]

def compute_features(tokens, mfw_list):
    total = len(tokens)
    if total == 0:
        return np.zeros(len(mfw_list))
    counts = Counter(tokens)
    return np.array([counts.get(w, 0) / total for w in mfw_list])

features_matrix = np.array([compute_features(text_tokens[name], mfw_list)
                             for name in text_names])

# Quadratic Delta
def zscore_normalize(fm):
    means = fm.mean(axis=0)
    stds = fm.std(axis=0, ddof=0)
    stds[stds == 0] = 1
    return (fm - means) / stds

def delta_quadratic(fm):
    z = zscore_normalize(fm)
    nt = z.shape[0]
    dist = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(i+1, nt):
            d = np.sqrt(np.mean((z[i] - z[j])**2))
            dist[i,j] = dist[j,i] = d
    return dist

dist_stylo = delta_quadratic(features_matrix)
print(f"Quadratic Delta computed ({MFW} MFW)")

# Cluster into groups
condensed_stylo = squareform(dist_stylo)
Z_stylo = linkage(condensed_stylo, method='ward')

# Determine optimal number of clusters (3-5) by silhouette-like criterion
from scipy.cluster.hierarchy import fcluster
best_k = 3
for k in [3, 4, 5]:
    labels = fcluster(Z_stylo, k, criterion='maxclust')
    # Check that no cluster is too small (<2)
    counts = Counter(labels)
    if min(counts.values()) >= 2:
        best_k = k

cluster_labels = fcluster(Z_stylo, best_k, criterion='maxclust')
cluster_map = {name: int(cluster_labels[i]) for i, name in enumerate(text_names)}

print(f"Stylometric clusters (k={best_k}):")
for k in range(1, best_k + 1):
    members = [name for name in text_names if cluster_map[name] == k]
    print(f"  Cluster {k}: {', '.join(members)}")

# Create cluster characters
cluster_characters = []
for k in range(1, best_k + 1):
    members = set(name for name in text_names if cluster_map[name] == k)
    if 2 <= len(members) <= n - 2:
        cluster_characters.append({
            'type': 'cluster',
            'label': f'stylometric cluster {k}',
            'texts': members,
            'n_texts': len(members),
        })

# Detect distinctive spelling features
# Find words where some texts consistently use one variant and others use another
spelling_characters = []
# Check the top variant groups from Step 1
for code, words in variant_groups.items():
    if len(words) < 2:
        continue
    # For each pair of variants, check if they split the corpus
    words_list = sorted(words, key=lambda w: -word_freq.get(w, 0))
    if len(words_list) < 2:
        continue
    main_variant = words_list[0]
    for alt_variant in words_list[1:]:
        # Which texts use the main variant vs the alternative?
        uses_main = set()
        uses_alt = set()
        for name in text_names:
            toks = text_tokens[name]
            c_main = toks.count(main_variant)
            c_alt = toks.count(alt_variant)
            if c_main > 0 and c_alt == 0:
                uses_main.add(name)
            elif c_alt > 0 and c_main == 0:
                uses_alt.add(name)
        # Only keep if both variants are used by 3+ texts
        if len(uses_alt) >= 3 and len(uses_main) >= 3:
            spelling_characters.append({
                'type': 'spelling',
                'label': f'uses "{alt_variant}" (not "{main_variant}")',
                'texts': uses_alt,
                'n_texts': len(uses_alt),
                'main_form': main_variant,
                'alt_form': alt_variant,
            })

# Deduplicate (same text set → keep the one with most common alt form)
seen_textsets = {}
for sc in spelling_characters:
    key = frozenset(sc['texts'])
    if key not in seen_textsets or word_freq[sc['alt_form']] > word_freq[seen_textsets[key]['alt_form']]:
        seen_textsets[key] = sc
spelling_characters = list(seen_textsets.values())

print(f"\nSpelling-variant characters: {len(spelling_characters)}")
for sc in spelling_characters[:10]:
    print(f"  {sc['label']} — in {sc['n_texts']} texts: "
          f"{', '.join(sorted(sc['texts']))}")


# ══════════════════════════════════════════════════════════════
# STEP 5: Assemble binary character matrix
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 5: Assemble character matrix")
print("="*70)

# Collect all characters
all_characters = (phrase_characters + term_characters +
                  positional_characters + cluster_characters +
                  spelling_characters)

print(f"\nCharacter counts by type:")
for ctype in ['phrase', 'term', 'position', 'cluster', 'spelling']:
    ct = sum(1 for c in all_characters if c['type'] == ctype)
    print(f"  {ctype}: {ct}")
print(f"  TOTAL: {len(all_characters)}")

# Build binary matrix
matrix = np.zeros((n, len(all_characters)), dtype=int)
for j, char in enumerate(all_characters):
    for i, name in enumerate(text_names):
        if name in char['texts']:
            matrix[i, j] = 1

# Write NEXUS file
nexus_path = OUT_DIR / 'proxy_characters.nex'
with open(nexus_path, 'w', encoding='utf-8') as f:
    f.write("#NEXUS\n")
    f.write("Begin data;\n")
    f.write(f"  Dimensions ntax={n} nchar={len(all_characters)};\n")
    f.write('  Format datatype=standard symbols="01" gap=- missing=?;\n')
    f.write("Matrix\n")
    max_len = max(len(name) for name in text_names)
    for i, name in enumerate(text_names):
        bits = ''.join(str(b) for b in matrix[i])
        f.write(f"{name.ljust(max_len + 2)}{bits}\n")
    f.write(";\nEnd;\n")
print(f"\nNEXUS file: {nexus_path}")

# Write character mapping file
mapping_path = OUT_DIR / 'proxy_characters_mapping.csv'
with open(mapping_path, 'w', encoding='utf-8') as f:
    f.write("#This file maps labels (0|1) to their meaning for each character.\n")
    f.write("character;type;label_0;label_1;n_texts\n")
    for char in all_characters:
        safe_label = char['label'].replace(';', ',')
        f.write(f"{safe_label};{char['type']};absent;{safe_label};{char['n_texts']}\n")
print(f"Mapping file: {mapping_path}")

# Write evidence file (first 50 characters for readability)
evidence_path = OUT_DIR / 'proxy_characters_evidence.txt'
with open(evidence_path, 'w', encoding='utf-8') as f:
    f.write("EVIDENCE FILE: Source passages for each character assignment\n")
    f.write("=" * 70 + "\n\n")
    for j, char in enumerate(all_characters[:100]):  # first 100 for readability
        f.write(f"Character {j+1}: \"{char['label']}\" (type: {char['type']})\n")
        f.write(f"  Present in: {', '.join(sorted(char['texts']))}\n")
        if char['type'] == 'phrase':
            # Show the actual passage in each text
            phrase_tuple = tuple(char['label'].split())
            for name in sorted(char['texts']):
                tokens = norm_tokens[name]
                for pos in range(len(tokens) - len(phrase_tuple) + 1):
                    if tuple(tokens[pos:pos+len(phrase_tuple)]) == phrase_tuple:
                        # Show context (5 words before and after)
                        start = max(0, pos - 5)
                        end = min(len(tokens), pos + len(phrase_tuple) + 5)
                        context = ' '.join(tokens[start:end])
                        rel_pos = pos / len(tokens)
                        f.write(f"    {name} (pos {rel_pos:.2f}): ...{context}...\n")
                        break
        elif char['type'] == 'term':
            for name in sorted(char['texts'])[:5]:  # limit for readability
                tokens = norm_tokens[name]
                for pos, tok in enumerate(tokens):
                    if tok == char['label']:
                        start = max(0, pos - 4)
                        end = min(len(tokens), pos + 5)
                        context = ' '.join(tokens[start:end])
                        rel_pos = pos / len(tokens)
                        f.write(f"    {name} (pos {rel_pos:.2f}): ...{context}...\n")
                        break
        f.write("\n")
print(f"Evidence file: {evidence_path}")


# ══════════════════════════════════════════════════════════════
# VALIDATION: Compare against expert annotation matrix
# (This section uses expert data — NOT part of the pipeline)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("VALIDATION: Compare proxy matrix against expert annotations")
print("="*70)
print("(Expert data used ONLY for validation, not in the pipeline itself)\n")

with open('/Users/slang/claude/processus_data.json') as f:
    data = json.load(f)
texts_meta = data['texts']
categories = data['categories']
meta_by_name = {t['e_name']: t for t in texts_meta}
GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}

common_names = sorted(set(meta_by_name.keys()) & set(text_names))
nc = len(common_names)
upper = np.triu_indices(nc, k=1)

def get_group(name):
    return meta_by_name[name]['new_group']

# Expert annotation distances
def annotation_values(t):
    s = set()
    for c in categories:
        for v in t['annotations'][c]['values']:
            s.add((c, v))
    return s

anno_sets = {name: annotation_values(meta_by_name[name]) for name in common_names}
dist_anno = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        si, sj = anno_sets[common_names[i]], anno_sets[common_names[j]]
        jac = len(si & sj) / len(si | sj) if len(si | sj) > 0 else 0
        dist_anno[i,j] = dist_anno[j,i] = 1 - jac

# Proxy distances (Jaccard on the binary character matrix)
name_idx = {name: i for i, name in enumerate(text_names)}
proxy_matrix_common = matrix[[name_idx[n] for n in common_names]]
dist_proxy = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        si = set(np.where(proxy_matrix_common[i] == 1)[0])
        sj = set(np.where(proxy_matrix_common[j] == 1)[0])
        union_size = len(si | sj)
        jac = len(si & sj) / union_size if union_size > 0 else 0
        dist_proxy[i,j] = dist_proxy[j,i] = 1 - jac

# Also compute text 4-gram distances and stylometric distances for comparison
raw_ngrams = {}
for name in common_names:
    words = plain_texts[name].lower().split()
    raw_ngrams[name] = set(tuple(words[i:i+4]) for i in range(len(words)-3))

dist_text4 = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        si, sj = raw_ngrams[common_names[i]], raw_ngrams[common_names[j]]
        jac = len(si & sj) / len(si | sj) if len(si | sj) > 0 else 0
        dist_text4[i,j] = dist_text4[j,i] = 1 - jac

dist_stylo_common = dist_stylo[np.ix_([name_idx[n] for n in common_names],
                                       [name_idx[n] for n in common_names])]

# Evaluate all methods
def evaluate(label, dist_method):
    r_p, _ = pearsonr(dist_method[upper], dist_anno[upper])
    r_s, _ = spearmanr(dist_method[upper], dist_anno[upper])
    nn_agree = 0
    for i in range(nc):
        dm = dist_method[i].copy(); dm[i] = np.inf
        da = dist_anno[i].copy(); da[i] = np.inf
        if np.argmin(dm) == np.argmin(da):
            nn_agree += 1
    cond_m = squareform(dist_method)
    cond_a = squareform(dist_anno)
    Z_m = linkage(cond_m, method='ward')
    Z_a = linkage(cond_a, method='ward')
    _, cm = cophenet(Z_m, cond_m)
    _, ca = cophenet(Z_a, cond_a)
    r_c, _ = pearsonr(cm, ca)
    return r_p, r_s, nn_agree, r_c

print(f"{'Method':<30} {'Pearson r':>10} {'Spearman ρ':>11} {'NN':>6} {'Coph r':>8}")
print("-" * 69)
for label, dist in [("Proxy character matrix", dist_proxy),
                    ("Text 4-gram", dist_text4),
                    ("Quadratic Delta 300", dist_stylo_common)]:
    rp, rs, nn, rc = evaluate(label, dist)
    print(f"{label:<30} {rp:>10.3f} {rs:>11.3f} {nn:>2d}/{nc}   {rc:>7.3f}")


# Group separation
print(f"\nGroup separation ratios:")
for label, dist in [("Proxy", dist_proxy), ("Text 4-gram", dist_text4),
                    ("Quadratic Δ 300", dist_stylo_common), ("Expert", dist_anno)]:
    within = []
    between = []
    for i in range(nc):
        for j in range(i+1, nc):
            gi, gj = get_group(common_names[i]), get_group(common_names[j])
            sim = 1 - dist[i,j]
            if gi == gj:
                within.append(sim)
            else:
                between.append(sim)
    ratio = np.mean(within) / np.mean(between) if np.mean(between) > 0 else float('inf')
    print(f"  {label:<20} within={np.mean(within):.4f}, "
          f"between={np.mean(between):.4f}, ratio={ratio:.2f}×")


# ══════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════

import networkx as nx

rp_proxy, rs_proxy, nn_proxy, rc_proxy = evaluate("Proxy", dist_proxy)
rp_s, rs_s, nn_s, rc_s = evaluate("Stylo", dist_stylo_common)
rp_t4, rs_t4, nn_t4, rc_t4 = evaluate("4gram", dist_text4)


# ── FIGURE DD: Proxy vs Expert dendrograms ──

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax, (dist, title, stats) in zip(axes, [
    (dist_proxy, "Proxy Character Matrix\n(discovered from text)",
     f"r={rp_proxy:.3f}, ρ={rs_proxy:.3f}\nNN={nn_proxy}/{nc}, coph={rc_proxy:.3f}"),
    (dist_stylo_common, "Quadratic Delta (300 MFW)\n(stylometry only)",
     f"r={rp_s:.3f}, ρ={rs_s:.3f}\nNN={nn_s}/{nc}, coph={rc_s:.3f}"),
    (dist_anno, "Expert Annotations\n(reference)", None),
]):
    condensed = squareform(dist)
    Z = linkage(condensed, method='ward')
    labels = [f"{nm} ({get_group(nm)})" for nm in common_names]
    label_colors = {f"{nm} ({get_group(nm)})": GROUP_COLORS[get_group(nm)]
                    for nm in common_names}
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=50,
               leaf_font_size=9, color_threshold=0)
    for lbl in ax.get_xticklabels():
        lbl.set_color(label_colors.get(lbl.get_text(), 'black'))
        lbl.set_fontweight('bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=10)
    if stats:
        ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
                va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Gruppe {g}')
                  for g in ['I', 'II', 'III']]
axes[0].legend(handles=legend_patches, loc='upper right', fontsize=9)
fig.suptitle("Proxy Pipeline vs Expert Annotations: Dendrogram Comparison",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figDD_proxy_dendrograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFig DD saved: proxy dendrograms")


# ── FIGURE EE: Ablation analysis ──

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
char_types_list = ['phrase', 'term', 'position', 'cluster', 'spelling']
type_labels_list = ['Shared phrases', 'Recurring terms', 'Positional',
                    'Stylo clusters', 'Spelling variants']
type_colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22']
full_results = (rp_proxy, rs_proxy, nn_proxy, rc_proxy)

ablation_results = {}
for ctype in char_types_list:
    chars_sub = [c for c in all_characters if c['type'] != ctype]
    if not chars_sub:
        continue
    mat_sub = np.zeros((nc, len(chars_sub)), dtype=int)
    for j, char in enumerate(chars_sub):
        for i, nm in enumerate(common_names):
            if nm in char['texts']:
                mat_sub[i, j] = 1
    dist_sub = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(i+1, nc):
            si = set(np.where(mat_sub[i] == 1)[0])
            sj = set(np.where(mat_sub[j] == 1)[0])
            us = len(si | sj)
            jac = len(si & sj) / us if us > 0 else 0
            dist_sub[i,j] = dist_sub[j,i] = 1 - jac
    ablation_results[ctype] = evaluate(f"no {ctype}", dist_sub)

for ax_idx, metric_name in enumerate(['Pearson r', 'Spearman ρ', 'NN rate', 'Cophenetic r']):
    ax = axes[ax_idx]
    full_val = full_results[ax_idx] if ax_idx != 2 else full_results[2] / nc
    drops = []
    for ctype in char_types_list:
        if ctype in ablation_results:
            abl_val = ablation_results[ctype][ax_idx] if ax_idx != 2 else ablation_results[ctype][2] / nc
            drops.append(full_val - abl_val)
        else:
            drops.append(0)
    colors = [c if d > 0 else '#ccc' for c, d in zip(type_colors_list, drops)]
    ax.bar(range(len(char_types_list)), drops, color=colors)
    ax.set_xticks(range(len(char_types_list)))
    ax.set_xticklabels(type_labels_list, rotation=45, ha='right', fontsize=8)
    ax.set_title(metric_name, fontsize=11, fontweight='bold')
    ax.set_ylabel('Drop when removed', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5)

fig.suptitle("Ablation: How Much Does Each Character Type Contribute?",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figEE_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig EE saved: ablation")


# ── FIGURE FF: Proxy vs Expert distance scatter ──

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (label, dist, rp, rs, nn, rc) in zip(axes, [
    ("Proxy characters", dist_proxy, rp_proxy, rs_proxy, nn_proxy, rc_proxy),
    ("Text 4-gram", dist_text4, rp_t4, rs_t4, nn_t4, rc_t4),
    ("Quadratic Δ 300", dist_stylo_common, rp_s, rs_s, nn_s, rc_s),
]):
    for i in range(nc):
        for j in range(i+1, nc):
            gi, gj = get_group(common_names[i]), get_group(common_names[j])
            color = GROUP_COLORS[gi] if gi == gj else '#ccc'
            marker = 'o' if gi == gj else 'x'
            ax.scatter(dist[i,j], dist_anno[i,j], c=color, marker=marker,
                       s=20, alpha=0.6)
    ax.set_xlabel(f'{label} distance', fontsize=10)
    ax.set_ylabel('Expert annotation distance', fontsize=10)
    ax.set_title(f'{label}\nr={rp:.3f}, ρ={rs:.3f}, NN={nn}/{nc}, coph={rc:.3f}',
                 fontsize=10, fontweight='bold')
    x, y = dist[upper], dist_anno[upper]
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m * xr + b, 'k--', alpha=0.5, linewidth=1)

legend_patches = ([mpatches.Patch(color=GROUP_COLORS[g], label=f'Within Gruppe {g}')
                   for g in ['I', 'II', 'III']] +
                  [mpatches.Patch(color='#ccc', label='Between groups')])
axes[0].legend(handles=legend_patches, fontsize=7, loc='lower right')
fig.suptitle("Distance Correlation with Expert Annotations", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figFF_proxy_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig FF saved: scatter comparison")


# ── FIGURE GG: Character matrix heatmap ──

char_order = sorted(range(len(all_characters)),
                    key=lambda j: (all_characters[j]['type'], all_characters[j]['n_texts']))
matrix_sorted = matrix[:, char_order]
types_sorted = [all_characters[j]['type'] for j in char_order]

fig, ax = plt.subplots(figsize=(18, 7))
ax.imshow(matrix_sorted, aspect='auto', cmap='Blues', interpolation='nearest')
type_colormap = {'phrase': '#e74c3c', 'term': '#3498db', 'position': '#2ecc71',
                 'cluster': '#9b59b6', 'spelling': '#e67e22'}
type_bar = np.array([list(matplotlib.colors.to_rgb(type_colormap[t])) for t in types_sorted])
ax_top = fig.add_axes([ax.get_position().x0, ax.get_position().y1 + 0.01,
                        ax.get_position().width, 0.02])
ax_top.imshow(type_bar.reshape(1, -1, 3), aspect='auto', interpolation='nearest')
ax_top.set_xticks([]); ax_top.set_yticks([])
ax.set_yticks(range(n))
ax.set_yticklabels([f"{nm} ({get_group(nm)})" for nm in text_names], fontsize=9)
for lbl in ax.get_yticklabels():
    nm = lbl.get_text().split(' ')[0]
    if nm in meta_by_name:
        lbl.set_color(GROUP_COLORS[get_group(nm)])
        lbl.set_fontweight('bold')
ax.set_xlabel(f'Characters ({len(all_characters)} total)', fontsize=11)
ax.set_title('Proxy Character Matrix\n(blue = present, white = absent; top bar = character type)',
             fontsize=13, fontweight='bold')
legend_patches = [mpatches.Patch(color=type_colormap[t],
                                  label=f'{t} ({sum(1 for c in all_characters if c["type"]==t)})')
                  for t in ['phrase', 'term', 'position', 'cluster', 'spelling']]
ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
          title='Character types', title_fontsize=9)
plt.savefig(OUT_DIR / 'processus_figGG_proxy_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig GG saved: character matrix heatmap")


# ── FIGURE HH: Network visualisation of proxy relationships ──

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
sim_proxy = 1 - dist_proxy
sim_anno = 1 - dist_anno

for ax, (sim, title) in zip(axes, [
    (sim_proxy, "Proxy Pipeline Network"),
    (sim_anno, "Expert Annotation Network (reference)"),
]):
    G = nx.Graph()
    for nm in common_names:
        G.add_node(nm, group=get_group(nm))
    # Add edges for nearest neighbours and high-similarity pairs
    edges_added = set()
    for i, nm in enumerate(common_names):
        s = sim[i].copy(); s[i] = -1
        nn_idx = np.argmax(s)
        nn_nm = common_names[nn_idx]
        edge = tuple(sorted([nm, nn_nm]))
        if edge not in edges_added:
            G.add_edge(nm, nn_nm, weight=sim[i, nn_idx], edge_type='nn')
            edges_added.add(edge)
    # Also add pairs with high similarity
    thresh = np.percentile(sim[upper], 85)
    for i in range(nc):
        for j in range(i+1, nc):
            if sim[i,j] >= thresh:
                edge = tuple(sorted([common_names[i], common_names[j]]))
                if edge not in edges_added:
                    G.add_edge(common_names[i], common_names[j],
                               weight=sim[i,j], edge_type='high')
                    edges_added.add(edge)

    pos = nx.spring_layout(G, k=2.5, seed=42, weight='weight')
    node_colors = [GROUP_COLORS[get_group(nm)] for nm in G.nodes()]
    # Draw high-sim edges first (thin, dashed)
    high_edges = [(u,v) for u,v,d in G.edges(data=True) if d['edge_type'] == 'high']
    nn_edges = [(u,v) for u,v,d in G.edges(data=True) if d['edge_type'] == 'nn']
    nx.draw_networkx_edges(G, pos, edgelist=high_edges, ax=ax,
                           style='dashed', alpha=0.3, edge_color='#aaa', width=1)
    nn_weights = [G[u][v]['weight'] * 4 for u,v in nn_edges]
    nx.draw_networkx_edges(G, pos, edgelist=nn_edges, ax=ax,
                           alpha=0.6, edge_color='#555', width=nn_weights)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=500, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Gruppe {g}')
                  for g in ['I', 'II', 'III']]
axes[0].legend(handles=legend_patches, fontsize=10, loc='lower left')
fig.suptitle("Relationship Networks: Proxy vs Expert\n"
             "(solid = nearest-neighbour link, dashed = high similarity)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figHH_proxy_networks.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig HH saved: proxy networks")


# ── FIGURE II: Outlier and divergence analysis ──

# For each text, compute how much the proxy disagrees with experts
text_divergence = []
for i, nm in enumerate(common_names):
    # Rank divergence: how differently are this text's neighbours ranked?
    proxy_dists = dist_proxy[i].copy(); proxy_dists[i] = np.inf
    anno_dists = dist_anno[i].copy(); anno_dists[i] = np.inf
    rank_proxy = np.argsort(np.argsort(proxy_dists))
    rank_anno = np.argsort(np.argsort(anno_dists))
    rank_diff = np.mean(np.abs(rank_proxy - rank_anno))
    # NN agreement
    nn_match = int(np.argmin(proxy_dists) == np.argmin(anno_dists))
    # Mean absolute distance difference
    mean_dist_diff = np.mean(np.abs(dist_proxy[i] - dist_anno[i]))
    text_divergence.append({
        'name': nm, 'group': get_group(nm),
        'rank_divergence': rank_diff,
        'nn_match': nn_match,
        'mean_dist_diff': mean_dist_diff,
        'n_chars': np.sum(matrix[[name_idx[nm]]]),
    })

fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# Panel 1: Rank divergence per text
ax = axes[0, 0]
td_sorted = sorted(text_divergence, key=lambda x: -x['rank_divergence'])
names_s = [d['name'] for d in td_sorted]
bars = ax.barh(range(nc), [d['rank_divergence'] for d in td_sorted],
               color=[GROUP_COLORS[d['group']] for d in td_sorted])
for i, d in enumerate(td_sorted):
    marker = '✓' if d['nn_match'] else '✗'
    ax.text(d['rank_divergence'] + 0.1, i, f" {marker}", fontsize=11,
            va='center', color='green' if d['nn_match'] else 'red')
ax.set_yticks(range(nc))
ax.set_yticklabels(names_s, fontsize=10)
ax.set_xlabel('Mean rank divergence (proxy vs expert)', fontsize=10)
ax.set_title('Per-Text Divergence\n(✓ = correct NN, ✗ = wrong NN)', fontsize=12, fontweight='bold')
ax.invert_yaxis()

# Panel 2: Where proxy sees similarity but experts don't (and vice versa)
ax = axes[0, 1]
# Find the most divergent PAIRS
pair_divergences = []
for i in range(nc):
    for j in range(i+1, nc):
        diff = dist_proxy[i,j] - dist_anno[i,j]
        pair_divergences.append({
            'pair': f"{common_names[i]}–{common_names[j]}",
            'proxy_dist': dist_proxy[i,j],
            'anno_dist': dist_anno[i,j],
            'diff': diff,
            'abs_diff': abs(diff),
            'same_group': get_group(common_names[i]) == get_group(common_names[j]),
            'group_i': get_group(common_names[i]),
            'group_j': get_group(common_names[j]),
        })

pair_divergences.sort(key=lambda x: -x['abs_diff'])

# Plot: proxy says close but experts say far (blue) vs experts say close but proxy says far (red)
proxy_closer = [p for p in pair_divergences[:20] if p['diff'] < 0]
expert_closer = [p for p in pair_divergences[:20] if p['diff'] > 0]

top_pairs = pair_divergences[:15]
y_positions = range(len(top_pairs))
colors = ['#3498db' if p['diff'] < 0 else '#e74c3c' for p in top_pairs]
ax.barh(y_positions, [p['diff'] for p in top_pairs], color=colors, alpha=0.7)
ax.set_yticks(y_positions)
ax.set_yticklabels([p['pair'] for p in top_pairs], fontsize=8)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Distance difference (proxy − expert)\n'
              '← proxy says closer | expert says closer →', fontsize=9)
ax.set_title('Most Divergent Pairs\n(blue = proxy closer, red = expert closer)',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()

# Panel 3: Character count vs divergence
ax = axes[1, 0]
for d in text_divergence:
    ax.scatter(d['n_chars'], d['rank_divergence'],
               c=GROUP_COLORS[d['group']], s=100, edgecolors='black',
               linewidths=1, zorder=5)
    ax.annotate(d['name'], (d['n_chars'], d['rank_divergence']),
                fontsize=8, ha='left', va='bottom',
                xytext=(5, 3), textcoords='offset points')
ax.set_xlabel('Number of characters present', fontsize=10)
ax.set_ylabel('Mean rank divergence', fontsize=10)
ax.set_title('Character Richness vs Divergence\n'
             '(does having more characters help or hurt?)',
             fontsize=12, fontweight='bold')

# Panel 4: Per-group accuracy
ax = axes[1, 1]
group_stats = defaultdict(lambda: {'total': 0, 'nn_correct': 0, 'rank_divs': []})
for d in text_divergence:
    g = d['group']
    group_stats[g]['total'] += 1
    group_stats[g]['nn_correct'] += d['nn_match']
    group_stats[g]['rank_divs'].append(d['rank_divergence'])

groups = ['I', 'II', 'III']
x_pos = np.arange(len(groups))
nn_rates = [group_stats[g]['nn_correct'] / group_stats[g]['total'] for g in groups]
mean_divs = [np.mean(group_stats[g]['rank_divs']) for g in groups]

bars1 = ax.bar(x_pos - 0.15, nn_rates, 0.3,
               color=[GROUP_COLORS[g] for g in groups], alpha=0.7, label='NN accuracy')
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + 0.15, mean_divs, 0.3,
                color=[GROUP_COLORS[g] for g in groups], alpha=0.3,
                hatch='///', label='Mean rank divergence')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Gruppe {g}\n({group_stats[g]["total"]} texts)' for g in groups])
ax.set_ylabel('NN accuracy (solid)', fontsize=10)
ax2.set_ylabel('Mean rank divergence (hatched)', fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title('Per-Group Performance\n(accuracy and divergence by Gruppe)',
             fontsize=12, fontweight='bold')

fig.suptitle("Where Does the Proxy Pipeline Diverge from Expert Annotations?",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figII_divergence_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig II saved: divergence analysis")


# ── FIGURE JJ: Per-text divergence profiles ──
# For the 5 most divergent texts, show proxy vs expert similarity profiles

divergent_texts = sorted(text_divergence, key=lambda x: -x['rank_divergence'])[:5]

fig, axes = plt.subplots(1, len(divergent_texts), figsize=(5 * len(divergent_texts), 6))
if len(divergent_texts) == 1:
    axes = [axes]

for ax, dt in zip(axes, divergent_texts):
    idx = common_names.index(dt['name'])
    other_names = [nm for nm in common_names if nm != dt['name']]
    other_idx = [common_names.index(nm) for nm in other_names]

    proxy_sims = [1 - dist_proxy[idx, j] for j in other_idx]
    anno_sims = [1 - dist_anno[idx, j] for j in other_idx]

    # Sort by expert similarity
    order = np.argsort(anno_sims)[::-1]
    other_sorted = [other_names[k] for k in order]
    proxy_sorted = [proxy_sims[k] for k in order]
    anno_sorted = [anno_sims[k] for k in order]

    y = np.arange(len(other_sorted))
    ax.barh(y - 0.15, anno_sorted, 0.3, color='#e74c3c', alpha=0.7, label='Expert')
    ax.barh(y + 0.15, proxy_sorted, 0.3, color='#3498db', alpha=0.7, label='Proxy')

    # Mark NNs
    nn_proxy_idx = np.argmax([1 - dist_proxy[idx, j] for j in other_idx])
    nn_anno_idx = np.argmax([1 - dist_anno[idx, j] for j in other_idx])
    nn_proxy_nm = other_names[nn_proxy_idx]
    nn_anno_nm = other_names[nn_anno_idx]

    for k, nm in enumerate(other_sorted):
        if nm == nn_proxy_nm:
            ax.scatter(proxy_sorted[k] + 0.01, k + 0.15, marker='<', color='#3498db', s=80, zorder=5)
        if nm == nn_anno_nm:
            ax.scatter(anno_sorted[k] + 0.01, k - 0.15, marker='<', color='#e74c3c', s=80, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{nm} ({get_group(nm)})" for nm in other_sorted], fontsize=8)
    for lbl in ax.get_yticklabels():
        nm = lbl.get_text().split(' ')[0]
        lbl.set_color(GROUP_COLORS[get_group(nm)])
    ax.set_xlabel('Similarity', fontsize=9)
    ax.set_title(f"{dt['name']} ({get_group(dt['name'])})\n"
                 f"NN: proxy→{nn_proxy_nm}, expert→{nn_anno_nm}",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()

fig.suptitle("Divergence Profiles for the 5 Most Divergent Texts\n"
             "(◄ marks nearest neighbour for each method)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figJJ_divergence_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig JJ saved: divergence profiles")


# ── FIGURE KK: What makes outliers different? Shared character breakdown ──

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# For each NN disagreement, show what characters the proxy matches on
# vs what the expert matches on
nn_disagree = [(i, nm) for i, nm in enumerate(common_names)
               if text_divergence[i]['nn_match'] == 0]

for ax_idx, (idx, nm) in enumerate(nn_disagree[:6]):
    ax = axes[ax_idx // 3, ax_idx % 3]
    dp = dist_proxy[idx].copy(); dp[idx] = np.inf
    da = dist_anno[idx].copy(); da[idx] = np.inf
    proxy_nn = common_names[np.argmin(dp)]
    expert_nn = common_names[np.argmin(da)]

    # Count shared characters with proxy NN vs expert NN
    idx_p = common_names.index(proxy_nn)
    idx_e = common_names.index(expert_nn)
    text_row = matrix[name_idx[nm]]
    proxy_nn_row = matrix[name_idx[proxy_nn]]
    expert_nn_row = matrix[name_idx[expert_nn]]

    # Shared by type
    for ctype, color, label in [
        ('phrase', '#e74c3c', 'Phrases'),
        ('term', '#3498db', 'Terms'),
        ('position', '#2ecc71', 'Positional'),
        ('spelling', '#e67e22', 'Spelling'),
    ]:
        type_cols = [j for j, c in enumerate(all_characters) if c['type'] == ctype]
        if not type_cols:
            continue
        shared_proxy = sum(1 for j in type_cols if text_row[j] == 1 and proxy_nn_row[j] == 1)
        shared_expert = sum(1 for j in type_cols if text_row[j] == 1 and expert_nn_row[j] == 1)
        total = sum(1 for j in type_cols if text_row[j] == 1)
        ax.scatter(shared_proxy, shared_expert, c=color, s=100, zorder=5,
                   edgecolors='black', linewidths=1)
        ax.annotate(label, (shared_proxy, shared_expert), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

    # Diagonal line
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    ax.set_xlabel(f'Shared with proxy NN ({proxy_nn})', fontsize=9)
    ax.set_ylabel(f'Shared with expert NN ({expert_nn})', fontsize=9)
    ax.set_title(f'{nm} ({get_group(nm)})\nproxy→{proxy_nn} vs expert→{expert_nn}',
                 fontsize=10, fontweight='bold')

# Hide unused axes
for ax_idx in range(len(nn_disagree), 6):
    axes[ax_idx // 3, ax_idx % 3].set_visible(False)

fig.suptitle("NN Disagreements: What Characters Drive the Difference?\n"
             "(above diagonal = expert NN shares more; below = proxy NN shares more)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figKK_nn_disagreement_chars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig KK saved: NN disagreement character breakdown")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PIPELINE SUMMARY")
print("="*70)

print(f"""
Characters discovered from text (no expert knowledge used):
  Shared phrases (4-grams in 3+ texts):  {sum(1 for c in all_characters if c['type']=='phrase')}
  Recurring content terms:                {sum(1 for c in all_characters if c['type']=='term')}
  Positional variants:                    {sum(1 for c in all_characters if c['type']=='position')}
  Stylometric cluster membership:         {sum(1 for c in all_characters if c['type']=='cluster')}
  Spelling-variant features:              {sum(1 for c in all_characters if c['type']=='spelling')}
  ─────────────────────────────────────
  TOTAL:                                  {len(all_characters)}

Output files:
  {nexus_path}              — NEXUS matrix for SplitsTree
  {mapping_path}   — character meanings
  {evidence_path}  — source passages

Validation against expert annotations:
  Pearson r:    {rp_proxy:.3f}
  Spearman ρ:   {rs_proxy:.3f}
  NN agreement: {nn_proxy}/{nc}
  Cophenetic r: {rc_proxy:.3f}

Figures:
  Fig DD: Proxy vs expert dendrograms
  Fig EE: Ablation analysis (character type contributions)
  Fig FF: Distance correlation scatter plots
  Fig GG: Full character matrix heatmap
  Fig HH: Relationship networks (proxy vs expert)
  Fig II: Divergence analysis (per-text, per-pair, per-group)
  Fig JJ: Divergence profiles for most divergent texts
  Fig KK: NN disagreement character breakdown
""")

print("Nearest-neighbour detail (proxy vs expert):")
for i, name in enumerate(common_names):
    dp = dist_proxy[i].copy(); dp[i] = np.inf
    da = dist_anno[i].copy(); da[i] = np.inf
    nn_proxy_name = common_names[np.argmin(dp)]
    nn_anno_name = common_names[np.argmin(da)]
    match = "✓" if nn_proxy_name == nn_anno_name else "✗"
    print(f"  {name:6s} ({get_group(name):>3s}): "
          f"proxy→{nn_proxy_name:6s}  expert→{nn_anno_name:6s}  {match}")
