#!/usr/bin/env python3
"""
Cascading Pipeline: Methods Informing Methods
==============================================
Unlike the weighted-average approach, this pipeline uses each method's
OUTPUT as INPUT to the next step. The idea:

  Step 1: text-matcher → map which regions of each text are COPIED
          vs ORIGINAL (the scribe's own composition)
  Step 2: Compute SEPARATE distances for copied and original regions.
          Copied regions → transmission evidence (who copied from whom).
          Original regions → authorial evidence (scribal identity/style).
  Step 3: Use structural markers (where theory overtakes practice) as
          an alignment feature — texts that transition at the same point
          may share structural heritage even without shared words.
  Step 4: Build a multi-dimensional similarity profile per pair, then
          combine dimensions with interpretive logic (not just weights).

Key principle: each step fills a gap the previous step couldn't.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('processus-universalis-graphics')
OUT_DIR.mkdir(exist_ok=True)

TXT_DIR = Path("processus/processus_prev_work/processus_universalis-main/"
               "ProcessusUniversalis_relevant-files-for-2025/"
               "txt-files-lowercase_processus")

XML_PATH = Path("processus/processus_prev_work/processus_universalis-main/"
                "ProcessusUniversalis_relevant-files-for-2025/sammlung_aller_texte.xml")

GROUP_MAP = {
    'E2': 'I', 'E3': 'I', 'E11': 'I', 'E22': 'I', 'E35': 'I',
    'E16': 'II', 'E17': 'II', 'E19': 'II', 'E27': 'II', 'E32b': 'II',
    'E34': 'III', 'E37': 'III', 'E38': 'III', 'E39': 'III',
    'E42': 'III', 'E44': 'III', 'E45': 'III',
}
GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}

A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11',
}


def tokenize(text):
    return re.findall(r'[a-zäöüß]+', text.lower())


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
print("Loading texts...")
plain_texts = {}
for f in sorted(TXT_DIR.glob("*.txt")):
    m = re.search(r'(E\d+[a-z]?)', f.stem)
    if m:
        ename = m.group(1)
        if ename in GROUP_MAP:
            plain_texts[ename] = f.read_text(encoding='utf-8', errors='replace').strip()

text_names = sorted(plain_texts.keys())
n = len(text_names)
name_idx = {nm: i for i, nm in enumerate(text_names)}
text_tokens = {nm: tokenize(plain_texts[nm]) for nm in text_names}
print(f"  {n} texts loaded")

# Expert annotations
import xml.etree.ElementTree as ET
tree = ET.parse(XML_PATH)
root = tree.getroot()
anno_features = {}
for div in root.findall('div'):
    dtype = div.get('type', '')
    m = re.search(r'a(\d+)', dtype)
    if not m:
        continue
    a_key = 'a' + m.group(1)
    ename = A_TO_E.get(a_key)
    if not ename or ename not in GROUP_MAP:
        continue
    features = set()
    for keys_el in div.findall('.//keys'):
        kvals = keys_el.get('n', '')
        ktype = keys_el.get('type', '')
        if kvals and 'FEHLT' not in kvals:
            for val in kvals.split(';'):
                val = val.strip()
                if val:
                    features.add(f"{ktype}::{val}")
    if features:
        anno_features[ename] = features

common = sorted([nm for nm in text_names if nm in anno_features])
nc = len(common)
cidx = [name_idx[nm] for nm in common]

expert_dist = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        a = anno_features[common[i]]
        b = anno_features[common[j]]
        if len(a | b) > 0:
            expert_dist[i, j] = expert_dist[j, i] = 1 - len(a & b) / len(a | b)

def upper_tri(mat):
    return mat[np.triu_indices(len(mat), k=1)]

expert_flat = upper_tri(expert_dist)


# ══════════════════════════════════════════════════════════════
# STEP 1: TEXT-MATCHER → Build a copying map for each text
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: text-matcher → Identify copied vs original regions")
print("=" * 70)

from text_matcher.matcher import ExtendedMatch
from difflib import SequenceMatcher
from nltk.metrics.distance import edit_distance as editDistance


class GermanText:
    def __init__(self, raw_text, label):
        self.text = raw_text
        self.label = label
        tokenizer_pattern = re.compile(r'[a-zA-ZäöüÄÖÜß]+')
        spans_iter = list(tokenizer_pattern.finditer(self.text))
        self.spans = [(m.start(), m.end()) for m in spans_iter]
        self.tokens = [self.text[s:e].lower() for s, e in self.spans]
        self.length = self.spans[-1][-1] if self.spans else 1

    def ngrams(self, n):
        return [tuple(self.tokens[i:i+n]) for i in range(len(self.tokens) - n + 1)]


class GermanMatcher:
    def __init__(self, textA, textB, threshold=3, cutoff=5, ngram_size=3):
        self.textA = textA
        self.textB = textB
        self.ngram_size = ngram_size
        self.textAgrams = textA.ngrams(ngram_size)
        self.textBgrams = textB.ngrams(ngram_size)

        sequence = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        matching_blocks = sequence.get_matching_blocks()
        self.initial_matches = [m for m in matching_blocks if m.size > threshold]
        self.healed_matches = self._heal_neighbors()
        self.extended_matches = self._extend_matches()
        self.extended_matches = [m for m in self.extended_matches
                                  if min(m.sizeA, m.sizeB) >= cutoff]
        self.numMatches = len(self.extended_matches)

    def _heal_neighbors(self, min_distance=8):
        healed = []
        matches = self.initial_matches.copy()
        if len(matches) <= 1:
            for m in matches:
                healed.append(ExtendedMatch(m.a, m.b, m.size, m.size))
            return healed
        skip_next = False
        for i in range(len(matches)):
            if skip_next:
                skip_next = False
                continue
            match = matches[i]
            if i + 1 < len(matches):
                nxt = matches[i + 1]
                if (nxt.a - (match.a + match.size)) < min_distance:
                    sizeA = (nxt.a + nxt.size) - match.a
                    sizeB = (nxt.b + nxt.size) - match.b
                    em = ExtendedMatch(match.a, match.b, sizeA, sizeB)
                    em.healed = True
                    healed.append(em)
                    skip_next = True
                else:
                    healed.append(ExtendedMatch(match.a, match.b, match.size, match.size))
            else:
                healed.append(ExtendedMatch(match.a, match.b, match.size, match.size))
        return healed

    def _edit_ratio(self, wordA, wordB):
        distance = editDistance(wordA, wordB)
        avg_len = (len(wordA) + len(wordB)) / 2
        return distance / avg_len if avg_len > 0 else 1.0

    def _extend_matches(self, cutoff=0.4):
        extended = False
        for match in self.healed_matches:
            if match.a > 0 and match.b > 0:
                wordA = self.textAgrams[match.a - 1][0]
                wordB = self.textBgrams[match.b - 1][0]
                if self._edit_ratio(wordA, wordB) < cutoff:
                    match.a -= 1
                    match.b -= 1
                    match.sizeA += 1
                    match.sizeB += 1
                    match.extendedBackwards += 1
                    extended = True
            idxA = match.a + match.sizeA + 1
            idxB = match.b + match.sizeB + 1
            if idxA < len(self.textAgrams) and idxB < len(self.textBgrams):
                wordA = self.textAgrams[idxA][-1]
                wordB = self.textBgrams[idxB][-1]
                if self._edit_ratio(wordA, wordB) < cutoff:
                    match.sizeA += 1
                    match.sizeB += 1
                    match.extendedForwards += 1
                    extended = True
        if extended:
            self._extend_matches()
        return self.healed_matches


# Build text objects
gt_objects = {nm: GermanText(plain_texts[nm], nm) for nm in text_names}

# Run all pairs — collect match positions for each text
# For each text, build a boolean mask: which token positions are "copied" (matched
# with at least one other text) vs "original" (unique to this text)
copied_mask = {nm: np.zeros(len(gt_objects[nm].tokens), dtype=bool) for nm in text_names}

# Also store per-pair match info
pair_matches = {}
pair_count = 0
total_pairs = n * (n - 1) // 2

for i in range(n):
    for j in range(i+1, n):
        pair_count += 1
        na, nb = text_names[i], text_names[j]
        matcher = GermanMatcher(gt_objects[na], gt_objects[nb])

        match_spans_a = []
        match_spans_b = []
        for match in matcher.extended_matches:
            lenA = match.sizeA + matcher.ngram_size - 1
            lenB = match.sizeB + matcher.ngram_size - 1
            # Mark tokens as copied
            copied_mask[na][match.a:match.a + lenA] = True
            copied_mask[nb][match.b:match.b + lenB] = True
            match_spans_a.append((match.a, match.a + lenA))
            match_spans_b.append((match.b, match.b + lenB))

        pair_matches[(na, nb)] = {
            'n_matches': matcher.numMatches,
            'spans_a': match_spans_a,
            'spans_b': match_spans_b,
        }

        if pair_count % 20 == 0:
            print(f"  {pair_count}/{total_pairs} pairs done...")

print(f"  {total_pairs}/{total_pairs} pairs done.")

# Report copying statistics
print("\nCopying map per text (% of tokens that appear in at least one other text):")
for nm in text_names:
    n_tokens = len(gt_objects[nm].tokens)
    n_copied = copied_mask[nm].sum()
    pct = 100 * n_copied / n_tokens if n_tokens > 0 else 0
    print(f"  {nm} ({GROUP_MAP[nm]}): {n_copied}/{n_tokens} tokens copied ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 2: Split each text into COPIED and ORIGINAL tokens
#         Compute separate distance matrices for each
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Separate distances for copied vs original regions")
print("=" * 70)

# Extract tokens
copied_tokens = {}
original_tokens = {}
for nm in text_names:
    toks = text_tokens[nm]
    mask = copied_mask[nm]
    # Align lengths (tokenization may differ slightly)
    min_len = min(len(toks), len(mask))
    copied_tokens[nm] = [toks[i] for i in range(min_len) if mask[i]]
    original_tokens[nm] = [toks[i] for i in range(min_len) if not mask[i]]

print("\nToken split:")
for nm in text_names:
    nc_t = len(copied_tokens[nm])
    no_t = len(original_tokens[nm])
    total = nc_t + no_t
    pct = 100 * nc_t / total if total > 0 else 0
    print(f"  {nm}: {nc_t} copied + {no_t} original = {total} total ({pct:.0f}% copied)")

# ── 2a: Quadratic Delta on ORIGINAL tokens only ──
# This measures scribal style independent of what was copied
print("\nComputing Quadratic Delta on ORIGINAL tokens only...")

all_orig_tokens = []
for nm in text_names:
    all_orig_tokens.extend(original_tokens[nm])
orig_vocab = Counter(all_orig_tokens)
MFW = 200  # Use fewer MFW since original regions are shorter
mfw_list = [w for w, _ in orig_vocab.most_common(MFW)]

def compute_freq(tokens, mfw):
    total = len(tokens)
    if total < 10:  # too short for meaningful frequencies
        return None
    c = Counter(tokens)
    return np.array([c.get(w, 0) / total for w in mfw])

orig_features = {}
for nm in text_names:
    f = compute_freq(original_tokens[nm], mfw_list)
    if f is not None:
        orig_features[nm] = f

# Z-score normalize and compute Delta
valid_texts = [nm for nm in text_names if nm in orig_features]
print(f"  {len(valid_texts)}/{n} texts have enough original tokens for stylometry")

if len(valid_texts) >= 2:
    feat_matrix = np.array([orig_features[nm] for nm in valid_texts])
    means = feat_matrix.mean(axis=0)
    stds = feat_matrix.std(axis=0, ddof=0)
    stds[stds == 0] = 1
    z = (feat_matrix - means) / stds

    dist_orig_stylo_full = np.zeros((n, n))
    vt_idx = {nm: i for i, nm in enumerate(valid_texts)}
    for i in range(len(valid_texts)):
        for j in range(i+1, len(valid_texts)):
            d = np.sqrt(np.mean((z[i] - z[j])**2))
            ii = name_idx[valid_texts[i]]
            jj = name_idx[valid_texts[j]]
            dist_orig_stylo_full[ii, jj] = dist_orig_stylo_full[jj, ii] = d

# ── 2b: Quadratic Delta on FULL text (for comparison) ──
print("Computing Quadratic Delta on FULL text...")
all_tokens_flat = []
for nm in text_names:
    all_tokens_flat.extend(text_tokens[nm])
full_vocab = Counter(all_tokens_flat)
mfw_full = [w for w, _ in full_vocab.most_common(300)]
full_features = np.array([
    np.array([Counter(text_tokens[nm]).get(w, 0) / max(len(text_tokens[nm]), 1)
              for w in mfw_full])
    for nm in text_names
])
fm_means = full_features.mean(axis=0)
fm_stds = full_features.std(axis=0, ddof=0)
fm_stds[fm_stds == 0] = 1
z_full = (full_features - fm_means) / fm_stds
dist_full_stylo = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = np.sqrt(np.mean((z_full[i] - z_full[j])**2))
        dist_full_stylo[i, j] = dist_full_stylo[j, i] = d


# ── 2c: 4-gram Jaccard on ORIGINAL tokens only ──
print("Computing 4-gram Jaccard on ORIGINAL tokens only...")
orig_ngrams = {}
for nm in text_names:
    toks = original_tokens[nm]
    orig_ngrams[nm] = set(tuple(toks[i:i+4]) for i in range(max(0, len(toks)-3)))

dist_orig_4gram = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si = orig_ngrams[text_names[i]]
        sj = orig_ngrams[text_names[j]]
        u = len(si | sj)
        jac = len(si & sj) / u if u > 0 else 0
        dist_orig_4gram[i, j] = dist_orig_4gram[j, i] = 1 - jac

# Full 4-gram for comparison
full_ngrams = {}
for nm in text_names:
    toks = text_tokens[nm]
    full_ngrams[nm] = set(tuple(toks[i:i+4]) for i in range(len(toks)-3))

dist_full_4gram = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si = full_ngrams[text_names[i]]
        sj = full_ngrams[text_names[j]]
        u = len(si | sj)
        jac = len(si & sj) / u if u > 0 else 0
        dist_full_4gram[i, j] = dist_full_4gram[j, i] = 1 - jac


# ── 2d: text-matcher distance (transmission evidence) ──
print("Computing text-matcher distance (from Step 1)...")
tm_score = np.zeros((n, n))
tm_max_len = np.zeros((n, n))
for (na, nb), info in pair_matches.items():
    i, j = name_idx[na], name_idx[nb]
    total_matched = sum(e - s for s, e in info['spans_a'])
    max_match = max((e - s for s, e in info['spans_a']), default=0)
    norm = min(len(gt_objects[na].tokens), len(gt_objects[nb].tokens))
    tm_score[i, j] = tm_score[j, i] = total_matched / norm if norm > 0 else 0
    tm_max_len[i, j] = tm_max_len[j, i] = max_match

dist_tm = 1 - tm_score / (tm_score.max() + 1e-10)


# ══════════════════════════════════════════════════════════════
# STEP 3: Structural alignment features
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Structural alignment — where does theory overtake practice?")
print("=" * 70)

# Vocabulary lists (from language_chemistry_divergence.py)
CHEM_PRACTICAL = set("""
wasser feuer erde salz sal nitrum salpeter vitriol schwefel sulphur
quecksilber mercurius aqua destillir distillir calcinir sublimir
coagulir filtriren solvi dissolvir precipitir extrahir putrefactio
putrefaction fixir ferment philosophisch lösen lösung kochen sieden
digeriren schmelzen gießen seihen pressen wägen mischen
kolben glas ofen athanor retort helm tiegel phiol balneo balneum
spiritus tinctur essenz extract oleum kristall caput mortuum
saltz wasser fewer erden erdrich kalck kalch laugen asche aschen
""".split())

CHEM_THEORETICAL = set("""
stein philosophorum lapis gold silber sol luna materia prima
universalis elixir tinctura transmutation projektion quintessenz
arcanum magisterium philosophen weisen adepten
corpus anima spiritus seele leib geist
conjunction copulation hochzeit vermählung
sulphur mercurius sal principien element
gott göttlich himmlisch natur natürlich
schöpfung chaos creation universum
analogie correspondenz gleichnis
""".split())

def sliding_window_density(tokens, vocab, window_pct=0.10, step=0.02):
    """Compute density of vocab words in sliding windows across text."""
    n_tok = len(tokens)
    if n_tok < 20:
        return [], []
    window_size = max(10, int(n_tok * window_pct))
    step_size = max(1, int(n_tok * step))
    positions = []
    densities = []
    for start in range(0, n_tok - window_size + 1, step_size):
        window = tokens[start:start + window_size]
        count = sum(1 for w in window if w in vocab)
        positions.append((start + window_size / 2) / n_tok)
        densities.append(count / len(window))
    return positions, densities


def find_transition_point(tokens):
    """Find where theoretical vocabulary sustainably exceeds practical.
    Returns a position in [0, 1] or None if no clear transition."""
    pos_p, dens_p = sliding_window_density(tokens, CHEM_PRACTICAL)
    pos_t, dens_t = sliding_window_density(tokens, CHEM_THEORETICAL)
    if not pos_p or not pos_t:
        return None

    # Compute balance: positive = practical dominant, negative = theoretical dominant
    balance = [p - t for p, t in zip(dens_p, dens_t)]

    # Skip first 30% (cosmological preamble) and find last crossover
    last_crossover = None
    for idx in range(len(balance) - 1):
        if pos_p[idx] < 0.30:
            continue
        if balance[idx] >= 0 and balance[idx + 1] < 0:
            last_crossover = pos_p[idx]

    return last_crossover


transition_points = {}
for nm in text_names:
    tp = find_transition_point(text_tokens[nm])
    transition_points[nm] = tp
    if tp is not None:
        print(f"  {nm} ({GROUP_MAP[nm]}): transition at {tp:.0%}")
    else:
        print(f"  {nm} ({GROUP_MAP[nm]}): no clear transition detected")

# Structural distance: difference in transition points
dist_structural = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        tp_i = transition_points[text_names[i]]
        tp_j = transition_points[text_names[j]]
        if tp_i is not None and tp_j is not None:
            d = abs(tp_i - tp_j)
        else:
            d = 0.5  # default distance when transition can't be detected
        dist_structural[i, j] = dist_structural[j, i] = d


# ══════════════════════════════════════════════════════════════
# STEP 4: Embeddings on ORIGINAL regions only
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Embeddings on ORIGINAL (non-copied) regions")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    from numpy.linalg import norm as np_norm

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Reconstruct original text (non-copied regions)
    original_texts = {}
    for nm in text_names:
        mask = copied_mask[nm]
        gt = gt_objects[nm]
        # Build text from non-copied spans
        original_spans = []
        in_original = False
        start = None
        for idx in range(min(len(mask), len(gt.spans))):
            if not mask[idx]:
                if not in_original:
                    start = gt.spans[idx][0]
                    in_original = True
                end = gt.spans[idx][1]
            else:
                if in_original:
                    original_spans.append(gt.text[start:end])
                    in_original = False
        if in_original:
            original_spans.append(gt.text[start:end])
        original_texts[nm] = ' '.join(original_spans) if original_spans else gt.text[:50]

    # Embed original and full texts
    def chunk_and_embed(text, chunk_size=80, overlap=40):
        words = text.split()
        if len(words) < 20:
            return model.encode([text])[0:1]
        chunks = []
        for i in range(0, len(words) - chunk_size + 1, chunk_size - overlap):
            chunks.append(' '.join(words[i:i+chunk_size]))
        if not chunks:
            chunks = [text]
        return model.encode(chunks)

    emb_original = {}
    emb_full = {}
    for nm in text_names:
        embs_o = chunk_and_embed(original_texts[nm])
        embs_f = chunk_and_embed(plain_texts[nm])
        emb_original[nm] = np.mean(embs_o, axis=0)
        emb_full[nm] = np.mean(embs_f, axis=0)

    def cosine_dist_matrix(emb_dict, names):
        nn = len(names)
        dist = np.zeros((nn, nn))
        for i in range(nn):
            for j in range(i+1, nn):
                a, b = emb_dict[names[i]], emb_dict[names[j]]
                sim = np.dot(a, b) / (np_norm(a) * np_norm(b) + 1e-10)
                dist[i, j] = dist[j, i] = max(0, 1 - sim)
        return dist

    dist_emb_orig = cosine_dist_matrix(emb_original, text_names)
    dist_emb_full = cosine_dist_matrix(emb_full, text_names)

    HAS_EMB = True
    print("  Done. Embeddings computed for original and full text.")
except ImportError:
    print("  sentence-transformers not available, skipping embeddings.")
    dist_emb_orig = np.zeros((n, n))
    dist_emb_full = np.zeros((n, n))
    HAS_EMB = False


# ══════════════════════════════════════════════════════════════
# STEP 5: Cascading logic — combine with INTERPRETATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Cascading combination with interpretive logic")
print("=" * 70)

# The idea: different evidence types inform different relationship types.
#
# For a pair of texts, we now have:
#   - tm_score: how much verbatim text they share (transmission evidence)
#   - dist_orig_stylo: how different their NON-COPIED regions are (scribal style)
#   - dist_orig_4gram: phrasal overlap in NON-COPIED regions (independent composition)
#   - dist_structural: whether they transition at the same structural point
#   - dist_emb_orig: semantic similarity of NON-COPIED regions
#
# Cascading logic:
#   1. If two texts share extensive verbatim material → they are closely related
#      (text-matcher dominance). But HOW close? Check their original regions.
#   2. If two texts DON'T share much verbatim material → fall back to
#      stylometric + 4-gram + embedding distance on original regions.
#   3. Structural alignment provides an independent signal for all pairs.

# Normalize all to [0, 1]
def normalize_dist(d):
    flat = upper_tri(d)
    if flat.max() - flat.min() < 1e-10:
        return d.copy()
    return (d - flat.min()) / (flat.max() - flat.min())

d_tm_n = normalize_dist(dist_tm)
d_full_stylo_n = normalize_dist(dist_full_stylo)
d_orig_stylo_n = normalize_dist(dist_orig_stylo_full)
d_full_4gram_n = normalize_dist(dist_full_4gram)
d_orig_4gram_n = normalize_dist(dist_orig_4gram)
d_struct_n = normalize_dist(dist_structural)
if HAS_EMB:
    d_emb_orig_n = normalize_dist(dist_emb_orig)
    d_emb_full_n = normalize_dist(dist_emb_full)


# ── Cascading combination ──
# For each pair: determine the "regime" based on text-matcher score,
# then use different method mixes for different regimes.

# Determine thresholds from the data
tm_scores_flat = []
for i in range(n):
    for j in range(i+1, n):
        tm_scores_flat.append(tm_score[i, j])
tm_75th = np.percentile(tm_scores_flat, 75)
tm_90th = np.percentile(tm_scores_flat, 90)
print(f"  text-matcher score percentiles: 75th={tm_75th:.4f}, 90th={tm_90th:.4f}")

dist_cascade = np.zeros((n, n))
pair_regime = np.zeros((n, n), dtype=int)  # 0=no copy, 1=some copy, 2=heavy copy

for i in range(n):
    for j in range(i+1, n):
        score = tm_score[i, j]

        if score >= tm_90th:
            # HEAVY COPYING: these texts are near-copies.
            # text-matcher dominates, but original-region style distinguishes versions.
            d = 0.60 * d_tm_n[i, j] + 0.20 * d_orig_stylo_n[i, j] + 0.10 * d_struct_n[i, j]
            if HAS_EMB:
                d += 0.10 * d_emb_orig_n[i, j]
            pair_regime[i, j] = pair_regime[j, i] = 2

        elif score >= tm_75th:
            # MODERATE COPYING: shared tradition, but with significant original content.
            # Mix of text-matcher evidence and original-region analysis.
            d = 0.30 * d_tm_n[i, j] + 0.25 * d_orig_4gram_n[i, j] + \
                0.20 * d_orig_stylo_n[i, j] + 0.15 * d_struct_n[i, j]
            if HAS_EMB:
                d += 0.10 * d_emb_orig_n[i, j]
            pair_regime[i, j] = pair_regime[j, i] = 1

        else:
            # NO SIGNIFICANT COPYING: independent compositions.
            # Fall back entirely to stylometric + phrasal + semantic comparison.
            d = 0.35 * d_full_4gram_n[i, j] + 0.30 * d_full_stylo_n[i, j] + \
                0.20 * d_struct_n[i, j]
            if HAS_EMB:
                d += 0.15 * d_emb_orig_n[i, j]
            pair_regime[i, j] = pair_regime[j, i] = 0

        dist_cascade[i, j] = dist_cascade[j, i] = d


# ── Also compute an optimized-weight version for comparison ──
# This uses the same distances but finds optimal FIXED weights
from scipy.optimize import minimize

dist_components = {
    'tm': d_tm_n,
    'orig_stylo': d_orig_stylo_n,
    'full_stylo': d_full_stylo_n,
    'orig_4gram': d_orig_4gram_n,
    'full_4gram': d_full_4gram_n,
    'structural': d_struct_n,
}
if HAS_EMB:
    dist_components['emb_orig'] = d_emb_orig_n
    dist_components['emb_full'] = d_emb_full_n

comp_keys = list(dist_components.keys())
n_comp = len(comp_keys)
comp_stack = np.array([upper_tri(dist_components[k][np.ix_(cidx, cidx)]) for k in comp_keys])

def neg_rho(w):
    w = np.maximum(w, 0)
    s = w.sum()
    if s < 1e-10:
        return 0
    w = w / s
    return -spearmanr(w @ comp_stack, expert_flat)[0]

best_opt_score = -1
best_opt_w = None
for _ in range(10):
    start = np.random.dirichlet(np.ones(n_comp))
    res = minimize(neg_rho, start, method='Nelder-Mead', options={'maxiter': 5000})
    w = np.maximum(res.x, 0)
    if w.sum() > 0:
        w = w / w.sum()
    score = spearmanr(w @ comp_stack, expert_flat)[0]
    if score > best_opt_score:
        best_opt_score = score
        best_opt_w = w.copy()

print(f"\n  Optimized weights on cascading components (rho={best_opt_score:.4f}):")
for k, w in zip(comp_keys, best_opt_w):
    if w > 0.01:
        print(f"    {k:<20} {w:.3f}")


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

def evaluate(dist_full, expert_dist=expert_dist, cidx=cidx, common=common):
    d = dist_full[np.ix_(cidx, cidx)]
    flat_d = upper_tri(d)
    flat_e = upper_tri(expert_dist)
    r_p, _ = pearsonr(flat_d, flat_e)
    r_s, _ = spearmanr(flat_d, flat_e)
    nn_agree = 0
    nc = len(common)
    for i in range(nc):
        dm = d[i].copy(); dm[i] = np.inf
        da = expert_dist[i].copy(); da[i] = np.inf
        if np.argmin(dm) == np.argmin(da):
            nn_agree += 1
    cond_m = squareform(d, checks=False)
    cond_e = squareform(expert_dist, checks=False)
    Z_m = linkage(cond_m, method='ward')
    Z_e = linkage(cond_e, method='ward')
    cm = cophenet(Z_m)
    ce = cophenet(Z_e)
    r_c, _ = pearsonr(cm, ce)
    return r_p, r_s, nn_agree, r_c

methods_eval = [
    ("Full 4-gram Jaccard", dist_full_4gram),
    ("Full Quadratic Delta", dist_full_stylo),
    ("text-matcher", dist_tm),
    ("ORIGINAL-only Quad. Delta", dist_orig_stylo_full),
    ("ORIGINAL-only 4-gram", dist_orig_4gram),
    ("Structural transition", dist_structural),
]

if HAS_EMB:
    methods_eval.append(("Embedding (full text)", dist_emb_full))
    methods_eval.append(("Embedding (ORIGINAL only)", dist_emb_orig))

methods_eval.append(("CASCADING pipeline", dist_cascade))

# Build optimized cascade distance
opt_cascade_dist = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = 0
        for ki, k in enumerate(comp_keys):
            d += best_opt_w[ki] * dist_components[k][i, j]
        opt_cascade_dist[i, j] = opt_cascade_dist[j, i] = d
methods_eval.append(("OPTIMIZED cascade weights", opt_cascade_dist))

print(f"\n{'Method':<35} {'Pearson r':>10} {'Spearman ρ':>11} {'NN':>7} {'Coph r':>8}")
print("-" * 75)
for label, dist in methods_eval:
    rp, rs, nn, rc = evaluate(dist)
    marker = "  "
    if "CASCADING" in label or "OPTIMIZED" in label:
        marker = "→ "
    print(f"{marker}{label:<33} {rp:>10.3f} {rs:>11.3f} {nn:>3d}/{nc}   {rc:>7.3f}")


# Regime statistics
print("\nPair regime breakdown:")
regime_names = {0: 'no copying', 1: 'moderate copying', 2: 'heavy copying'}
for regime in [0, 1, 2]:
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_regime[i, j] == regime:
                count += 1
    print(f"  Regime {regime} ({regime_names[regime]}): {count} pairs")

# Per-regime accuracy
print("\nPer-regime NN accuracy:")
d_cascade_c = dist_cascade[np.ix_(cidx, cidx)]
for regime in [0, 1, 2]:
    regime_texts = set()
    for i in range(nc):
        for j in range(i+1, nc):
            ii, jj = cidx[i], cidx[j]
            if pair_regime[ii, jj] == regime:
                regime_texts.add(i)
                regime_texts.add(j)
    if regime_texts:
        correct = 0
        total = 0
        for i in regime_texts:
            dm = d_cascade_c[i].copy(); dm[i] = np.inf
            da = expert_dist[i].copy(); da[i] = np.inf
            if np.argmin(dm) == np.argmin(da):
                correct += 1
            total += 1
        print(f"  Regime {regime} ({regime_names[regime]}): {correct}/{total} NN correct")


# ══════════════════════════════════════════════════════════════
# FIGURE GGG: The cascading pipeline explained
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure GGG: Cascading pipeline overview...")

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# Panel 1: Copying map — for each text, show % copied
ax1 = fig.add_subplot(gs[0, 0])
copying_pct = [100 * copied_mask[nm].sum() / len(copied_mask[nm]) for nm in text_names]
colors = [GROUP_COLORS[GROUP_MAP[nm]] for nm in text_names]
bars = ax1.barh(range(n), copying_pct, color=colors, alpha=0.8)
ax1.set_yticks(range(n))
ax1.set_yticklabels(text_names, fontsize=8)
for lbl in ax1.get_yticklabels():
    nm = lbl.get_text()
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax1.set_xlabel('% of tokens appearing in another text', fontsize=9)
ax1.set_title('Step 1: Copying Map\n(text-matcher output)', fontsize=11, fontweight='bold')
ax1.invert_yaxis()

# Panel 2: Original-only vs Full-text stylometry scatter
ax2 = fig.add_subplot(gs[0, 1])
full_flat = upper_tri(dist_full_stylo[np.ix_(cidx, cidx)])
orig_flat = upper_tri(dist_orig_stylo_full[np.ix_(cidx, cidx)])
pair_idx = list(zip(*np.triu_indices(nc, k=1)))
for pi, (i, j) in enumerate(pair_idx):
    gi, gj = GROUP_MAP[common[i]], GROUP_MAP[common[j]]
    same = gi == gj
    ax2.scatter(full_flat[pi], orig_flat[pi],
                c='green' if same else 'red', s=15, alpha=0.5)
ax2.set_xlabel('Full-text Quad. Delta', fontsize=9)
ax2.set_ylabel('Original-only Quad. Delta', fontsize=9)
ax2.plot([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1]], 'k:', alpha=0.3)
ax2.set_title('Step 2: Does removing\ncopied text change distances?', fontsize=11, fontweight='bold')
ax2.legend(handles=[
    Patch(facecolor='green', label='Same Gruppe'),
    Patch(facecolor='red', label='Different Gruppe'),
], fontsize=8, loc='upper left')

# Panel 3: Structural transition points
ax3 = fig.add_subplot(gs[0, 2])
for i, nm in enumerate(text_names):
    tp = transition_points[nm]
    if tp is not None:
        ax3.barh(i, tp, color=GROUP_COLORS[GROUP_MAP[nm]], alpha=0.8)
        ax3.plot([tp, tp], [i-0.4, i+0.4], 'k-', linewidth=2)
    else:
        ax3.barh(i, 0.02, color='gray', alpha=0.3)
ax3.set_yticks(range(n))
ax3.set_yticklabels(text_names, fontsize=8)
for lbl in ax3.get_yticklabels():
    lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])
ax3.set_xlabel('Position where theory > practice', fontsize=9)
ax3.set_title('Step 3: Structural\ntransition points', fontsize=11, fontweight='bold')
ax3.invert_yaxis()
ax3.set_xlim(0, 1)

# Panel 4: Regime assignment
ax4 = fig.add_subplot(gs[0, 3])
regime_matrix = pair_regime[np.ix_(cidx, cidx)].astype(float)
np.fill_diagonal(regime_matrix, np.nan)
from matplotlib.colors import ListedColormap
cmap_regime = ListedColormap(['#e8f4fd', '#ffd699', '#ff6b6b'])
im = ax4.imshow(regime_matrix, cmap=cmap_regime, vmin=0, vmax=2, interpolation='nearest')
ax4.set_xticks(range(nc))
ax4.set_yticks(range(nc))
ax4.set_xticklabels(common, fontsize=7, rotation=90)
ax4.set_yticklabels(common, fontsize=7)
for lbl in ax4.get_xticklabels() + ax4.get_yticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax4.set_title('Pair regime assignment\n(which method mix?)', fontsize=11, fontweight='bold')
# Legend
from matplotlib.patches import Patch as MPatch
legend_elements = [
    MPatch(facecolor='#e8f4fd', label='No copying\n(style+vocab)'),
    MPatch(facecolor='#ffd699', label='Moderate copying\n(mixed)'),
    MPatch(facecolor='#ff6b6b', label='Heavy copying\n(tm dominant)'),
]
ax4.legend(handles=legend_elements, fontsize=7, loc='upper right',
           bbox_to_anchor=(1.0, -0.05))

# Panel 5-8: Dendrograms
for col, (dist_mat, title) in enumerate([
    (dist_full_stylo, "Full Quad. Delta\n(baseline)"),
    (dist_orig_stylo_full, "ORIGINAL-only\nQuad. Delta"),
    (dist_cascade, "CASCADING\npipeline"),
    (expert_dist, "Expert\n(reference)"),
]):
    ax = fig.add_subplot(gs[1, col])
    if title == "Expert\n(reference)":
        d = dist_mat
    else:
        d = dist_mat[np.ix_(cidx, cidx)]
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    cond = squareform(d, checks=False)
    Z = linkage(cond, method='ward')
    dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=8)
    for lbl in ax.get_xticklabels():
        nm = lbl.get_text()
        if nm in GROUP_MAP:
            lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
    rp, rs, nn_count, rc = evaluate(dist_mat) if title != "Expert\n(reference)" else (1, 1, nc, 1)
    if title != "Expert\n(reference)":
        title += f'\nρ={rs:.3f}, NN={nn_count}/{nc}'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=9)

# Panel 9-12 (bottom row): What cascading reveals
# Gain from removing copied text
ax9 = fig.add_subplot(gs[2, 0:2])
rp_full, rs_full, nn_full, _ = evaluate(dist_full_stylo)
rp_orig, rs_orig, nn_orig, _ = evaluate(dist_orig_stylo_full)
rp_f4, rs_f4, nn_f4, _ = evaluate(dist_full_4gram)
rp_o4, rs_o4, nn_o4, _ = evaluate(dist_orig_4gram)
rp_ef, rs_ef, nn_ef, _ = evaluate(dist_emb_full) if HAS_EMB else (0, 0, 0, 0)
rp_eo, rs_eo, nn_eo, _ = evaluate(dist_emb_orig) if HAS_EMB else (0, 0, 0, 0)

methods = ['Quad. Delta', '4-gram Jaccard']
full_rhos = [rs_full, rs_f4]
orig_rhos = [rs_orig, rs_o4]
full_nns = [nn_full / nc, nn_f4 / nc]
orig_nns = [nn_orig / nc, nn_o4 / nc]
if HAS_EMB:
    methods.append('Embedding')
    full_rhos.append(rs_ef)
    orig_rhos.append(rs_eo)
    full_nns.append(nn_ef / nc)
    orig_nns.append(nn_eo / nc)

x = np.arange(len(methods))
w = 0.18
ax9.bar(x - 1.5*w, full_rhos, w, label='Full text ρ', color='#3498db', alpha=0.7)
ax9.bar(x - 0.5*w, orig_rhos, w, label='Original-only ρ', color='#2ecc71', alpha=0.7)
ax9.bar(x + 0.5*w, full_nns, w, label='Full text NN rate', color='#3498db', alpha=0.4, hatch='//')
ax9.bar(x + 1.5*w, orig_nns, w, label='Original-only NN rate', color='#2ecc71', alpha=0.4, hatch='//')
ax9.set_xticks(x)
ax9.set_xticklabels(methods, fontsize=10)
ax9.set_ylabel('Score', fontsize=10)
ax9.legend(fontsize=8)
ax9.set_title('Effect of Removing Copied Regions\n(does the "scribe\'s own voice" help?)',
              fontsize=11, fontweight='bold')
ax9.set_ylim(0, 1)

# Final comparison
ax10 = fig.add_subplot(gs[2, 2:4])
rp_c, rs_c, nn_c, rc_c = evaluate(dist_cascade)
rp_oc, rs_oc, nn_oc, rc_oc = evaluate(opt_cascade_dist)

all_methods = [
    ('Full 4-gram', rs_f4, nn_f4),
    ('Full Quad.Delta', rs_full, nn_full),
    ('text-matcher', evaluate(dist_tm)[1], evaluate(dist_tm)[2]),
    ('Orig-only Delta', rs_orig, nn_orig),
    ('Orig-only 4-gram', rs_o4, nn_o4),
    ('Structural', evaluate(dist_structural)[1], evaluate(dist_structural)[2]),
]
if HAS_EMB:
    all_methods.append(('Emb (orig)', rs_eo, nn_eo))
all_methods.append(('CASCADE', rs_c, nn_c))
all_methods.append(('OPT CASCADE', rs_oc, nn_oc))

labels_m = [m[0] for m in all_methods]
rhos_m = [m[1] for m in all_methods]
nns_m = [m[2] / nc for m in all_methods]

x = np.arange(len(labels_m))
colors_bar = ['#95a5a6'] * (len(labels_m) - 2) + ['#e74c3c', '#e67e22']
ax10.bar(x - 0.15, rhos_m, 0.3, label='Spearman ρ', color=colors_bar, alpha=0.8)
ax10.bar(x + 0.15, nns_m, 0.3, label='NN rate', color=colors_bar, alpha=0.5, hatch='//')
ax10.set_xticks(x)
ax10.set_xticklabels(labels_m, fontsize=8, rotation=45, ha='right')
ax10.set_ylabel('Score', fontsize=10)
ax10.legend(fontsize=8)
ax10.set_title('All Methods: Final Comparison\n(cascading methods in red/orange)',
               fontsize=11, fontweight='bold')
ax10.set_ylim(0, 1)

legend_handles = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=10)
fig.suptitle("Cascading Pipeline: Each Method Informs the Next",
             fontsize=16, fontweight='bold')
plt.savefig(OUT_DIR / 'processus_figGGG_cascading_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig GGG saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE HHH: The key insight — copied vs original tokens
# ══════════════════════════════════════════════════════════════
print("Generating Figure HHH: Copied vs original text maps...")

fig, axes = plt.subplots(n, 1, figsize=(18, n * 0.9 + 2), sharex=True)
for idx, nm in enumerate(text_names):
    ax = axes[idx]
    mask = copied_mask[nm]
    n_tok = len(mask)

    # Plot copied (red) and original (blue) regions
    positions = np.arange(n_tok) / n_tok
    copied_y = np.where(mask, 1, np.nan)
    original_y = np.where(~mask, 1, np.nan)

    ax.fill_between(positions, 0, copied_y, color='#ff6b6b', alpha=0.7, step='mid')
    ax.fill_between(positions, 0, original_y, color='#3498db', alpha=0.4, step='mid')

    # Mark transition point
    tp = transition_points[nm]
    if tp is not None:
        ax.axvline(x=tp, color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks([])
    ax.set_ylabel(nm, fontsize=9, rotation=0, ha='right', va='center',
                  color=GROUP_COLORS[GROUP_MAP[nm]], fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

axes[-1].set_xlabel('Position in text (0% = beginning, 100% = end)', fontsize=11)
axes[0].legend(handles=[
    Patch(facecolor='#ff6b6b', alpha=0.7, label='Copied (shared with other texts)'),
    Patch(facecolor='#3498db', alpha=0.4, label='Original (unique to this text)'),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Theory > Practice transition'),
], fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 1.8))

fig.suptitle("Text-by-Text Copying Map: Which Regions Are Shared vs Original?",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0.06, 0, 1, 0.96])
plt.savefig(OUT_DIR / 'processus_figHHH_copying_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig HHH saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE III: What changes when you remove copied text?
# ══════════════════════════════════════════════════════════════
print("Generating Figure III: Per-pair distance change...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Scatter: full vs original-only distances, colored by regime
d_full_c = dist_full_stylo[np.ix_(cidx, cidx)]
d_orig_c = dist_orig_stylo_full[np.ix_(cidx, cidx)]
d_full_f = upper_tri(d_full_c)
d_orig_f = upper_tri(d_orig_c)
regimes = upper_tri(pair_regime[np.ix_(cidx, cidx)])

regime_colors = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}
for pi in range(len(d_full_f)):
    r = int(regimes[pi])
    ax1.scatter(d_full_f[pi], d_orig_f[pi], c=regime_colors[r], s=20, alpha=0.6)

ax1.plot([d_full_f.min(), d_full_f.max()], [d_full_f.min(), d_full_f.max()],
         'k:', alpha=0.3)
ax1.set_xlabel('Full-text Quad. Delta distance', fontsize=11)
ax1.set_ylabel('Original-only Quad. Delta distance', fontsize=11)
ax1.set_title('How Distances Change When Copied Text Is Removed\n'
              '(above diagonal = more different without copies)',
              fontsize=12, fontweight='bold')
ax1.legend(handles=[
    Patch(facecolor='#3498db', label='No copying'),
    Patch(facecolor='#f39c12', label='Moderate copying'),
    Patch(facecolor='#e74c3c', label='Heavy copying'),
], fontsize=9)

# Which specific pairs change most?
delta = d_orig_f - d_full_f
top_changes = np.argsort(np.abs(delta))[::-1][:15]
pair_list = list(zip(*np.triu_indices(nc, k=1)))

change_data = []
for rank, pi in enumerate(top_changes):
    i, j = pair_list[pi]
    change_data.append({
        'pair': f'{common[i]}↔{common[j]}',
        'delta': delta[pi],
        'full': d_full_f[pi],
        'orig': d_orig_f[pi],
        'regime': int(regimes[pi]),
    })

y_pos = range(len(change_data))
bars = ax2.barh(y_pos, [d['delta'] for d in change_data],
                color=[regime_colors[d['regime']] for d in change_data], alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([d['pair'] for d in change_data], fontsize=8)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_xlabel('Distance change (original − full)\n'
               '(positive = more different without copies)', fontsize=10)
ax2.set_title('Top 15 Most-Affected Pairs\n'
              '(which relationships change most?)',
              fontsize=12, fontweight='bold')
ax2.invert_yaxis()

fig.suptitle("The Cascading Insight: Removing Copied Text Reveals Hidden Relationships",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_DIR / 'processus_figIII_distance_changes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig III saved.")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASCADING PIPELINE SUMMARY")
print("=" * 70)

rp_c, rs_c, nn_c, rc_c = evaluate(dist_cascade)
rp_oc, rs_oc, nn_oc, rc_oc = evaluate(opt_cascade_dist)

print(f"\nFinal comparison:")
print(f"  Full 4-gram (baseline):      ρ={rs_f4:.3f}, NN={nn_f4}/{nc}")
print(f"  Full Quad. Delta (baseline):  ρ={rs_full:.3f}, NN={nn_full}/{nc}")
print(f"  CASCADING pipeline:           ρ={rs_c:.3f}, NN={nn_c}/{nc}")
print(f"  OPTIMIZED cascade weights:    ρ={rs_oc:.3f}, NN={nn_oc}/{nc}")

print(f"\nKey finding: Original-only stylometry vs full-text:")
print(f"  Full Quad. Delta:     ρ={rs_full:.3f}, NN={nn_full}/{nc}")
print(f"  Original-only Delta:  ρ={rs_orig:.3f}, NN={nn_orig}/{nc}")
if HAS_EMB:
    print(f"\nKey finding: Original-only embeddings vs full-text:")
    print(f"  Full embedding:     ρ={rs_ef:.3f}, NN={nn_ef}/{nc}")
    print(f"  Original-only emb:  ρ={rs_eo:.3f}, NN={nn_eo}/{nc}")

print(f"\nFigures saved:")
print(f"  Fig GGG: processus_figGGG_cascading_pipeline.png")
print(f"  Fig HHH: processus_figHHH_copying_maps.png")
print(f"  Fig III: processus_figIII_distance_changes.png")
