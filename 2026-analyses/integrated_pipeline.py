#!/usr/bin/env python3
"""
Integrated Multi-Method Pipeline
=================================
Systematically combines all available text comparison methods to find
the optimal blend for approximating expert annotations.

Methods available:
  1. Proxy character matrix (Jaccard on discovered binary characters)
  2. Quadratic Delta (300 MFW stylometry)
  3. 4-gram Jaccard overlap
  4. text-matcher (longest common substring matching)
  5. Sentence embeddings (full text, early half, late half)

Produces Figures CCC through FFF and documentation.
"""

import re
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('processus-universalis-graphics')
OUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════

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

# ── Load texts ──
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
print(f"  {n} texts loaded")

# ── Load expert annotations ──
print("Loading expert annotations...")
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
print(f"  {nc} texts have expert annotations")

# Expert distance matrix
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
# COMPUTE ALL DISTANCE MATRICES
# ══════════════════════════════════════════════════════════════

def tokenize(text):
    return re.findall(r'[a-zäöüß]+', text.lower())

text_tokens = {name: tokenize(plain_texts[name]) for name in text_names}
name_idx = {name: i for i, name in enumerate(text_names)}

# ── Method 1: Proxy character matrix ──
print("\nComputing Method 1: Proxy character matrix...")

# Cologne phonetic
def cologne_phonetic(word):
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

# Phonetic normalization
corpus_vocab = set()
for name in text_names:
    corpus_vocab.update(text_tokens[name])
phon_map = {}
phon_groups = defaultdict(set)
for w in corpus_vocab:
    code = cologne_phonetic(w)
    phon_map[w] = code
    phon_groups[code].add(w)

def normalize_tokens(tokens):
    return [phon_map.get(t, t) for t in tokens]

norm_tokens = {name: normalize_tokens(text_tokens[name]) for name in text_names}

# Discover recurring content elements (shared 4-grams on phonetic tokens)
phon_ngrams = {}
for name in text_names:
    toks = norm_tokens[name]
    grams = set(tuple(toks[i:i+4]) for i in range(len(toks)-3))
    phon_ngrams[name] = grams

# Find 4-grams shared by at least 2 texts
gram_to_texts = defaultdict(set)
for name in text_names:
    for gram in phon_ngrams[name]:
        gram_to_texts[gram].add(name)

shared_grams = {gram: texts for gram, texts in gram_to_texts.items()
                if len(texts) >= 2}

# Build binary character matrix
char_list = sorted(shared_grams.keys(), key=lambda g: (-len(shared_grams[g]), g))
# Limit to top characters for tractability
MAX_CHARS = 500
char_list = char_list[:MAX_CHARS]
matrix = np.zeros((n, len(char_list)), dtype=int)
for ci, gram in enumerate(char_list):
    for name in shared_grams[gram]:
        matrix[name_idx[name], ci] = 1

# Proxy distances (Jaccard on binary matrix)
common_matrix = matrix[[name_idx[nm] for nm in common]]
dist_proxy = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        si = set(np.where(common_matrix[i] == 1)[0])
        sj = set(np.where(common_matrix[j] == 1)[0])
        u = len(si | sj)
        jac = len(si & sj) / u if u > 0 else 0
        dist_proxy[i, j] = dist_proxy[j, i] = 1 - jac

print(f"  {len(char_list)} proxy characters, matrix shape: {matrix.shape}")


# ── Method 2: Quadratic Delta (300 MFW) ──
print("Computing Method 2: Quadratic Delta (300 MFW)...")

all_tokens_flat = []
for name in text_names:
    all_tokens_flat.extend(text_tokens[name])
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

def delta_quadratic(fm):
    z_means = fm.mean(axis=0)
    z_stds = fm.std(axis=0, ddof=0)
    z_stds[z_stds == 0] = 1
    z = (fm - z_means) / z_stds
    nt = z.shape[0]
    dist = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(i+1, nt):
            d = np.sqrt(np.mean((z[i] - z[j])**2))
            dist[i, j] = dist[j, i] = d
    return dist

dist_stylo_full = delta_quadratic(features_matrix)
cidx = [name_idx[nm] for nm in common]
dist_stylo = dist_stylo_full[np.ix_(cidx, cidx)]
print("  Done.")


# ── Method 3: 4-gram Jaccard ──
print("Computing Method 3: 4-gram Jaccard overlap...")

raw_ngrams = {}
for name in text_names:
    words = plain_texts[name].lower().split()
    raw_ngrams[name] = set(tuple(words[i:i+4]) for i in range(len(words)-3))

dist_4gram = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        si = raw_ngrams[common[i]]
        sj = raw_ngrams[common[j]]
        u = len(si | sj)
        jac = len(si & sj) / u if u > 0 else 0
        dist_4gram[i, j] = dist_4gram[j, i] = 1 - jac
print("  Done.")


# ── Method 4: text-matcher (longest common substring) ──
print("Computing Method 4: text-matcher (longest common substring)...")

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
        self.trigrams = list(self._ngrams(3))

    def _ngrams(self, n):
        return [tuple(self.tokens[i:i+n]) for i in range(len(self.tokens) - n + 1)]

    def ngrams(self, n):
        return list(self._ngrams(n))


class GermanMatcher:
    def __init__(self, textA, textB, threshold=3, cutoff=5, ngram_size=3):
        self.textA = textA
        self.textB = textB
        self.threshold = threshold
        self.cutoff = cutoff
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
                next_match = matches[i + 1]
                if (next_match.a - (match.a + match.size)) < min_distance:
                    sizeA = (next_match.a + next_match.size) - match.a
                    sizeB = (next_match.b + next_match.size) - match.b
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

    def get_passage(self, text, start, length):
        ngram_size = self.ngram_size
        actual_len = length + ngram_size - 1
        spans = text.spans[start:start + actual_len]
        if len(spans) == 0:
            return ""
        return text.text[spans[0][0]:spans[-1][-1]]


# Build text objects
gt_objects = {name: GermanText(plain_texts[name], name) for name in text_names}

# Run all pairs
tm_score = np.zeros((n, n))
tm_match_count = np.zeros((n, n))
tm_total_words = np.zeros((n, n))
tm_max_len = np.zeros((n, n))  # longest match per pair

pair_count = 0
total_pairs = n * (n - 1) // 2
for i in range(n):
    for j in range(i+1, n):
        pair_count += 1
        na, nb = text_names[i], text_names[j]
        matcher = GermanMatcher(gt_objects[na], gt_objects[nb])
        total_matched = 0
        max_match = 0
        for match in matcher.extended_matches:
            lenA = match.sizeA + matcher.ngram_size - 1
            total_matched += lenA
            if lenA > max_match:
                max_match = lenA
        norm = min(len(gt_objects[na].tokens), len(gt_objects[nb].tokens))
        score = total_matched / norm if norm > 0 else 0
        tm_score[i, j] = tm_score[j, i] = score
        tm_match_count[i, j] = tm_match_count[j, i] = matcher.numMatches
        tm_total_words[i, j] = tm_total_words[j, i] = total_matched
        tm_max_len[i, j] = tm_max_len[j, i] = max_match
        if pair_count % 20 == 0:
            print(f"  {pair_count}/{total_pairs} pairs done...")

print(f"  {total_pairs}/{total_pairs} pairs done.")

# Distance from text-matcher
tm_dist_full = 1 - tm_score / (tm_score.max() + 1e-10)
dist_tm = tm_dist_full[np.ix_(cidx, cidx)]

# Also: longest-match-only distance
tm_maxlen_norm = tm_max_len / (tm_max_len.max() + 1e-10)
dist_tm_maxlen = 1 - tm_maxlen_norm[np.ix_(cidx, cidx)]

print("  Done.")


# ── Method 5: Sentence embeddings ──
print("Computing Method 5: Sentence embeddings...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def chunk_text(text, chunk_size=80, overlap=40):
        words = text.split()
        chunks = []
        for i in range(0, len(words) - chunk_size + 1, chunk_size - overlap):
            chunks.append(' '.join(words[i:i+chunk_size]))
        if not chunks:
            chunks = [text]
        return chunks

    text_embeddings = {}
    early_embeddings = {}
    late_embeddings = {}
    for name in text_names:
        chunks = chunk_text(plain_texts[name])
        embs = model.encode(chunks)
        text_embeddings[name] = np.mean(embs, axis=0)
        mid = len(chunks) // 2
        if mid > 0:
            early_embeddings[name] = np.mean(embs[:mid], axis=0)
            late_embeddings[name] = np.mean(embs[mid:], axis=0)
        else:
            early_embeddings[name] = np.mean(embs, axis=0)
            late_embeddings[name] = np.mean(embs, axis=0)

    # Cosine distance
    from numpy.linalg import norm as np_norm
    def cosine_dist_matrix(emb_dict, names):
        n = len(names)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                a, b = emb_dict[names[i]], emb_dict[names[j]]
                sim = np.dot(a, b) / (np_norm(a) * np_norm(b) + 1e-10)
                dist[i, j] = dist[j, i] = 1 - sim
        return dist

    dist_emb_full = cosine_dist_matrix(text_embeddings, common)
    dist_emb_early = cosine_dist_matrix(early_embeddings, common)
    dist_emb_late = cosine_dist_matrix(late_embeddings, common)
    HAS_EMBEDDINGS = True
    print("  Done.")
except ImportError:
    print("  sentence-transformers not available, skipping embeddings.")
    dist_emb_full = np.zeros((nc, nc))
    dist_emb_early = np.zeros((nc, nc))
    dist_emb_late = np.zeros((nc, nc))
    HAS_EMBEDDINGS = False


# ══════════════════════════════════════════════════════════════
# NORMALIZE ALL DISTANCE MATRICES TO [0,1]
# ══════════════════════════════════════════════════════════════

def normalize_dist(d):
    """Normalize distance matrix to [0,1] range."""
    flat = upper_tri(d)
    mn, mx = flat.min(), flat.max()
    if mx - mn < 1e-10:
        return d.copy()
    return (d - mn) / (mx - mn)

dist_matrices = {
    'proxy':     normalize_dist(dist_proxy),
    'stylo':     normalize_dist(dist_stylo),
    '4gram':     normalize_dist(dist_4gram),
    'tm':        normalize_dist(dist_tm),
    'tm_maxlen': normalize_dist(dist_tm_maxlen),
}

if HAS_EMBEDDINGS:
    dist_matrices['emb_full'] = normalize_dist(dist_emb_full)
    dist_matrices['emb_early'] = normalize_dist(dist_emb_early)
    dist_matrices['emb_late'] = normalize_dist(dist_emb_late)

method_labels = {
    'proxy': 'Proxy characters\n(Cologne phonetic 4-grams)',
    'stylo': 'Quadratic Delta\n(300 MFW)',
    '4gram': '4-gram Jaccard\n(raw overlap)',
    'tm':    'text-matcher\n(long common substr)',
    'tm_maxlen': 'text-matcher\n(longest match only)',
    'emb_full': 'Embedding\n(full text)',
    'emb_early': 'Embedding\n(early half)',
    'emb_late': 'Embedding\n(late half)',
}


# ══════════════════════════════════════════════════════════════
# EVALUATE ALL INDIVIDUAL METHODS
# ══════════════════════════════════════════════════════════════

def evaluate(dist, expert_dist=expert_dist, common=common):
    """Full evaluation: Pearson r, Spearman rho, NN agreement, cophenetic r."""
    flat_d = upper_tri(dist)
    flat_e = upper_tri(expert_dist)
    r_p, _ = pearsonr(flat_d, flat_e)
    r_s, _ = spearmanr(flat_d, flat_e)
    nn_agree = 0
    nc = len(common)
    for i in range(nc):
        dm = dist[i].copy(); dm[i] = np.inf
        da = expert_dist[i].copy(); da[i] = np.inf
        if np.argmin(dm) == np.argmin(da):
            nn_agree += 1
    cond_m = squareform(dist, checks=False)
    cond_e = squareform(expert_dist, checks=False)
    Z_m = linkage(cond_m, method='ward')
    Z_e = linkage(cond_e, method='ward')
    cm = cophenet(Z_m)
    ce = cophenet(Z_e)
    r_c, _ = pearsonr(cm, ce)
    return r_p, r_s, nn_agree, r_c

print("\n" + "=" * 80)
print("INDIVIDUAL METHOD EVALUATION")
print("=" * 80)
print(f"{'Method':<35} {'Pearson r':>10} {'Spearman ρ':>11} {'NN':>7} {'Coph r':>8}")
print("-" * 75)

individual_results = {}
for key in dist_matrices:
    rp, rs, nn, rc = evaluate(dist_matrices[key])
    individual_results[key] = {'r': rp, 'rho': rs, 'nn': nn, 'coph': rc}
    print(f"  {key:<33} {rp:>10.3f} {rs:>11.3f} {nn:>3d}/{nc}   {rc:>7.3f}")


# ══════════════════════════════════════════════════════════════
# SYSTEMATIC COMBINATION SEARCH
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SEARCHING FOR OPTIMAL COMBINATIONS")
print("=" * 80)

method_keys = list(dist_matrices.keys())
n_methods = len(method_keys)

# Stack upper triangles for fast linear combination
flat_stack = np.array([upper_tri(dist_matrices[k]) for k in method_keys])
n_pairs = len(expert_flat)


def evaluate_weights(weights, metric='rho'):
    """Evaluate a weighted combination of methods."""
    weights = np.array(weights)
    if weights.sum() < 1e-10:
        return 0.0
    weights = weights / weights.sum()
    combined_flat = weights @ flat_stack
    if metric == 'rho':
        return spearmanr(combined_flat, expert_flat)[0]
    elif metric == 'r':
        return pearsonr(combined_flat, expert_flat)[0]
    elif metric == 'nn':
        # Build matrix from flat
        combined_dist = np.zeros((nc, nc))
        idx = np.triu_indices(nc, k=1)
        combined_dist[idx] = combined_flat
        combined_dist = combined_dist + combined_dist.T
        nn_agree = 0
        for i in range(nc):
            dm = combined_dist[i].copy(); dm[i] = np.inf
            da = expert_dist[i].copy(); da[i] = np.inf
            if np.argmin(dm) == np.argmin(da):
                nn_agree += 1
        return nn_agree


def build_combined_dist(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    combined_flat = weights @ flat_stack
    combined_dist = np.zeros((nc, nc))
    idx = np.triu_indices(nc, k=1)
    combined_dist[idx] = combined_flat
    combined_dist = combined_dist + combined_dist.T
    return combined_dist


# Strategy 1: Exhaustive search over all subsets of 2-4 methods, grid of weights
print("\nStrategy 1: Grid search over method subsets...")

best_results = []

# All combinations of 2, 3, 4, and 5 methods
for size in range(2, min(n_methods + 1, 6)):
    for combo in combinations(range(n_methods), size):
        # Grid search over weights
        if size == 2:
            grid = np.linspace(0, 1, 21)
            best_score = -1
            best_w = None
            for w0 in grid:
                w1 = 1 - w0
                weights = np.zeros(n_methods)
                weights[combo[0]] = w0
                weights[combo[1]] = w1
                score = evaluate_weights(weights, metric='rho')
                if score > best_score:
                    best_score = score
                    best_w = weights.copy()
        elif size == 3:
            grid = np.linspace(0, 1, 11)
            best_score = -1
            best_w = None
            for w0 in grid:
                for w1 in grid:
                    w2 = 1 - w0 - w1
                    if w2 < -0.01:
                        continue
                    w2 = max(0, w2)
                    weights = np.zeros(n_methods)
                    weights[combo[0]] = w0
                    weights[combo[1]] = w1
                    weights[combo[2]] = w2
                    score = evaluate_weights(weights, metric='rho')
                    if score > best_score:
                        best_score = score
                        best_w = weights.copy()
        else:
            # For 4+ methods, use equal weights then refine
            weights = np.zeros(n_methods)
            for idx in combo:
                weights[idx] = 1.0 / size
            best_score = evaluate_weights(weights, metric='rho')
            best_w = weights.copy()
            # Try emphasizing each method
            for emphasis_idx in combo:
                for emphasis_weight in [0.4, 0.5, 0.6]:
                    w = np.zeros(n_methods)
                    remaining = 1 - emphasis_weight
                    for idx in combo:
                        if idx == emphasis_idx:
                            w[idx] = emphasis_weight
                        else:
                            w[idx] = remaining / (size - 1)
                    score = evaluate_weights(w, metric='rho')
                    if score > best_score:
                        best_score = score
                        best_w = w.copy()

        combo_names = '+'.join(method_keys[i] for i in combo)
        nn = evaluate_weights(best_w, metric='nn')
        r = evaluate_weights(best_w, metric='r')
        best_results.append({
            'combo': combo_names,
            'rho': best_score,
            'r': r,
            'nn': nn,
            'weights': best_w.copy(),
            'size': size,
        })

# Sort by Spearman rho
best_results.sort(key=lambda x: -x['rho'])

print(f"\nTop 20 combinations (by Spearman ρ):")
print(f"{'Rank':<5} {'Combination':<45} {'ρ':>7} {'r':>7} {'NN':>5} {'Weights'}")
print("-" * 100)
for i, res in enumerate(best_results[:20]):
    w_str = ', '.join(f'{method_keys[j]}={res["weights"][j]:.2f}'
                      for j in range(n_methods) if res['weights'][j] > 0.01)
    print(f"  {i+1:<3} {res['combo']:<45} {res['rho']:>7.3f} {res['r']:>7.3f} "
          f"{res['nn']:>3d}/{nc} {w_str}")


# Strategy 2: scipy.optimize for continuous optimization
print("\nStrategy 2: Continuous optimization (Nelder-Mead)...")

def neg_rho(w):
    return -evaluate_weights(w, metric='rho')

# Try multiple starting points
best_opt_score = -1
best_opt_weights = None

# Start from equal weights
starts = [np.ones(n_methods) / n_methods]
# Start from best individual method
best_ind = max(individual_results, key=lambda k: individual_results[k]['rho'])
best_ind_idx = method_keys.index(best_ind)
w_start = np.zeros(n_methods)
w_start[best_ind_idx] = 1.0
starts.append(w_start)
# Start from top grid result
starts.append(best_results[0]['weights'])

for start in starts:
    result = minimize(neg_rho, start, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 0.001})
    # Clamp negatives to zero
    w = np.maximum(result.x, 0)
    if w.sum() > 0:
        w = w / w.sum()
    score = evaluate_weights(w, metric='rho')
    if score > best_opt_score:
        best_opt_score = score
        best_opt_weights = w.copy()

print(f"  Best optimized ρ = {best_opt_score:.4f}")
w_str = ', '.join(f'{method_keys[j]}={best_opt_weights[j]:.3f}'
                  for j in range(n_methods) if best_opt_weights[j] > 0.01)
print(f"  Weights: {w_str}")

# Full evaluation of the optimized combination
opt_dist = build_combined_dist(best_opt_weights)
opt_rp, opt_rs, opt_nn, opt_rc = evaluate(opt_dist)
print(f"  Full eval: r={opt_rp:.3f}, ρ={opt_rs:.3f}, NN={opt_nn}/{nc}, coph={opt_rc:.3f}")


# Strategy 3: NN-maximizing search (brute force over grid for NN agreement)
print("\nStrategy 3: Maximizing nearest-neighbor agreement...")

best_nn_score = 0
best_nn_weights = None
best_nn_rho = 0

# Dense grid for top 3 methods by individual NN
sorted_by_nn = sorted(individual_results.items(), key=lambda x: -x[1]['nn'])
top_nn_keys = [k for k, v in sorted_by_nn[:4]]
top_nn_idx = [method_keys.index(k) for k in top_nn_keys]

grid = np.linspace(0, 1, 21)
for w0 in grid:
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w0 - w1 - w2
            if w3 < -0.01:
                continue
            w3 = max(0, w3)
            weights = np.zeros(n_methods)
            weights[top_nn_idx[0]] = w0
            weights[top_nn_idx[1]] = w1
            weights[top_nn_idx[2]] = w2
            if len(top_nn_idx) > 3:
                weights[top_nn_idx[3]] = w3
            nn = evaluate_weights(weights, metric='nn')
            rho = evaluate_weights(weights, metric='rho')
            if nn > best_nn_score or (nn == best_nn_score and rho > best_nn_rho):
                best_nn_score = nn
                best_nn_weights = weights.copy()
                best_nn_rho = rho

nn_dist = build_combined_dist(best_nn_weights)
nn_rp, nn_rs, nn_nn, nn_rc = evaluate(nn_dist)
w_str = ', '.join(f'{method_keys[j]}={best_nn_weights[j]:.2f}'
                  for j in range(n_methods) if best_nn_weights[j] > 0.01)
print(f"  Best NN = {nn_nn}/{nc}, ρ={nn_rs:.3f}, r={nn_rp:.3f}")
print(f"  Weights: {w_str}")


# ══════════════════════════════════════════════════════════════
# CROSS-VALIDATION: Leave-one-pair-out
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("CROSS-VALIDATION: Leave-one-pair-out")
print("=" * 80)

n_pair_total = len(expert_flat)
# For each held-out pair, optimize weights on remaining pairs, evaluate on held-out
cv_errors_individual = {k: [] for k in method_keys}
cv_errors_optimal = []

pair_indices = list(zip(*np.triu_indices(nc, k=1)))

for hold_idx in range(n_pair_total):
    train_mask = np.ones(n_pair_total, dtype=bool)
    train_mask[hold_idx] = False

    train_expert = expert_flat[train_mask]
    train_stack = flat_stack[:, train_mask]

    # Individual methods
    for ki, k in enumerate(method_keys):
        pred = flat_stack[ki, hold_idx]
        cv_errors_individual[k].append((pred - expert_flat[hold_idx])**2)

    # Optimized combination (fit on train, predict on held-out)
    def neg_rho_train(w):
        w = np.maximum(w, 0)
        s = w.sum()
        if s < 1e-10:
            return 0.0
        w = w / s
        pred = w @ train_stack
        return -spearmanr(pred, train_expert)[0]

    res = minimize(neg_rho_train, best_opt_weights, method='Nelder-Mead',
                   options={'maxiter': 2000})
    w = np.maximum(res.x, 0)
    if w.sum() > 0:
        w = w / w.sum()
    pred = w @ flat_stack[:, hold_idx:hold_idx+1]
    cv_errors_optimal.append((pred[0] - expert_flat[hold_idx])**2)

cv_rmse_individual = {k: np.sqrt(np.mean(v)) for k, v in cv_errors_individual.items()}
cv_rmse_optimal = np.sqrt(np.mean(cv_errors_optimal))

print(f"\n{'Method':<35} {'CV RMSE':>10}")
print("-" * 47)
for k in method_keys:
    print(f"  {k:<33} {cv_rmse_individual[k]:>10.4f}")
print(f"  {'Optimized combination':<33} {cv_rmse_optimal:>10.4f}")


# ══════════════════════════════════════════════════════════════
# FIGURE CCC: Method Comparison Overview
# ══════════════════════════════════════════════════════════════

print("\nGenerating Figure CCC: Method comparison overview...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# Panel 1: Bar chart of all metrics for individual methods
ax = axes[0]
methods_sorted = sorted(individual_results.keys(), key=lambda k: -individual_results[k]['rho'])
x = np.arange(len(methods_sorted))
w_bar = 0.2
rhos = [individual_results[k]['rho'] for k in methods_sorted]
rs = [individual_results[k]['r'] for k in methods_sorted]
nns = [individual_results[k]['nn'] / nc for k in methods_sorted]
cophs = [individual_results[k]['coph'] for k in methods_sorted]

ax.bar(x - 1.5*w_bar, rs, w_bar, label='Pearson r', color='#e74c3c', alpha=0.8)
ax.bar(x - 0.5*w_bar, rhos, w_bar, label='Spearman ρ', color='#3498db', alpha=0.8)
ax.bar(x + 0.5*w_bar, nns, w_bar, label='NN rate', color='#2ecc71', alpha=0.8)
ax.bar(x + 1.5*w_bar, cophs, w_bar, label='Cophenetic r', color='#9b59b6', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([k.replace('_', '\n') for k in methods_sorted], fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Individual Method Performance\n(higher = better)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

# Panel 2: Correlation matrix between methods
ax = axes[1]
method_corr = np.zeros((n_methods, n_methods))
for i in range(n_methods):
    for j in range(n_methods):
        method_corr[i, j], _ = pearsonr(flat_stack[i], flat_stack[j])

im = ax.imshow(method_corr, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
ax.set_xticks(range(n_methods))
ax.set_yticks(range(n_methods))
ax.set_xticklabels([k.replace('_', '\n') for k in method_keys], fontsize=8, rotation=45, ha='right')
ax.set_yticklabels(method_keys, fontsize=8)
for i in range(n_methods):
    for j in range(n_methods):
        ax.text(j, i, f'{method_corr[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if method_corr[i,j] > 0.7 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Inter-Method Correlation\n(how similar are the methods?)', fontsize=12, fontweight='bold')

# Panel 3: What each method captures best (scatter of r vs rho)
ax = axes[2]
for k in method_keys:
    r = individual_results[k]['r']
    rho = individual_results[k]['rho']
    nn = individual_results[k]['nn']
    ax.scatter(r, rho, s=nn * 30 + 50, alpha=0.8, zorder=5)
    ax.annotate(k.replace('_', '\n'), (r, rho), fontsize=8,
                textcoords='offset points', xytext=(8, 5))

# Add optimized point
ax.scatter(opt_rp, opt_rs, s=200, marker='*', c='gold',
           edgecolors='black', linewidths=1.5, zorder=10)
ax.annotate('OPTIMIZED\nCOMBINATION', (opt_rp, opt_rs), fontsize=9, fontweight='bold',
            textcoords='offset points', xytext=(10, -15), color='darkgoldenrod')

ax.set_xlabel('Pearson r (linear correlation)', fontsize=11)
ax.set_ylabel('Spearman ρ (rank correlation)', fontsize=11)
ax.set_title('Method Landscape\n(size = NN agreement)', fontsize=12, fontweight='bold')
ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
ax.set_xlim(0.2, 1.0)
ax.set_ylim(0.6, 1.0)

fig.suptitle("Multi-Method Overview: What Each Method Captures",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figCCC_method_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig CCC saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE DDD: Optimal combination dendrogram
# ══════════════════════════════════════════════════════════════

print("Generating Figure DDD: Optimal combination dendrogram...")

fig, axes = plt.subplots(1, 4, figsize=(28, 8))

best_individual_key = max(individual_results, key=lambda k: individual_results[k]['rho'])

for ax, (dist, title) in zip(axes, [
    (dist_matrices[best_individual_key],
     f"Best Individual: {best_individual_key}\nr={individual_results[best_individual_key]['r']:.3f}, "
     f"ρ={individual_results[best_individual_key]['rho']:.3f}, "
     f"NN={individual_results[best_individual_key]['nn']}/{nc}"),
    (opt_dist,
     f"Optimized ρ Combination\nr={opt_rp:.3f}, ρ={opt_rs:.3f}, NN={opt_nn}/{nc}"),
    (nn_dist,
     f"Optimized NN Combination\nr={nn_rp:.3f}, ρ={nn_rs:.3f}, NN={nn_nn}/{nc}"),
    (expert_dist,
     f"Expert Annotations\n(reference)"),
]):
    np.fill_diagonal(dist, 0)
    dist_sym = (dist + dist.T) / 2
    cond = squareform(dist_sym, checks=False)
    Z = linkage(cond, method='ward')
    dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=9)
    for lbl in ax.get_xticklabels():
        nm = lbl.get_text()
        if nm in GROUP_MAP:
            lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=10)

legend_handles = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=10)
fig.suptitle("Dendrogram Comparison: Individual vs Optimized Combinations vs Expert",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.92])
plt.savefig(OUT_DIR / 'processus_figDDD_optimal_dendrograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig DDD saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE EEE: Weight sensitivity analysis
# ══════════════════════════════════════════════════════════════

print("Generating Figure EEE: Weight sensitivity...")

# For the top 2-method combination, sweep weights
top2_combo = best_results[0]['combo'].split('+')
top2_combo = [k for k in top2_combo if k in method_keys][:2]

if len(top2_combo) == 2:
    idx0 = method_keys.index(top2_combo[0])
    idx1 = method_keys.index(top2_combo[1])

    sweep = np.linspace(0, 1, 101)
    sweep_r = []
    sweep_rho = []
    sweep_nn = []

    for w0 in sweep:
        weights = np.zeros(n_methods)
        weights[idx0] = w0
        weights[idx1] = 1 - w0
        sweep_r.append(evaluate_weights(weights, metric='r'))
        sweep_rho.append(evaluate_weights(weights, metric='rho'))
        sweep_nn.append(evaluate_weights(weights, metric='nn'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(sweep, sweep_r, 'r-', linewidth=2, label='Pearson r')
    ax1.plot(sweep, sweep_rho, 'b-', linewidth=2, label='Spearman ρ')
    ax1b = ax1.twinx()
    ax1b.plot(sweep, sweep_nn, 'g--', linewidth=2, label=f'NN (of {nc})')
    ax1b.set_ylabel(f'NN agreement (of {nc})', color='green', fontsize=11)

    # Mark optima
    best_rho_idx = np.argmax(sweep_rho)
    best_nn_idx = np.argmax(sweep_nn)
    ax1.axvline(x=sweep[best_rho_idx], color='blue', linestyle=':', alpha=0.5)
    ax1.axvline(x=sweep[best_nn_idx], color='green', linestyle=':', alpha=0.5)

    ax1.set_xlabel(f'Weight on {top2_combo[0]} (remainder on {top2_combo[1]})', fontsize=11)
    ax1.set_ylabel('Correlation with expert', fontsize=11)
    ax1.set_title(f'Weight Sensitivity: {top2_combo[0]} + {top2_combo[1]}',
                  fontsize=12, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    # Panel 2: Heatmap for top 3-method combination, fixing one weight and sweeping other two
    top3_results = [r for r in best_results if r['size'] == 3]
    if top3_results:
        top3_combo = top3_results[0]['combo'].split('+')
        top3_combo = [k for k in top3_combo if k in method_keys][:3]

        if len(top3_combo) == 3:
            idx_a = method_keys.index(top3_combo[0])
            idx_b = method_keys.index(top3_combo[1])
            idx_c = method_keys.index(top3_combo[2])

            grid_res = 51
            rho_grid = np.zeros((grid_res, grid_res))
            ws = np.linspace(0, 1, grid_res)

            for i, wa in enumerate(ws):
                for j, wb in enumerate(ws):
                    wc = 1 - wa - wb
                    if wc < -0.01:
                        rho_grid[i, j] = np.nan
                        continue
                    wc = max(0, wc)
                    weights = np.zeros(n_methods)
                    weights[idx_a] = wa
                    weights[idx_b] = wb
                    weights[idx_c] = wc
                    rho_grid[i, j] = evaluate_weights(weights, metric='rho')

            im = ax2.imshow(rho_grid, extent=[0, 1, 1, 0], cmap='viridis',
                           interpolation='bilinear', aspect='auto')
            plt.colorbar(im, ax=ax2, label='Spearman ρ')

            # Mark optimum
            valid = ~np.isnan(rho_grid)
            if valid.any():
                opt_ij = np.unravel_index(np.nanargmax(rho_grid), rho_grid.shape)
                ax2.scatter(ws[opt_ij[1]], ws[opt_ij[0]], c='red', s=100, marker='*',
                           edgecolors='white', linewidths=2, zorder=10)

            ax2.set_xlabel(f'Weight on {top3_combo[1]}', fontsize=11)
            ax2.set_ylabel(f'Weight on {top3_combo[0]}', fontsize=11)
            ax2.set_title(f'3-Method Landscape\n({top3_combo[2]} = remainder)\nρ heatmap',
                          fontsize=12, fontweight='bold')

    fig.suptitle("Weight Sensitivity Analysis: How Robust Are the Optimal Weights?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_DIR / 'processus_figEEE_weight_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig EEE saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE FFF: Per-text diagnostic — who benefits from combination?
# ══════════════════════════════════════════════════════════════

print("Generating Figure FFF: Per-text NN diagnostic...")

fig, ax = plt.subplots(figsize=(14, 8))

# For each text: show NN correctness across methods
methods_to_show = [best_individual_key, 'optimized_rho', 'optimized_nn']
dist_to_show = [dist_matrices[best_individual_key], opt_dist, nn_dist]
labels_show = [f'Best individual\n({best_individual_key})', 'Optimized ρ', 'Optimized NN']

# Build a grid: texts x methods, cell = 1 if NN correct, 0 if not
nn_grid = np.zeros((nc, len(dist_to_show) + len(method_keys)))
all_labels = [k for k in method_keys] + labels_show
all_dists = [dist_matrices[k] for k in method_keys] + dist_to_show

for mi, (label, dist) in enumerate(zip(all_labels, all_dists)):
    for i in range(nc):
        dm = dist[i].copy(); dm[i] = np.inf
        da = expert_dist[i].copy(); da[i] = np.inf
        nn_grid[i, mi] = 1 if np.argmin(dm) == np.argmin(da) else 0

im = ax.imshow(nn_grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels([l.replace('\n', ' ') for l in all_labels], fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(nc))
ax.set_yticklabels(common, fontsize=9)

# Color text labels by group
for lbl in ax.get_yticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])

# Add column totals
for mi in range(len(all_labels)):
    total = int(nn_grid[:, mi].sum())
    ax.text(mi, nc + 0.3, f'{total}', ha='center', va='top', fontsize=9, fontweight='bold')

# Separator line between individual and combined
ax.axvline(x=len(method_keys) - 0.5, color='black', linewidth=2, linestyle='--')
ax.text(len(method_keys) - 0.3, -1, 'Combined →', fontsize=9, fontweight='bold', ha='left')

ax.set_title("Per-Text Nearest-Neighbor Accuracy Across All Methods\n"
             "(Green = correct NN, Red = wrong NN)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figFFF_per_text_nn.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig FFF saved.")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("INTEGRATED PIPELINE SUMMARY")
print("=" * 80)

print(f"\n{'Method':<35} {'r':>7} {'ρ':>7} {'NN':>7} {'Coph':>7}")
print("-" * 67)
for k in method_keys:
    res = individual_results[k]
    print(f"  {k:<33} {res['r']:>7.3f} {res['rho']:>7.3f} {res['nn']:>3d}/{nc}   {res['coph']:>6.3f}")
print("-" * 67)
print(f"  {'Optimized ρ combo':<33} {opt_rp:>7.3f} {opt_rs:>7.3f} {opt_nn:>3d}/{nc}   {opt_rc:>6.3f}")
print(f"  {'Optimized NN combo':<33} {nn_rp:>7.3f} {nn_rs:>7.3f} {nn_nn:>3d}/{nc}   {nn_rc:>6.3f}")

print(f"\nOptimal ρ weights:")
for j in range(n_methods):
    if best_opt_weights[j] > 0.01:
        print(f"  {method_keys[j]:<20} {best_opt_weights[j]:.3f}")

print(f"\nOptimal NN weights:")
for j in range(n_methods):
    if best_nn_weights[j] > 0.01:
        print(f"  {method_keys[j]:<20} {best_nn_weights[j]:.3f}")

print(f"\nCross-validated RMSE:")
for k in method_keys:
    print(f"  {k:<33} {cv_rmse_individual[k]:.4f}")
print(f"  {'Optimized combination':<33} {cv_rmse_optimal:.4f}")

print(f"\nFigures saved:")
print(f"  Fig CCC: processus_figCCC_method_overview.png")
print(f"  Fig DDD: processus_figDDD_optimal_dendrograms.png")
print(f"  Fig EEE: processus_figEEE_weight_sensitivity.png")
print(f"  Fig FFF: processus_figFFF_per_text_nn.png")
