#!/usr/bin/env python3
"""
Capstone Analysis: Comprehensive Synthesis
===========================================
Pulls together all methods, pair reports, and findings into a single
comparative analysis. Addresses:

  1. Per-text, per-method accuracy — who gets what right?
  2. Spelling normalization vs original spelling
  3. What each method contributes to the overall picture
  4. Expert annotation feature analysis — which categories drive expert distances?
  5. Recommendations for future projects

Produces Figures MMM–PPP and the final documentation.
"""

import re
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
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('processus-universalis-graphics')
REPORT_DIR = Path('detailed_pair_reports')

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
print("Loading all data...")
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

# Expert annotations (full detail — category-level)
import xml.etree.ElementTree as ET
tree = ET.parse(XML_PATH)
root = tree.getroot()

anno_by_category = {}  # text → {category → set of values}
anno_features = {}     # text → flat set of features
for div in root.findall('div'):
    dtype = div.get('type', '')
    m_a = re.search(r'a(\d+)', dtype)
    if not m_a:
        continue
    a_key = 'a' + m_a.group(1)
    ename = A_TO_E.get(a_key)
    if not ename or ename not in GROUP_MAP:
        continue
    features = set()
    categories = defaultdict(set)
    for keys_el in div.findall('.//keys'):
        kvals = keys_el.get('n', '')
        ktype = keys_el.get('type', '')
        if kvals and 'FEHLT' not in kvals:
            for val in kvals.split(';'):
                val = val.strip()
                if val:
                    features.add(f"{ktype}::{val}")
                    categories[ktype].add(val)
    if features:
        anno_features[ename] = features
        anno_by_category[ename] = dict(categories)

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

# Expert NN for each text
expert_nn = {}
for i in range(nc):
    da = expert_dist[i].copy(); da[i] = np.inf
    expert_nn[common[i]] = common[np.argmin(da)]


# ══════════════════════════════════════════════════════════════
# COMPUTE ALL METHOD DISTANCES
# ══════════════════════════════════════════════════════════════
print("Computing all method distances...")

# ── Method A: Quadratic Delta (300 MFW) — raw spelling ──
all_tokens_flat = []
for nm in text_names:
    all_tokens_flat.extend(text_tokens[nm])
vocab_counts = Counter(all_tokens_flat)
MFW = 300
mfw_list = [w for w, _ in vocab_counts.most_common(MFW)]
features_raw = np.array([
    np.array([Counter(text_tokens[nm]).get(w, 0) / max(len(text_tokens[nm]), 1)
              for w in mfw_list])
    for nm in text_names
])
fm = features_raw
fm_z = (fm - fm.mean(0)) / np.where(fm.std(0) == 0, 1, fm.std(0))
dist_stylo_raw = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = np.sqrt(np.mean((fm_z[i] - fm_z[j])**2))
        dist_stylo_raw[i, j] = dist_stylo_raw[j, i] = d

# ── Method B: Quadratic Delta — Cologne-phonetic normalized ──
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

corpus_vocab = set()
for nm in text_names:
    corpus_vocab.update(text_tokens[nm])
phon_map = {w: cologne_phonetic(w) for w in corpus_vocab}

norm_tokens = {nm: [phon_map.get(t, t) for t in text_tokens[nm]] for nm in text_names}

all_norm_flat = []
for nm in text_names:
    all_norm_flat.extend(norm_tokens[nm])
norm_vocab = Counter(all_norm_flat)
mfw_norm = [w for w, _ in norm_vocab.most_common(MFW)]
features_norm = np.array([
    np.array([Counter(norm_tokens[nm]).get(w, 0) / max(len(norm_tokens[nm]), 1)
              for w in mfw_norm])
    for nm in text_names
])
fn = features_norm
fn_z = (fn - fn.mean(0)) / np.where(fn.std(0) == 0, 1, fn.std(0))
dist_stylo_norm = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = np.sqrt(np.mean((fn_z[i] - fn_z[j])**2))
        dist_stylo_norm[i, j] = dist_stylo_norm[j, i] = d

# ── Method C: 4-gram Jaccard — raw spelling ──
raw_ngrams = {}
for nm in text_names:
    toks = text_tokens[nm]
    raw_ngrams[nm] = set(tuple(toks[i:i+4]) for i in range(len(toks)-3))

dist_4gram_raw = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si, sj = raw_ngrams[text_names[i]], raw_ngrams[text_names[j]]
        u = len(si | sj)
        dist_4gram_raw[i, j] = dist_4gram_raw[j, i] = 1 - (len(si & sj) / u if u > 0 else 0)

# ── Method D: 4-gram Jaccard — Cologne-phonetic normalized ──
norm_ngrams = {}
for nm in text_names:
    toks = norm_tokens[nm]
    norm_ngrams[nm] = set(tuple(toks[i:i+4]) for i in range(len(toks)-3))

dist_4gram_norm = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si, sj = norm_ngrams[text_names[i]], norm_ngrams[text_names[j]]
        u = len(si | sj)
        dist_4gram_norm[i, j] = dist_4gram_norm[j, i] = 1 - (len(si & sj) / u if u > 0 else 0)

# ── Method E: text-matcher ──
from text_matcher.matcher import ExtendedMatch
from difflib import SequenceMatcher
from nltk.metrics.distance import edit_distance as editDistance

class GermanText:
    def __init__(self, raw_text, label):
        self.text = raw_text
        self.label = label
        pat = re.compile(r'[a-zA-ZäöüÄÖÜß]+')
        spans_iter = list(pat.finditer(self.text))
        self.spans = [(m.start(), m.end()) for m in spans_iter]
        self.tokens = [self.text[s:e].lower() for s, e in self.spans]
        self.length = self.spans[-1][-1] if self.spans else 1
    def ngrams(self, nn):
        return [tuple(self.tokens[i:i+nn]) for i in range(len(self.tokens) - nn + 1)]

class GermanMatcher:
    def __init__(self, textA, textB, threshold=3, cutoff=5, ngram_size=3):
        self.textA = textA; self.textB = textB; self.ngram_size = ngram_size
        self.textAgrams = textA.ngrams(ngram_size)
        self.textBgrams = textB.ngrams(ngram_size)
        seq = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        blocks = seq.get_matching_blocks()
        initial = [m for m in blocks if m.size > threshold]
        healed = self._heal(initial)
        extended = self._extend(healed)
        self.extended_matches = [m for m in extended if min(m.sizeA, m.sizeB) >= cutoff]
        self.numMatches = len(self.extended_matches)
    def _heal(self, matches, min_dist=8):
        healed = []; skip = False
        if len(matches) <= 1:
            return [ExtendedMatch(m.a, m.b, m.size, m.size) for m in matches]
        for i in range(len(matches)):
            if skip: skip = False; continue
            m = matches[i]
            if i+1 < len(matches):
                nxt = matches[i+1]
                if (nxt.a - (m.a + m.size)) < min_dist:
                    em = ExtendedMatch(m.a, m.b, (nxt.a+nxt.size)-m.a, (nxt.b+nxt.size)-m.b)
                    em.healed = True; healed.append(em); skip = True
                else: healed.append(ExtendedMatch(m.a, m.b, m.size, m.size))
            else: healed.append(ExtendedMatch(m.a, m.b, m.size, m.size))
        return healed
    def _extend(self, matches, cutoff=0.4):
        extended = False
        for m in matches:
            if m.a > 0 and m.b > 0:
                wA, wB = self.textAgrams[m.a-1][0], self.textBgrams[m.b-1][0]
                d = editDistance(wA, wB); avg = (len(wA)+len(wB))/2
                if avg > 0 and d/avg < cutoff:
                    m.a -= 1; m.b -= 1; m.sizeA += 1; m.sizeB += 1; extended = True
            idxA, idxB = m.a+m.sizeA+1, m.b+m.sizeB+1
            if idxA < len(self.textAgrams) and idxB < len(self.textBgrams):
                wA, wB = self.textAgrams[idxA][-1], self.textBgrams[idxB][-1]
                d = editDistance(wA, wB); avg = (len(wA)+len(wB))/2
                if avg > 0 and d/avg < cutoff:
                    m.sizeA += 1; m.sizeB += 1; extended = True
        if extended: self._extend(matches)
        return matches

print("  Running text-matcher...")
gt = {nm: GermanText(plain_texts[nm], nm) for nm in text_names}
tm_score = np.zeros((n, n))
pc = 0
for i in range(n):
    for j in range(i+1, n):
        pc += 1
        na, nb = text_names[i], text_names[j]
        matcher = GermanMatcher(gt[na], gt[nb])
        total = sum(m.sizeA + 3 - 1 for m in matcher.extended_matches)
        norm = min(len(gt[na].tokens), len(gt[nb].tokens))
        tm_score[i, j] = tm_score[j, i] = total / norm if norm > 0 else 0
        if pc % 30 == 0: print(f"    {pc}/136...")
dist_tm = 1 - tm_score / (tm_score.max() + 1e-10)
print("  Done.")

# ── Method F: Sentence embeddings ──
print("  Computing embeddings...")
try:
    from sentence_transformers import SentenceTransformer
    from numpy.linalg import norm as np_norm
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    def chunk_embed(text, cs=80, ov=40):
        words = text.split()
        chunks = [' '.join(words[i:i+cs]) for i in range(0, max(1, len(words)-cs+1), cs-ov)]
        if not chunks: chunks = [text]
        return model.encode(chunks)
    emb = {}
    emb_early = {}
    for nm in text_names:
        e = chunk_embed(plain_texts[nm])
        emb[nm] = np.mean(e, axis=0)
        mid = len(e)//2
        emb_early[nm] = np.mean(e[:max(1,mid)], axis=0)

    dist_emb = np.zeros((n, n))
    dist_emb_early = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            a, b = emb[text_names[i]], emb[text_names[j]]
            sim = np.dot(a,b)/(np_norm(a)*np_norm(b)+1e-10)
            dist_emb[i,j] = dist_emb[j,i] = max(0, 1-sim)
            a2, b2 = emb_early[text_names[i]], emb_early[text_names[j]]
            sim2 = np.dot(a2,b2)/(np_norm(a2)*np_norm(b2)+1e-10)
            dist_emb_early[i,j] = dist_emb_early[j,i] = max(0, 1-sim2)
    HAS_EMB = True
    print("  Done.")
except ImportError:
    dist_emb = np.zeros((n,n)); dist_emb_early = np.zeros((n,n)); HAS_EMB = False

# ── Best combined (from previous work): 3% stylo + 97% 4gram ──
def normalize(d, cidx):
    dc = d[np.ix_(cidx, cidx)]
    flat = upper_tri(dc)
    mn, mx = flat.min(), flat.max()
    if mx - mn < 1e-10: return dc
    return (dc - mn) / (mx - mn)

d_s = normalize(dist_stylo_raw, cidx)
d_4 = normalize(dist_4gram_raw, cidx)
dist_combined = 0.03 * d_s + 0.97 * d_4


# ══════════════════════════════════════════════════════════════
# EVALUATION FRAMEWORK
# ══════════════════════════════════════════════════════════════

def evaluate(dist_c, label=""):
    flat_d = upper_tri(dist_c)
    r_p, _ = pearsonr(flat_d, expert_flat)
    r_s, _ = spearmanr(flat_d, expert_flat)
    nn_agree = 0
    nn_details = {}
    for i in range(nc):
        dm = dist_c[i].copy(); dm[i] = np.inf
        da = expert_dist[i].copy(); da[i] = np.inf
        method_nn = common[np.argmin(dm)]
        expert_nn_i = common[np.argmin(da)]
        correct = method_nn == expert_nn_i
        if correct: nn_agree += 1
        nn_details[common[i]] = {
            'method_nn': method_nn, 'expert_nn': expert_nn_i,
            'correct': correct
        }
    cond_m = squareform(dist_c, checks=False)
    cond_e = squareform(expert_dist, checks=False)
    Z_m = linkage(cond_m, method='ward')
    Z_e = linkage(cond_e, method='ward')
    cm = cophenet(Z_m)
    ce = cophenet(Z_e)
    r_c, _ = pearsonr(cm, ce)
    return {'r': r_p, 'rho': r_s, 'nn': nn_agree, 'coph': r_c,
            'nn_details': nn_details}


# Build all method results
all_methods = {}

method_defs = [
    ('Quad.Delta raw', dist_stylo_raw),
    ('Quad.Delta normalized', dist_stylo_norm),
    ('4-gram raw', dist_4gram_raw),
    ('4-gram normalized', dist_4gram_norm),
    ('text-matcher', dist_tm),
]
if HAS_EMB:
    method_defs.append(('Embedding full', dist_emb))
    method_defs.append(('Embedding early-half', dist_emb_early))

for label, dist_full in method_defs:
    dist_c = dist_full[np.ix_(cidx, cidx)]
    all_methods[label] = evaluate(dist_c, label)
    all_methods[label]['dist'] = dist_c

# Combined
all_methods['Combined (3%S+97%4g)'] = evaluate(dist_combined)
all_methods['Combined (3%S+97%4g)']['dist'] = dist_combined

print("\n" + "=" * 80)
print("ALL METHODS EVALUATED")
print("=" * 80)
print(f"{'Method':<30} {'r':>7} {'ρ':>7} {'NN':>7} {'Coph':>7}")
print("-" * 62)
for label in all_methods:
    r = all_methods[label]
    print(f"  {label:<28} {r['r']:>7.3f} {r['rho']:>7.3f} {r['nn']:>3d}/{nc}   {r['coph']:>6.3f}")


# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: Spelling normalization impact
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 1: Spelling normalization impact")
print("=" * 80)

# Find words that collapse under Cologne phonetic
variant_groups = defaultdict(set)
for w in corpus_vocab:
    code = phon_map[w]
    variant_groups[code].add(w)

# Focus on high-impact collapses
big_groups = {code: words for code, words in variant_groups.items()
              if len(words) >= 3 and any(w in [w2 for w2, _ in vocab_counts.most_common(100)] for w in words)}

print(f"\nHigh-frequency words with spelling variants (Cologne phonetic):")
for code, words in sorted(big_groups.items(), key=lambda x: -max(vocab_counts[w] for w in x[1]))[:20]:
    word_list = sorted(words, key=lambda w: -vocab_counts[w])[:6]
    total = sum(vocab_counts[w] for w in words)
    print(f"  [{code}] ({total}x): {', '.join(f'{w}({vocab_counts[w]})' for w in word_list)}")

# Per-text: which texts are most affected by normalization?
print(f"\nPer-text normalization impact (vocabulary reduction):")
for nm in text_names:
    raw_vocab = set(text_tokens[nm])
    norm_vocab_set = set(norm_tokens[nm])
    reduction = 1 - len(norm_vocab_set) / len(raw_vocab) if raw_vocab else 0
    print(f"  {nm} ({GROUP_MAP[nm]}): {len(raw_vocab)} → {len(norm_vocab_set)} "
          f"unique forms ({reduction:.1%} reduction)")

# Direct comparison: raw vs normalized
print(f"\nDirect comparison (raw vs normalized):")
print(f"  Quad.Delta raw:        ρ={all_methods['Quad.Delta raw']['rho']:.3f}, "
      f"NN={all_methods['Quad.Delta raw']['nn']}/{nc}")
print(f"  Quad.Delta normalized: ρ={all_methods['Quad.Delta normalized']['rho']:.3f}, "
      f"NN={all_methods['Quad.Delta normalized']['nn']}/{nc}")
print(f"  4-gram raw:            ρ={all_methods['4-gram raw']['rho']:.3f}, "
      f"NN={all_methods['4-gram raw']['nn']}/{nc}")
print(f"  4-gram normalized:     ρ={all_methods['4-gram normalized']['rho']:.3f}, "
      f"NN={all_methods['4-gram normalized']['nn']}/{nc}")

# Which NN assignments change?
for method_pair in [('Quad.Delta raw', 'Quad.Delta normalized'),
                    ('4-gram raw', '4-gram normalized')]:
    raw_r = all_methods[method_pair[0]]
    norm_r = all_methods[method_pair[1]]
    changes = []
    for nm in common:
        raw_nn = raw_r['nn_details'][nm]['method_nn']
        norm_nn = norm_r['nn_details'][nm]['method_nn']
        exp_nn = raw_r['nn_details'][nm]['expert_nn']
        if raw_nn != norm_nn:
            raw_correct = "correct" if raw_nn == exp_nn else "wrong"
            norm_correct = "correct" if norm_nn == exp_nn else "wrong"
            changes.append(f"    {nm}: {raw_nn}({raw_correct}) → {norm_nn}({norm_correct})")
    if changes:
        print(f"\n  NN changes from normalization ({method_pair[0]} → {method_pair[1]}):")
        for c in changes:
            print(c)


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: Expert annotation categories
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 2: Expert annotation categories")
print("=" * 80)

# What categories exist?
all_categories = set()
for nm in common:
    all_categories.update(anno_by_category.get(nm, {}).keys())
all_categories = sorted(all_categories)

print(f"\n{len(all_categories)} annotation categories found:")
for cat in all_categories:
    n_texts = sum(1 for nm in common if cat in anno_by_category.get(nm, {}))
    n_values = len(set(v for nm in common for v in anno_by_category.get(nm, {}).get(cat, [])))
    print(f"  {cat}: {n_texts} texts, {n_values} distinct values")

# Per-category distances: which categories drive expert clustering?
category_correlations = {}
for cat in all_categories:
    cat_dist = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(i+1, nc):
            vals_i = anno_by_category.get(common[i], {}).get(cat, set())
            vals_j = anno_by_category.get(common[j], {}).get(cat, set())
            u = len(vals_i | vals_j)
            if u > 0:
                cat_dist[i, j] = cat_dist[j, i] = 1 - len(vals_i & vals_j) / u
            else:
                cat_dist[i, j] = cat_dist[j, i] = 0.5
    cat_flat = upper_tri(cat_dist)
    # Correlation with each computational method
    for method_label in all_methods:
        if method_label not in category_correlations:
            category_correlations[method_label] = {}
        method_flat = upper_tri(all_methods[method_label]['dist'])
        try:
            r, _ = pearsonr(method_flat, cat_flat)
        except:
            r = 0
        category_correlations[method_label][cat] = r

print(f"\nWhich annotation categories does each method capture best?")
for method_label in ['Quad.Delta raw', '4-gram raw', 'text-matcher',
                     'Combined (3%S+97%4g)']:
    if method_label not in category_correlations:
        continue
    cats = category_correlations[method_label]
    top = sorted(cats.items(), key=lambda x: -abs(x[1]))[:5]
    print(f"\n  {method_label}:")
    for cat, r in top:
        print(f"    {cat}: r={r:.3f}")


# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: Per-text, per-method diagnostic
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 3: Per-text diagnostic — which method helps which text?")
print("=" * 80)

# Build a table: for each text, which methods get its NN right?
method_labels_ordered = [
    'Quad.Delta raw', 'Quad.Delta normalized',
    '4-gram raw', '4-gram normalized',
    'text-matcher',
]
if HAS_EMB:
    method_labels_ordered += ['Embedding full', 'Embedding early-half']
method_labels_ordered.append('Combined (3%S+97%4g)')

nn_matrix = np.zeros((nc, len(method_labels_ordered)), dtype=int)
for mi, ml in enumerate(method_labels_ordered):
    for ti, nm in enumerate(common):
        nn_matrix[ti, mi] = 1 if all_methods[ml]['nn_details'][nm]['correct'] else 0

print(f"\nPer-text NN accuracy:")
print(f"  {'Text':<8} {'Gr':>3} {'Expert NN':<8}", end="")
for ml in method_labels_ordered:
    short = ml[:10]
    print(f" {short:>10}", end="")
print()
print("-" * (25 + 11 * len(method_labels_ordered)))
for ti, nm in enumerate(common):
    print(f"  {nm:<8} {GROUP_MAP[nm]:>3} {expert_nn[nm]:<8}", end="")
    for mi in range(len(method_labels_ordered)):
        mark = "Y" if nn_matrix[ti, mi] else "."
        print(f" {mark:>10}", end="")
    print()

# "Easy" texts (most methods agree) vs "hard" texts
easy = [nm for ti, nm in enumerate(common) if nn_matrix[ti].sum() >= len(method_labels_ordered) - 1]
hard = [nm for ti, nm in enumerate(common) if nn_matrix[ti].sum() <= 2]
print(f"\n  Easy texts (nearly all methods correct): {', '.join(easy) if easy else 'none'}")
print(f"  Hard texts (almost no method correct):   {', '.join(hard) if hard else 'none'}")

# What makes hard texts hard?
for nm in hard:
    print(f"\n  Why is {nm} hard?")
    print(f"    Expert NN: {expert_nn[nm]}")
    print(f"    Token count: {len(text_tokens[nm])}")
    print(f"    Gruppe: {GROUP_MAP[nm]}")
    for ml in method_labels_ordered:
        d = all_methods[ml]['nn_details'][nm]
        print(f"    {ml}: → {d['method_nn']} "
              f"({'correct' if d['correct'] else 'WRONG'})")


# ══════════════════════════════════════════════════════════════
# FIGURE MMM: Grand method comparison
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure MMM: Grand method comparison...")

fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Panel 1: Bar chart of all methods
ax = fig.add_subplot(gs[0, 0])
labels = list(all_methods.keys())
rhos = [all_methods[l]['rho'] for l in labels]
rs = [all_methods[l]['r'] for l in labels]
nns = [all_methods[l]['nn'] / nc for l in labels]
cophs = [all_methods[l]['coph'] for l in labels]

x = np.arange(len(labels))
w = 0.2
ax.bar(x - 1.5*w, rs, w, label='Pearson r', color='#e74c3c', alpha=0.8)
ax.bar(x - 0.5*w, rhos, w, label='Spearman ρ', color='#3498db', alpha=0.8)
ax.bar(x + 0.5*w, nns, w, label='NN rate', color='#2ecc71', alpha=0.8)
ax.bar(x + 1.5*w, cophs, w, label='Coph r', color='#9b59b6', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Score')
ax.legend(fontsize=7, loc='upper left')
ax.set_title('All Methods vs Expert', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)

# Panel 2: Normalization impact
ax = fig.add_subplot(gs[0, 1])
pairs = [
    ('Quad.Delta\nraw', 'Quad.Delta\nnormalized',
     all_methods['Quad.Delta raw'], all_methods['Quad.Delta normalized']),
    ('4-gram\nraw', '4-gram\nnormalized',
     all_methods['4-gram raw'], all_methods['4-gram normalized']),
]
x = np.arange(len(pairs))
for i, (l_raw, l_norm, r_raw, r_norm) in enumerate(pairs):
    ax.bar(i - 0.15, r_raw['rho'], 0.25, color='#e74c3c', alpha=0.8,
           label='Raw spelling' if i == 0 else '')
    ax.bar(i + 0.15, r_norm['rho'], 0.25, color='#3498db', alpha=0.8,
           label='Normalized' if i == 0 else '')
    ax.text(i - 0.15, r_raw['rho'] + 0.01, f"NN={r_raw['nn']}", ha='center', fontsize=8)
    ax.text(i + 0.15, r_norm['rho'] + 0.01, f"NN={r_norm['nn']}", ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['Quadratic Delta', '4-gram Jaccard'], fontsize=10)
ax.set_ylabel('Spearman ρ')
ax.legend(fontsize=9)
ax.set_title('Normalization: Raw vs Cologne Phonetic', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)

# Panel 3: Per-text NN heatmap
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(nn_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(method_labels_ordered)))
ax.set_xticklabels([l.replace(' ', '\n')[:15] for l in method_labels_ordered],
                    fontsize=7, rotation=45, ha='right')
ax.set_yticks(range(nc))
ax.set_yticklabels(common, fontsize=8)
for lbl in ax.get_yticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
# Column totals
for mi in range(len(method_labels_ordered)):
    total = nn_matrix[:, mi].sum()
    ax.text(mi, nc + 0.3, f'{total}', ha='center', va='top', fontsize=8, fontweight='bold')
ax.set_title('Per-Text NN Accuracy\n(green=correct, red=wrong)', fontsize=11, fontweight='bold')

# Panel 4: Method contribution Venn-like analysis
ax = fig.add_subplot(gs[1, 0])
# Which texts does each method UNIQUELY get right?
method_correct = {}
for mi, ml in enumerate(method_labels_ordered):
    method_correct[ml] = set(common[ti] for ti in range(nc) if nn_matrix[ti, mi])

# Count unique contributions
unique_contributions = {}
for ml in method_labels_ordered:
    others = set()
    for ml2 in method_labels_ordered:
        if ml2 != ml:
            others |= method_correct[ml2]
    unique = method_correct[ml] - others
    unique_contributions[ml] = unique

# Also: texts only the combination gets right
combo_correct = method_correct.get('Combined (3%S+97%4g)', set())
any_individual = set()
for ml in method_labels_ordered:
    if ml != 'Combined (3%S+97%4g)':
        any_individual |= method_correct[ml]
combo_unique = combo_correct - any_individual

bar_data = [(ml.replace(' ', '\n')[:20], len(method_correct[ml]),
             len(unique_contributions[ml]))
            for ml in method_labels_ordered]
x = np.arange(len(bar_data))
ax.bar(x, [d[1] for d in bar_data], color='#3498db', alpha=0.7, label='Total correct')
ax.bar(x, [d[2] for d in bar_data], color='#e74c3c', alpha=0.8, label='Uniquely correct')
ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in bar_data], fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Number of texts')
ax.legend(fontsize=8)
ax.set_title('Method Contributions\n(red = only this method gets it right)',
             fontsize=11, fontweight='bold')

# Panel 5: Per-Gruppe performance
ax = fig.add_subplot(gs[1, 1])
for gi, gruppe in enumerate(['I', 'II', 'III']):
    gruppe_texts = [nm for nm in common if GROUP_MAP[nm] == gruppe]
    gruppe_idx = [common.index(nm) for nm in gruppe_texts]
    for mi, ml in enumerate(method_labels_ordered):
        correct = sum(1 for ti in gruppe_idx if nn_matrix[ti, mi])
        total = len(gruppe_idx)
        rate = correct / total if total > 0 else 0
        ax.scatter(mi + gi * 0.15 - 0.15, rate, s=total * 30 + 20,
                   c=GROUP_COLORS[gruppe], alpha=0.7,
                   edgecolors='black', linewidths=0.5)

ax.set_xticks(range(len(method_labels_ordered)))
ax.set_xticklabels([l.replace(' ', '\n')[:15] for l in method_labels_ordered],
                    fontsize=7, rotation=45, ha='right')
ax.set_ylabel('NN accuracy rate')
ax.set_title('Per-Gruppe Performance\n(size = group size)', fontsize=11, fontweight='bold')
legend_h = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
ax.legend(handles=legend_h, fontsize=8)
ax.set_ylim(-0.1, 1.1)

# Panel 6: Expert category coverage
ax = fig.add_subplot(gs[1, 2])
# Heatmap: methods x categories (correlation)
methods_for_heatmap = ['Quad.Delta raw', '4-gram raw', 'text-matcher']
if HAS_EMB:
    methods_for_heatmap.append('Embedding full')
methods_for_heatmap.append('Combined (3%S+97%4g)')

cats_with_data = [cat for cat in all_categories
                  if sum(1 for nm in common if cat in anno_by_category.get(nm, {})) >= 5]

heat_data = np.zeros((len(methods_for_heatmap), len(cats_with_data)))
for mi, ml in enumerate(methods_for_heatmap):
    for ci, cat in enumerate(cats_with_data):
        heat_data[mi, ci] = category_correlations.get(ml, {}).get(cat, 0)

im = ax.imshow(heat_data, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax.set_xticks(range(len(cats_with_data)))
ax.set_xticklabels(cats_with_data, fontsize=6, rotation=90)
ax.set_yticks(range(len(methods_for_heatmap)))
ax.set_yticklabels([l.replace(' ', '\n')[:20] for l in methods_for_heatmap], fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
ax.set_title('Which Expert Categories\nDoes Each Method Capture?', fontsize=11, fontweight='bold')

fig.suptitle("Capstone Analysis: All Methods, All Evidence",
             fontsize=16, fontweight='bold')
plt.savefig(OUT_DIR / 'processus_figMMM_capstone_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig MMM saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE NNN: The spelling question in detail
# ══════════════════════════════════════════════════════════════
print("Generating Figure NNN: Spelling normalization deep dive...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel 1: Top spelling variants and their frequencies per text
ax = axes[0, 0]
# Focus on und/undt/vndt/vnd type variants
und_group = [w for w in corpus_vocab if phon_map[w] == cologne_phonetic('und')]
und_group_sorted = sorted(und_group, key=lambda w: -vocab_counts[w])[:8]

# Per-text frequencies of und variants
freq_data = np.zeros((n, len(und_group_sorted)))
for ti, nm in enumerate(text_names):
    tok_counts = Counter(text_tokens[nm])
    total = len(text_tokens[nm])
    for wi, w in enumerate(und_group_sorted):
        freq_data[ti, wi] = tok_counts.get(w, 0) / total * 100

x = np.arange(n)
bottom = np.zeros(n)
colors_und = plt.cm.Set2(np.linspace(0, 1, len(und_group_sorted)))
for wi, w in enumerate(und_group_sorted):
    ax.bar(x, freq_data[:, wi], bottom=bottom, label=w,
           color=colors_und[wi], alpha=0.8)
    bottom += freq_data[:, wi]

ax.set_xticks(x)
ax.set_xticklabels(text_names, fontsize=8, rotation=45, ha='right')
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax.set_ylabel('Frequency (%)')
ax.set_title(f'"und" and its spelling variants\n({", ".join(und_group_sorted[:5])}...)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=7, ncol=4)

# Panel 2: Scatter — raw vs normalized distances
ax = axes[0, 1]
raw_flat = upper_tri(dist_stylo_raw[np.ix_(cidx, cidx)])
norm_flat = upper_tri(dist_stylo_norm[np.ix_(cidx, cidx)])
pair_list = list(zip(*np.triu_indices(nc, k=1)))
for pi in range(len(raw_flat)):
    i, j = pair_list[pi]
    same = GROUP_MAP[common[i]] == GROUP_MAP[common[j]]
    ax.scatter(raw_flat[pi], norm_flat[pi],
               c='green' if same else 'red', s=12, alpha=0.5)
ax.plot([raw_flat.min(), raw_flat.max()], [raw_flat.min(), raw_flat.max()], 'k:', alpha=0.3)
ax.set_xlabel('Quad.Delta distance (raw spelling)', fontsize=10)
ax.set_ylabel('Quad.Delta distance (normalized)', fontsize=10)
ax.set_title('How Normalization Changes Distances\n(green=same Gruppe, red=cross)',
             fontsize=11, fontweight='bold')

# Panel 3: Dendrograms side by side — raw vs normalized
ax = axes[1, 0]
cond = squareform(dist_stylo_raw[np.ix_(cidx, cidx)], checks=False)
Z = linkage(cond, method='ward')
dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=8)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP: lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
r_raw = all_methods['Quad.Delta raw']
ax.set_title(f'Quad.Delta — Raw Spelling\nρ={r_raw["rho"]:.3f}, NN={r_raw["nn"]}/{nc}',
             fontsize=11, fontweight='bold')

ax = axes[1, 1]
cond = squareform(dist_stylo_norm[np.ix_(cidx, cidx)], checks=False)
Z = linkage(cond, method='ward')
dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=8)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP: lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
r_norm = all_methods['Quad.Delta normalized']
ax.set_title(f'Quad.Delta — Cologne Normalized\nρ={r_norm["rho"]:.3f}, NN={r_norm["nn"]}/{nc}',
             fontsize=11, fontweight='bold')

fig.suptitle("The Spelling Question: Should You Normalize?",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / 'processus_figNNN_spelling_normalization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig NNN saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE OOO: Per-text deep dive — who benefits from what?
# ══════════════════════════════════════════════════════════════
print("Generating Figure OOO: Per-text deep dive...")

fig, axes = plt.subplots(3, 6, figsize=(30, 15))
axes = axes.flatten()

for ti, nm in enumerate(common):
    if ti >= 17: break
    ax = axes[ti]

    # Show this text's distance to all others, by method vs expert
    expert_dists = expert_dist[ti].copy()
    expert_dists[ti] = np.nan

    others = [common[j] for j in range(nc) if j != ti]
    other_idx = [j for j in range(nc) if j != ti]

    # Sort by expert distance
    order = np.argsort([expert_dist[ti, j] for j in other_idx])
    sorted_others = [others[k] for k in order]
    sorted_expert = [expert_dist[ti, other_idx[k]] for k in order]

    # Plot expert distances as reference
    y = np.arange(len(sorted_others))
    ax.barh(y, sorted_expert, height=0.8, color='lightgray', alpha=0.5, label='Expert')

    # Mark expert NN
    ax.barh(0, sorted_expert[0], height=0.8, color='gold', alpha=0.7)

    # Overlay method NNs as markers
    method_markers = [
        ('Quad.Delta raw', 's', '#e74c3c'),
        ('4-gram raw', 'o', '#3498db'),
        ('text-matcher', '^', '#2ecc71'),
        ('Combined (3%S+97%4g)', '*', '#f39c12'),
    ]

    for ml, marker, color in method_markers:
        method_nn = all_methods[ml]['nn_details'][nm]['method_nn']
        if method_nn in sorted_others:
            nn_pos = sorted_others.index(method_nn)
            ax.scatter(sorted_expert[nn_pos] + 0.02, nn_pos, marker=marker,
                      c=color, s=60, zorder=5, edgecolors='black', linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_others, fontsize=6)
    for lbl in ax.get_yticklabels():
        lbl_nm = lbl.get_text()
        if lbl_nm in GROUP_MAP:
            lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl_nm]])

    title_color = GROUP_COLORS[GROUP_MAP[nm]]
    ax.set_title(f'{nm} ({GROUP_MAP[nm]})', fontsize=10, fontweight='bold',
                color=title_color)
    ax.invert_yaxis()
    if ti == 0:
        legend_elements = [Patch(facecolor='gold', label='Expert NN')]
        for ml, marker, color in method_markers:
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                          markerfacecolor=color, markersize=8,
                                          label=ml[:15]))
        ax.legend(handles=legend_elements, fontsize=5, loc='lower right')

# Hide unused axes
for ti in range(nc, len(axes)):
    axes[ti].set_visible(False)

fig.suptitle("Per-Text Deep Dive: Where Does Each Method Place the Nearest Neighbor?",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / 'processus_figOOO_per_text_dive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig OOO saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE PPP: Recommendation flowchart data
# ══════════════════════════════════════════════════════════════
print("Generating Figure PPP: Method recommendation summary...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left: Method strengths/weaknesses summary
categories_summary = {
    'Quad.Delta\n(raw)': {
        'Tree shape': all_methods['Quad.Delta raw']['coph'],
        'Rank order': all_methods['Quad.Delta raw']['rho'],
        'NN accuracy': all_methods['Quad.Delta raw']['nn'] / nc,
        'Linear dist': all_methods['Quad.Delta raw']['r'],
    },
    'Quad.Delta\n(normalized)': {
        'Tree shape': all_methods['Quad.Delta normalized']['coph'],
        'Rank order': all_methods['Quad.Delta normalized']['rho'],
        'NN accuracy': all_methods['Quad.Delta normalized']['nn'] / nc,
        'Linear dist': all_methods['Quad.Delta normalized']['r'],
    },
    '4-gram\n(raw)': {
        'Tree shape': all_methods['4-gram raw']['coph'],
        'Rank order': all_methods['4-gram raw']['rho'],
        'NN accuracy': all_methods['4-gram raw']['nn'] / nc,
        'Linear dist': all_methods['4-gram raw']['r'],
    },
    'text-\nmatcher': {
        'Tree shape': all_methods['text-matcher']['coph'],
        'Rank order': all_methods['text-matcher']['rho'],
        'NN accuracy': all_methods['text-matcher']['nn'] / nc,
        'Linear dist': all_methods['text-matcher']['r'],
    },
    'Combined': {
        'Tree shape': all_methods['Combined (3%S+97%4g)']['coph'],
        'Rank order': all_methods['Combined (3%S+97%4g)']['rho'],
        'NN accuracy': all_methods['Combined (3%S+97%4g)']['nn'] / nc,
        'Linear dist': all_methods['Combined (3%S+97%4g)']['r'],
    },
}

method_names = list(categories_summary.keys())
metric_names = list(categories_summary[method_names[0]].keys())
data = np.array([[categories_summary[m][metric] for metric in metric_names]
                  for m in method_names])

# Radar-style as grouped bars
x = np.arange(len(metric_names))
width = 0.15
for mi, mn in enumerate(method_names):
    offset = (mi - len(method_names)/2 + 0.5) * width
    ax1.bar(x + offset, data[mi], width, label=mn, alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(metric_names, fontsize=10)
ax1.set_ylabel('Score (higher = better)')
ax1.legend(fontsize=8, loc='upper left')
ax1.set_title('Method Strengths by Metric', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1)

# Right: Decision tree text
ax2.axis('off')
raw_s = all_methods['Quad.Delta raw']
norm_s = all_methods['Quad.Delta normalized']
raw_4 = all_methods['4-gram raw']
norm_4 = all_methods['4-gram normalized']

# Determine normalization verdict dynamically
if norm_s['rho'] > raw_s['rho'] and raw_s['nn'] > norm_s['nn']:
    stylo_verdict = "A TRADEOFF"
    stylo_detail = (f"    • Normalized improves rank ordering (ρ={norm_s['rho']:.3f} vs {raw_s['rho']:.3f})\n"
                    f"    • But raw has better NN ({raw_s['nn']} vs {norm_s['nn']})\n"
                    f"    • Recommendation: try both, evaluate on your corpus")
elif norm_s['rho'] > raw_s['rho']:
    stylo_verdict = "NORMALIZE"
    stylo_detail = f"    • Normalized: ρ={norm_s['rho']:.3f} vs Raw: ρ={raw_s['rho']:.3f}"
else:
    stylo_verdict = "DO NOT NORMALIZE"
    stylo_detail = f"    • Raw: ρ={raw_s['rho']:.3f} vs Normalized: ρ={norm_s['rho']:.3f}"

decision_text = f"""
RECOMMENDATION FOR FUTURE PROJECTS
(when expert annotations are unavailable)

Step 1: ESTABLISH THE TREE
  Use 4-gram Jaccard on raw (unnormalized) text
  → Best NN accuracy as a single method
  → Captures phrasal overlap diagnostic of copying

Step 2: REFINE WITH STYLOMETRY
  Add ~3% Quadratic Delta (300 MFW)
  → Small improvement in rank ordering
  → Spelling differences may be informative

Step 3: EXAMINE CLOSE PAIRS
  Run text-matcher on pairs the tree identifies as close
  → Shows WHAT passages are shared and WHERE
  → Generates human-readable evidence
  → Use HTML pair reports for scholarly exploration

Step 4 (optional): SEMANTIC VALIDATION
  Sentence embeddings on the early half of texts
  → Independent check from a different angle
  → Useful for catching false positives
  → Late-half embeddings are NOT useful (texts converge)

NORMALIZE OR NOT?
  For stylometry: {stylo_verdict}
{stylo_detail}

  For 4-grams: MIXED — try both
    • Raw: ρ={raw_4['rho']:.3f} vs Normalized: ρ={norm_4['rho']:.3f}
    • Nearly identical; try both on your corpus
    • Raw is simpler and nearly as good

  For text-matcher: built-in fuzzy matching handles it
    • Edit-distance extension bridges spelling variants
    • No external normalization needed
"""
ax2.text(0.05, 0.95, decision_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle("Method Recommendations: What to Use and When",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / 'processus_figPPP_recommendations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig PPP saved.")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CAPSTONE SUMMARY")
print("=" * 80)

print(f"\nAll methods ranked by Spearman ρ:")
ranked = sorted(all_methods.items(), key=lambda x: -x[1]['rho'])
for rank, (label, r) in enumerate(ranked):
    print(f"  {rank+1}. {label:<30} ρ={r['rho']:.3f}  NN={r['nn']}/{nc}  "
          f"r={r['r']:.3f}  coph={r['coph']:.3f}")

print(f"\nSpelling normalization verdict:")
raw_rho_s = all_methods['Quad.Delta raw']['rho']
norm_rho_s = all_methods['Quad.Delta normalized']['rho']
raw_nn_s = all_methods['Quad.Delta raw']['nn']
norm_nn_s = all_methods['Quad.Delta normalized']['nn']
if norm_rho_s > raw_rho_s:
    print(f"  Stylometry: normalized has HIGHER rho (ρ={norm_rho_s:.3f} vs raw {raw_rho_s:.3f}), "
          f"but NN: raw={raw_nn_s} vs norm={norm_nn_s} — a TRADEOFF")
else:
    print(f"  Stylometry: raw is BETTER (ρ={raw_rho_s:.3f} vs {norm_rho_s:.3f})")
print(f"  4-gram: negligible difference (ρ={all_methods['4-gram raw']['rho']:.3f} vs "
      f"{all_methods['4-gram normalized']['rho']:.3f})")

print(f"\nFigures saved:")
print(f"  Fig MMM: processus_figMMM_capstone_overview.png")
print(f"  Fig NNN: processus_figNNN_spelling_normalization.png")
print(f"  Fig OOO: processus_figOOO_per_text_dive.png")
print(f"  Fig PPP: processus_figPPP_recommendations.png")
