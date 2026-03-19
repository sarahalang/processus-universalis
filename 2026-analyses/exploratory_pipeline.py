#!/usr/bin/env python3
"""
Exploratory Pipeline: Big Picture → Detail
===========================================
A research pipeline ordered by zoom level:

  STAGE 1: Stylometry → overall tree structure (the forest)
  STAGE 2: 4-gram overlap → refine within-group relationships (the trees)
  STAGE 3: Combined tree → best quantitative clustering
  STAGE 4: text-matcher → for each close pair, WHAT is shared (the branches)
  STAGE 5: Detailed exploration reports (the leaves)

Outputs:
  - Figures JJJ–LLL (overview graphics)
  - detailed_pair_reports/  (per-pair HTML exploration files)
  - EXPLORATION_REPORT.md   (master document linking everything)
"""

import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('processus-universalis-graphics')
OUT_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path('detailed_pair_reports')
REPORT_DIR.mkdir(exist_ok=True)

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

# Expert annotations
import xml.etree.ElementTree as ET
tree = ET.parse(XML_PATH)
root = tree.getroot()
anno_features = {}
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

def evaluate(dist_common):
    flat_d = upper_tri(dist_common)
    r_p, _ = pearsonr(flat_d, expert_flat)
    r_s, _ = spearmanr(flat_d, expert_flat)
    nn_agree = 0
    for i in range(nc):
        dm = dist_common[i].copy(); dm[i] = np.inf
        da = expert_dist[i].copy(); da[i] = np.inf
        if np.argmin(dm) == np.argmin(da):
            nn_agree += 1
    cond_m = squareform(dist_common, checks=False)
    cond_e = squareform(expert_dist, checks=False)
    Z_m = linkage(cond_m, method='ward')
    Z_e = linkage(cond_e, method='ward')
    cm = cophenet(Z_m)
    ce = cophenet(Z_e)
    r_c, _ = pearsonr(cm, ce)
    return r_p, r_s, nn_agree, r_c

print(f"  {n} texts, {nc} with expert annotations")


# ══════════════════════════════════════════════════════════════
# STAGE 1: STYLOMETRY → The Forest
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 1: Stylometry — establishing the overall tree")
print("=" * 70)

all_tokens_flat = []
for nm in text_names:
    all_tokens_flat.extend(text_tokens[nm])
vocab_counts = Counter(all_tokens_flat)
MFW = 300
mfw_list = [w for w, _ in vocab_counts.most_common(MFW)]

features_matrix = np.array([
    np.array([Counter(text_tokens[nm]).get(w, 0) / max(len(text_tokens[nm]), 1)
              for w in mfw_list])
    for nm in text_names
])
fm_means = features_matrix.mean(axis=0)
fm_stds = features_matrix.std(axis=0, ddof=0)
fm_stds[fm_stds == 0] = 1
z = (features_matrix - fm_means) / fm_stds

dist_stylo = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = np.sqrt(np.mean((z[i] - z[j])**2))
        dist_stylo[i, j] = dist_stylo[j, i] = d

dist_stylo_c = dist_stylo[np.ix_(cidx, cidx)]

# Cluster into groups via Ward linkage
cond_stylo = squareform(dist_stylo, checks=False)
Z_stylo = linkage(cond_stylo, method='ward')
stylo_clusters = fcluster(Z_stylo, t=3, criterion='maxclust')
stylo_group_map = {}
for i, nm in enumerate(text_names):
    stylo_group_map[nm] = int(stylo_clusters[i])

rp_s, rs_s, nn_s, rc_s = evaluate(dist_stylo_c)
print(f"  Quadratic Delta (300 MFW): r={rp_s:.3f}, ρ={rs_s:.3f}, NN={nn_s}/{nc}")
print(f"  Stylometric clusters (k=3): {stylo_group_map}")

# Identify nearest neighbors from stylometry
stylo_nn = {}
for i, nm in enumerate(text_names):
    dists = dist_stylo[i].copy()
    dists[i] = np.inf
    nn_idx = np.argmin(dists)
    stylo_nn[nm] = text_names[nn_idx]
    print(f"  {nm} → nearest: {text_names[nn_idx]} (d={dists[nn_idx]:.3f})")


# ══════════════════════════════════════════════════════════════
# STAGE 2: 4-GRAM OVERLAP → Refine within groups
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 2: 4-gram overlap — refining relationships")
print("=" * 70)

raw_ngrams = {}
for nm in text_names:
    toks = text_tokens[nm]
    raw_ngrams[nm] = set(tuple(toks[i:i+4]) for i in range(len(toks)-3))

dist_4gram = np.zeros((n, n))
sim_4gram = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si = raw_ngrams[text_names[i]]
        sj = raw_ngrams[text_names[j]]
        u = len(si | sj)
        jac = len(si & sj) / u if u > 0 else 0
        sim_4gram[i, j] = sim_4gram[j, i] = jac
        dist_4gram[i, j] = dist_4gram[j, i] = 1 - jac

dist_4gram_c = dist_4gram[np.ix_(cidx, cidx)]
rp_4, rs_4, nn_4, rc_4 = evaluate(dist_4gram_c)
print(f"  4-gram Jaccard: r={rp_4:.3f}, ρ={rs_4:.3f}, NN={nn_4}/{nc}")


# ══════════════════════════════════════════════════════════════
# STAGE 3: COMBINED TREE → Best quantitative clustering
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 3: Combined tree — optimal blend for clustering")
print("=" * 70)

# Normalize
def normalize(d):
    flat = upper_tri(d)
    mn, mx = flat.min(), flat.max()
    if mx - mn < 1e-10:
        return d.copy()
    return (d - mn) / (mx - mn)

d_s_n = normalize(dist_stylo_c)
d_4_n = normalize(dist_4gram_c)

# Grid search for best blend
best_rho = -1
best_w = 0
best_nn = 0
best_nn_w = 0
for w in np.linspace(0, 1, 101):
    d = w * d_s_n + (1 - w) * d_4_n
    flat = upper_tri(d)
    rho = spearmanr(flat, expert_flat)[0]
    if rho > best_rho:
        best_rho = rho
        best_w = w
    # NN
    nn_count = 0
    for i in range(nc):
        dm = d[i].copy(); dm[i] = np.inf
        da = expert_dist[i].copy(); da[i] = np.inf
        if np.argmin(dm) == np.argmin(da):
            nn_count += 1
    if nn_count > best_nn or (nn_count == best_nn and rho > spearmanr(upper_tri(best_nn_w * d_s_n + (1 - best_nn_w) * d_4_n), expert_flat)[0] if best_nn > 0 else -1):
        best_nn = nn_count
        best_nn_w = w

dist_combined = best_w * d_s_n + (1 - best_w) * d_4_n
rp_comb, rs_comb, nn_comb, rc_comb = evaluate(dist_combined)
print(f"  Best ρ blend: {best_w:.0%} stylo + {1-best_w:.0%} 4gram")
print(f"    r={rp_comb:.3f}, ρ={rs_comb:.3f}, NN={nn_comb}/{nc}, coph={rc_comb:.3f}")

dist_nn_opt = best_nn_w * d_s_n + (1 - best_nn_w) * d_4_n
rp_nn, rs_nn, nn_nn_count, rc_nn = evaluate(dist_nn_opt)
print(f"  Best NN blend: {best_nn_w:.0%} stylo + {1-best_nn_w:.0%} 4gram")
print(f"    r={rp_nn:.3f}, ρ={rs_nn:.3f}, NN={nn_nn_count}/{nc}, coph={rc_nn:.3f}")

# Build the final tree for exploration
dist_tree = dist_nn_opt  # use NN-optimized for tree
cond_tree = squareform(dist_tree, checks=False)
Z_tree = linkage(cond_tree, method='ward')

# Identify close pairs from the combined tree (below median distance)
flat_tree = upper_tri(dist_tree)
close_threshold = np.percentile(flat_tree, 25)  # closest 25% of pairs
close_pairs = []
pair_list = list(zip(*np.triu_indices(nc, k=1)))
for pi, (i, j) in enumerate(pair_list):
    if flat_tree[pi] <= close_threshold:
        close_pairs.append((common[i], common[j], flat_tree[pi]))

close_pairs.sort(key=lambda x: x[2])
print(f"\n  Close pairs (bottom 25%, threshold={close_threshold:.3f}):")
for na, nb, d in close_pairs:
    print(f"    {na}↔{nb} ({GROUP_MAP[na]}/{GROUP_MAP[nb]}): combined d={d:.3f}")


# ══════════════════════════════════════════════════════════════
# STAGE 4: TEXT-MATCHER → For close pairs, WHAT is shared?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 4: text-matcher — examining close pairs in detail")
print("=" * 70)

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
        self.textA = textA
        self.textB = textB
        self.ngram_size = ngram_size
        self.textAgrams = textA.ngrams(ngram_size)
        self.textBgrams = textB.ngrams(ngram_size)
        seq = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        blocks = seq.get_matching_blocks()
        self.initial_matches = [m for m in blocks if m.size > threshold]
        self.healed_matches = self._heal(self.initial_matches)
        self.extended_matches = self._extend(self.healed_matches)
        self.extended_matches = [m for m in self.extended_matches
                                  if min(m.sizeA, m.sizeB) >= cutoff]
        self.numMatches = len(self.extended_matches)

    def _heal(self, matches, min_dist=8):
        healed = []
        if len(matches) <= 1:
            for m in matches:
                healed.append(ExtendedMatch(m.a, m.b, m.size, m.size))
            return healed
        skip = False
        for i in range(len(matches)):
            if skip:
                skip = False
                continue
            m = matches[i]
            if i + 1 < len(matches):
                nxt = matches[i + 1]
                if (nxt.a - (m.a + m.size)) < min_dist:
                    em = ExtendedMatch(m.a, m.b,
                                       (nxt.a + nxt.size) - m.a,
                                       (nxt.b + nxt.size) - m.b)
                    em.healed = True
                    healed.append(em)
                    skip = True
                else:
                    healed.append(ExtendedMatch(m.a, m.b, m.size, m.size))
            else:
                healed.append(ExtendedMatch(m.a, m.b, m.size, m.size))
        return healed

    def _extend(self, matches, cutoff=0.4):
        extended = False
        for m in matches:
            if m.a > 0 and m.b > 0:
                wA = self.textAgrams[m.a - 1][0]
                wB = self.textBgrams[m.b - 1][0]
                d = editDistance(wA, wB)
                avg = (len(wA) + len(wB)) / 2
                if avg > 0 and d / avg < cutoff:
                    m.a -= 1; m.b -= 1; m.sizeA += 1; m.sizeB += 1
                    m.extendedBackwards += 1; extended = True
            idxA = m.a + m.sizeA + 1
            idxB = m.b + m.sizeB + 1
            if idxA < len(self.textAgrams) and idxB < len(self.textBgrams):
                wA = self.textAgrams[idxA][-1]
                wB = self.textBgrams[idxB][-1]
                d = editDistance(wA, wB)
                avg = (len(wA) + len(wB)) / 2
                if avg > 0 and d / avg < cutoff:
                    m.sizeA += 1; m.sizeB += 1
                    m.extendedForwards += 1; extended = True
        if extended:
            self._extend(matches)
        return matches

    def get_passage(self, text, start, length):
        end = min(start + length, len(text.spans))
        if start >= len(text.spans) or end <= start:
            return ""
        return text.text[text.spans[start][0]:text.spans[end-1][1]]

    def get_details(self):
        details = []
        for m in self.extended_matches:
            lenA = m.sizeA + self.ngram_size - 1
            lenB = m.sizeB + self.ngram_size - 1
            details.append({
                'passageA': self.get_passage(self.textA, m.a, lenA),
                'passageB': self.get_passage(self.textB, m.b, lenB),
                'lenA': lenA, 'lenB': lenB,
                'posA': (m.a + lenA/2) / max(len(self.textA.tokens), 1),
                'posB': (m.b + lenB/2) / max(len(self.textB.tokens), 1),
                'startA': m.a, 'startB': m.b,
                'healed': m.healed,
            })
        return details


# Build text objects
gt = {nm: GermanText(plain_texts[nm], nm) for nm in text_names}

# Run text-matcher on ALL pairs (we need it for the full picture),
# but the detailed reports focus on close pairs from Stage 3
print("Running text-matcher on all pairs...")
all_pair_details = {}
copied_mask = {nm: np.zeros(len(gt[nm].tokens), dtype=bool) for nm in text_names}
tm_score_full = np.zeros((n, n))

pair_count = 0
total_pairs = n * (n - 1) // 2
for i in range(n):
    for j in range(i+1, n):
        pair_count += 1
        na, nb = text_names[i], text_names[j]
        matcher = GermanMatcher(gt[na], gt[nb])
        details = matcher.get_details()

        total_matched = 0
        for d in details:
            total_matched += d['lenA']
            copied_mask[na][d['startA']:d['startA']+d['lenA']] = True
            copied_mask[nb][d['startB']:d['startB']+d['lenB']] = True

        norm = min(len(gt[na].tokens), len(gt[nb].tokens))
        tm_score_full[i, j] = tm_score_full[j, i] = total_matched / norm if norm > 0 else 0

        all_pair_details[(na, nb)] = {
            'n_matches': matcher.numMatches,
            'total_words': total_matched,
            'details': details,
        }

        if pair_count % 20 == 0:
            print(f"  {pair_count}/{total_pairs}...")

print(f"  Done. {total_pairs} pairs analyzed.")


# ══════════════════════════════════════════════════════════════
# STAGE 5: Generate exploration reports
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 5: Generating exploration reports")
print("=" * 70)

# ── 5a: Per-pair detailed HTML reports ──

def html_escape(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def generate_pair_report(na, nb, details_dict, tree_dist, stylo_dist, gram4_dist):
    """Generate an HTML report for a specific text pair."""
    key = (na, nb) if (na, nb) in details_dict else (nb, na)
    if key not in details_dict:
        return None

    info = details_dict[key]
    matches = info['details']
    matches_sorted = sorted(matches, key=lambda m: -m['lenA'])

    gi, gj = GROUP_MAP.get(na, '?'), GROUP_MAP.get(nb, '?')
    n_tok_a = len(gt[na].tokens)
    n_tok_b = len(gt[nb].tokens)

    # Compute per-pair copied masks
    mask_a = np.zeros(n_tok_a, dtype=bool)
    mask_b = np.zeros(n_tok_b, dtype=bool)
    for m in matches:
        mask_a[m['startA']:m['startA']+m['lenA']] = True
        mask_b[m['startB']:m['startB']+m['lenB']] = True

    pct_a = 100 * mask_a.sum() / n_tok_a if n_tok_a > 0 else 0
    pct_b = 100 * mask_b.sum() / n_tok_b if n_tok_b > 0 else 0

    # Build annotated full text (mark shared passages)
    def annotate_text(text_obj, pair_matches, role='A'):
        spans = []
        for m in pair_matches:
            if role == 'A':
                start_tok = m['startA']
                length = m['lenA']
            else:
                start_tok = m['startB']
                length = m['lenB']
            end_tok = min(start_tok + length, len(text_obj.spans))
            if start_tok < len(text_obj.spans) and end_tok > start_tok:
                char_start = text_obj.spans[start_tok][0]
                char_end = text_obj.spans[end_tok - 1][1]
                spans.append((char_start, char_end))

        # Sort and merge overlapping spans
        spans.sort()
        merged = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Build HTML with highlights
        raw = text_obj.text
        parts = []
        prev = 0
        for s, e in merged:
            parts.append(html_escape(raw[prev:s]))
            parts.append(f'<mark class="shared">{html_escape(raw[s:e])}</mark>')
            prev = e
        parts.append(html_escape(raw[prev:]))
        return ''.join(parts)

    text_a_html = annotate_text(gt[na], matches, 'A')
    text_b_html = annotate_text(gt[nb], matches, 'B')

    # Build passage comparison table
    passage_rows = []
    for idx, m in enumerate(matches_sorted):
        pos_a = f"{m['posA']:.0%}"
        pos_b = f"{m['posB']:.0%}"
        passage_rows.append(f"""
        <tr>
            <td>{idx+1}</td>
            <td>{m['lenA']}</td>
            <td>{pos_a}</td>
            <td>{pos_b}</td>
            <td class="passage">{html_escape(m['passageA'][:200])}</td>
            <td class="passage">{html_escape(m['passageB'][:200])}</td>
        </tr>""")

    passage_table = '\n'.join(passage_rows)

    # Who else is each text close to?
    def top_neighbors(nm, k=5):
        idx = name_idx[nm]
        dists = [(text_names[j], dist_stylo[idx, j], sim_4gram[idx, j],
                  tm_score_full[idx, j])
                 for j in range(n) if j != idx]
        dists.sort(key=lambda x: x[1])  # sort by stylometric distance
        rows = []
        for neighbor, sd, s4, tms in dists[:k]:
            rows.append(f"<tr><td>{neighbor} ({GROUP_MAP[neighbor]})</td>"
                       f"<td>{sd:.3f}</td><td>{s4:.4f}</td><td>{tms:.4f}</td></tr>")
        return '\n'.join(rows)

    neighbors_a = top_neighbors(na)
    neighbors_b = top_neighbors(nb)

    ii = name_idx[na]
    jj = name_idx[nb]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{na} ↔ {nb} — Detailed Comparison</title>
<style>
    body {{ font-family: 'Georgia', serif; max-width: 1400px; margin: 0 auto;
           padding: 20px; background: #fafafa; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #34495e; margin-top: 2em; }}
    h3 {{ color: #7f8c8d; }}
    .summary {{ background: white; border: 1px solid #ddd; border-radius: 8px;
                padding: 20px; margin: 20px 0; }}
    .summary table {{ border-collapse: collapse; width: 100%; }}
    .summary td, .summary th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
    .summary th {{ background: #f7f9fc; font-weight: bold; }}
    .gruppe-I {{ color: #e74c3c; font-weight: bold; }}
    .gruppe-II {{ color: #3498db; font-weight: bold; }}
    .gruppe-III {{ color: #2ecc71; font-weight: bold; }}
    .text-panel {{ display: flex; gap: 20px; margin: 20px 0; }}
    .text-box {{ flex: 1; background: white; border: 1px solid #ddd; border-radius: 8px;
                 padding: 20px; max-height: 600px; overflow-y: auto;
                 font-size: 14px; line-height: 1.8; }}
    .text-box h3 {{ position: sticky; top: 0; background: white; padding: 5px 0;
                    border-bottom: 1px solid #eee; margin-top: 0; }}
    mark.shared {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 2px;
                   padding: 0 2px; }}
    .passage-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    .passage-table th {{ background: #34495e; color: white; padding: 10px;
                         text-align: left; font-size: 13px; }}
    .passage-table td {{ padding: 8px 10px; border-bottom: 1px solid #eee;
                         font-size: 13px; vertical-align: top; }}
    .passage-table tr:hover {{ background: #f5f5f5; }}
    .passage {{ font-family: monospace; font-size: 12px; max-width: 350px;
                word-break: break-word; }}
    .neighbor-table {{ width: 100%; border-collapse: collapse; }}
    .neighbor-table th {{ background: #f7f9fc; padding: 8px; text-align: left;
                          border-bottom: 2px solid #ddd; }}
    .neighbor-table td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
    .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .stat-box {{ background: white; border: 1px solid #ddd; border-radius: 8px;
                 padding: 15px 20px; text-align: center; min-width: 150px; }}
    .stat-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
    .stat-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
    .nav {{ background: #34495e; color: white; padding: 10px 20px; border-radius: 8px;
            margin-bottom: 20px; }}
    .nav a {{ color: #3498db; text-decoration: none; }}
    .nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<div class="nav">
    <a href="../EXPLORATION_REPORT.md">← Back to overview</a> |
    Pair report: <strong>{na} ↔ {nb}</strong>
</div>

<h1>{na} <span class="gruppe-{gi}">({gi})</span> ↔
    {nb} <span class="gruppe-{gj}">({gj})</span></h1>

<div class="stats">
    <div class="stat-box">
        <div class="stat-value">{info['n_matches']}</div>
        <div class="stat-label">Shared passages</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{info['total_words']}</div>
        <div class="stat-label">Total shared words</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{pct_a:.0f}% / {pct_b:.0f}%</div>
        <div class="stat-label">% of {na} / {nb} that is shared</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{dist_stylo[ii,jj]:.3f}</div>
        <div class="stat-label">Stylometric distance</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{sim_4gram[ii,jj]:.4f}</div>
        <div class="stat-label">4-gram Jaccard</div>
    </div>
</div>

<h2>Full Texts with Shared Passages Highlighted</h2>
<p>Yellow highlighted regions are passages shared between the two texts
   (found by text-matcher with fuzzy matching). Scroll to compare.</p>

<div class="text-panel">
    <div class="text-box">
        <h3>{na} <span class="gruppe-{gi}">Gruppe {gi}</span>
            — {n_tok_a} words, {pct_a:.0f}% shared</h3>
        {text_a_html}
    </div>
    <div class="text-box">
        <h3>{nb} <span class="gruppe-{gj}">Gruppe {gj}</span>
            — {n_tok_b} words, {pct_b:.0f}% shared</h3>
        {text_b_html}
    </div>
</div>

<h2>Shared Passages (sorted by length)</h2>
<table class="passage-table">
    <thead>
        <tr>
            <th>#</th>
            <th>Words</th>
            <th>Pos in {na}</th>
            <th>Pos in {nb}</th>
            <th>Passage in {na}</th>
            <th>Passage in {nb}</th>
        </tr>
    </thead>
    <tbody>
        {passage_table}
    </tbody>
</table>

<h2>Context: Nearest Neighbors</h2>
<p>Where do {na} and {nb} sit relative to the rest of the corpus?</p>

<div class="text-panel">
    <div style="flex:1">
        <h3>{na}'s closest neighbors (by stylometry)</h3>
        <table class="neighbor-table">
            <tr><th>Text</th><th>Stylo dist</th><th>4-gram sim</th><th>text-matcher</th></tr>
            {neighbors_a}
        </table>
    </div>
    <div style="flex:1">
        <h3>{nb}'s closest neighbors (by stylometry)</h3>
        <table class="neighbor-table">
            <tr><th>Text</th><th>Stylo dist</th><th>4-gram sim</th><th>text-matcher</th></tr>
            {neighbors_b}
        </table>
    </div>
</div>

<h2>How to Read This Report</h2>
<ul>
    <li><strong>Shared passages</strong> (yellow highlights) are regions where text-matcher found
        near-verbatim agreement between the two manuscripts. These represent likely copying
        or derivation from a shared source.</li>
    <li><strong>Non-highlighted text</strong> is the scribe's original composition — material
        unique to this manuscript within the corpus.</li>
    <li><strong>Position</strong> is given as a percentage of the total text (0% = beginning,
        100% = end). Shared passages at corresponding positions (e.g., both at 70%) suggest
        structural preservation during copying.</li>
    <li><strong>Stylometric distance</strong> measures overall writing-style similarity
        (lower = more similar). <strong>4-gram Jaccard</strong> measures phrasal overlap.
        <strong>text-matcher score</strong> measures verbatim copying.</li>
</ul>

<p style="color: #999; font-size: 12px; margin-top: 40px;">
    Generated by exploratory_pipeline.py | Processus Universalis corpus analysis
</p>

</body>
</html>"""

    return html


# Generate reports for close pairs AND a few interesting non-close pairs
report_pairs = set()
for na, nb, d in close_pairs:
    report_pairs.add((na, nb))

# Also add expert NN pairs for comparison
for i in range(nc):
    da = expert_dist[i].copy(); da[i] = np.inf
    nn_idx = np.argmin(da)
    pair = tuple(sorted([common[i], common[nn_idx]]))
    report_pairs.add(pair)

# Add the most text-matcher-heavy pairs
tm_pairs = []
for i in range(n):
    for j in range(i+1, n):
        tm_pairs.append((text_names[i], text_names[j], tm_score_full[i, j]))
tm_pairs.sort(key=lambda x: -x[2])
for na, nb, _ in tm_pairs[:10]:
    report_pairs.add(tuple(sorted([na, nb])))

print(f"  Generating {len(report_pairs)} pair reports...")
report_index = []
for na, nb in sorted(report_pairs):
    ii, jj = name_idx[na], name_idx[nb]
    ci = common.index(na) if na in common else None
    cj = common.index(nb) if nb in common else None
    tree_d = dist_tree[ci, cj] if ci is not None and cj is not None else None

    html = generate_pair_report(na, nb, all_pair_details,
                                tree_d, dist_stylo[ii, jj], dist_4gram[ii, jj])
    if html:
        fname = f"{na}_{nb}.html"
        (REPORT_DIR / fname).write_text(html, encoding='utf-8')
        key = (na, nb) if (na, nb) in all_pair_details else (nb, na)
        info = all_pair_details.get(key, {'n_matches': 0, 'total_words': 0})
        report_index.append({
            'file': fname,
            'textA': na, 'textB': nb,
            'gruppeA': GROUP_MAP[na], 'gruppeB': GROUP_MAP[nb],
            'n_matches': info['n_matches'],
            'total_words': info['total_words'],
            'stylo_dist': float(dist_stylo[ii, jj]),
            'gram4_sim': float(sim_4gram[ii, jj]),
            'tm_score': float(tm_score_full[ii, jj]),
        })

print(f"  {len(report_index)} reports written to {REPORT_DIR}/")


# ── 5b: Summary table (JSON for programmatic use) ──
summary_data = {
    'texts': {nm: {
        'gruppe': GROUP_MAP[nm],
        'n_tokens': len(text_tokens[nm]),
        'pct_copied': float(100 * copied_mask[nm].sum() / len(copied_mask[nm])),
        'stylo_nn': stylo_nn[nm],
    } for nm in text_names},
    'pair_reports': report_index,
    'tree_evaluation': {
        'stylo_only': {'r': rp_s, 'rho': rs_s, 'nn': nn_s},
        '4gram_only': {'r': rp_4, 'rho': rs_4, 'nn': nn_4},
        'combined_rho': {'r': rp_comb, 'rho': rs_comb, 'nn': nn_comb,
                          'weight_stylo': best_w, 'weight_4gram': 1-best_w},
        'combined_nn': {'r': rp_nn, 'rho': rs_nn, 'nn': nn_nn_count,
                         'weight_stylo': best_nn_w, 'weight_4gram': 1-best_nn_w},
    },
}

(REPORT_DIR / 'summary.json').write_text(
    json.dumps(summary_data, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"  Summary JSON written to {REPORT_DIR}/summary.json")


# ══════════════════════════════════════════════════════════════
# FIGURE JJJ: The pipeline overview
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure JJJ: Pipeline overview...")

fig, axes = plt.subplots(1, 4, figsize=(28, 8))

# Panel 1: Stage 1 — Stylometric dendrogram
ax = axes[0]
cond = squareform(dist_stylo_c, checks=False)
Z = linkage(cond, method='ward')
dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=9)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax.set_title(f'Stage 1: Stylometry\n(Quad. Delta 300 MFW)\nρ={rs_s:.3f}, NN={nn_s}/{nc}',
             fontsize=11, fontweight='bold')
ax.set_ylabel('Ward distance')

# Panel 2: Stage 2 — 4-gram dendrogram
ax = axes[1]
cond = squareform(dist_4gram_c, checks=False)
Z = linkage(cond, method='ward')
dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=9)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax.set_title(f'Stage 2: 4-gram Overlap\n(phrasal similarity)\nρ={rs_4:.3f}, NN={nn_4}/{nc}',
             fontsize=11, fontweight='bold')

# Panel 3: Stage 3 — Combined tree
ax = axes[2]
dn = dendrogram(Z_tree, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=9)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax.set_title(f'Stage 3: Combined Tree\n({best_nn_w:.0%} stylo + {1-best_nn_w:.0%} 4gram)\n'
             f'ρ={rs_nn:.3f}, NN={nn_nn_count}/{nc}',
             fontsize=11, fontweight='bold')

# Panel 4: Expert reference
ax = axes[3]
cond = squareform(expert_dist, checks=False)
Z = linkage(cond, method='ward')
dn = dendrogram(Z, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=9)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax.set_title('Expert Annotations\n(reference)', fontsize=11, fontweight='bold')

legend_handles = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=10)
fig.suptitle("Exploratory Pipeline: Big Picture → Detail",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.93])
plt.savefig(OUT_DIR / 'processus_figJJJ_pipeline_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig JJJ saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE KKK: Where text-matcher adds to the tree
# ══════════════════════════════════════════════════════════════
print("Generating Figure KKK: Text-matcher detail layer...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Left: Dendrogram with close pairs annotated
ax = ax1
dn = dendrogram(Z_tree, labels=common, ax=ax, leaf_rotation=90, leaf_font_size=9)
for lbl in ax.get_xticklabels():
    nm = lbl.get_text()
    if nm in GROUP_MAP:
        lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])

# Get leaf order from dendrogram
leaf_order = dn['ivl']

ax.set_title(f'Combined Tree (Stage 3)\nwith text-matcher annotations',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Ward distance')

# Right: Table of close pairs with their text-matcher findings
ax2.axis('off')
table_data = []
headers = ['Pair', 'Gr.', 'Tree\ndist', 'Shared\npassages', 'Shared\nwords', 'Longest\nmatch']

for na, nb, d in close_pairs[:15]:
    key = (na, nb) if (na, nb) in all_pair_details else (nb, na)
    info = all_pair_details.get(key, {'n_matches': 0, 'total_words': 0, 'details': []})
    longest = max((m['lenA'] for m in info['details']), default=0)
    table_data.append([
        f'{na}↔{nb}',
        f'{GROUP_MAP[na]}/{GROUP_MAP[nb]}',
        f'{d:.3f}',
        str(info['n_matches']),
        str(info['total_words']),
        str(longest),
    ])

if table_data:
    table = ax2.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color rows by same/different gruppe
    for i, (na, nb, d) in enumerate(close_pairs[:15]):
        color = '#e8f8e8' if GROUP_MAP[na] == GROUP_MAP[nb] else '#f8e8e8'
        for j in range(len(headers)):
            table[i+1, j].set_facecolor(color)

ax2.set_title('Close Pairs: What text-matcher reveals\n'
              '(green = same Gruppe, red = cross-Gruppe)',
              fontsize=12, fontweight='bold')

fig.suptitle("Stage 4: text-matcher Adds Detail to the Tree",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figKKK_tm_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig KKK saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE LLL: Copying maps for the top pairs
# ══════════════════════════════════════════════════════════════
print("Generating Figure LLL: Copying maps for top pairs...")

# Show the top 6 most-matched pairs
top_tm_pairs = sorted(all_pair_details.items(),
                       key=lambda x: -x[1]['total_words'])[:6]

fig, axes = plt.subplots(len(top_tm_pairs), 1, figsize=(16, len(top_tm_pairs) * 2.5 + 1))
if len(top_tm_pairs) == 1:
    axes = [axes]

for ax_idx, ((na, nb), info) in enumerate(top_tm_pairs):
    ax = axes[ax_idx]
    matches = info['details']

    n_tok_a = len(gt[na].tokens)
    n_tok_b = len(gt[nb].tokens)

    # Draw text A (top) and text B (bottom) as bars
    # with shared passages connected
    for m in matches:
        posA_start = m['startA'] / n_tok_a
        posA_end = (m['startA'] + m['lenA']) / n_tok_a
        posB_start = m['startB'] / n_tok_b
        posB_end = (m['startB'] + m['lenB']) / n_tok_b

        # Draw match regions
        ax.fill_between([posA_start, posA_end], 0.6, 1.0,
                        color='#ff6b6b', alpha=0.5, linewidth=0)
        ax.fill_between([posB_start, posB_end], 0.0, 0.4,
                        color='#ff6b6b', alpha=0.5, linewidth=0)

        # Connect them
        ax.plot([posA_start, posB_start], [0.6, 0.4],
                color='#ff6b6b', alpha=0.15, linewidth=0.5)
        ax.plot([posA_end, posB_end], [0.6, 0.4],
                color='#ff6b6b', alpha=0.15, linewidth=0.5)

    # Draw base bars
    ax.fill_between([0, 1], 0.6, 1.0, color='#3498db', alpha=0.15, linewidth=0)
    ax.fill_between([0, 1], 0.0, 0.4, color='#3498db', alpha=0.15, linewidth=0)

    # Labels
    ax.text(-0.02, 0.8, f'{na} ({GROUP_MAP[na]})', ha='right', va='center',
            fontsize=10, fontweight='bold', color=GROUP_COLORS[GROUP_MAP[na]])
    ax.text(-0.02, 0.2, f'{nb} ({GROUP_MAP[nb]})', ha='right', va='center',
            fontsize=10, fontweight='bold', color=GROUP_COLORS[GROUP_MAP[nb]])

    ax.text(1.02, 0.8, f'{info["n_matches"]} passages', ha='left', va='center', fontsize=9)
    ax.text(1.02, 0.2, f'{info["total_words"]} words', ha='left', va='center', fontsize=9)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    if ax_idx == len(top_tm_pairs) - 1:
        ax.set_xlabel('Position in text (0% = beginning, 100% = end)', fontsize=11)
    else:
        ax.set_xticks([])

fig.suptitle("Passage-Level Alignment: Where Do Texts Share Material?",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0.08, 0, 0.92, 0.95])
plt.savefig(OUT_DIR / 'processus_figLLL_passage_alignment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig LLL saved.")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPLORATORY PIPELINE SUMMARY")
print("=" * 70)

print(f"\nStage 1 (Stylometry):  ρ={rs_s:.3f}, NN={nn_s}/{nc}")
print(f"Stage 2 (4-gram):      ρ={rs_4:.3f}, NN={nn_4}/{nc}")
print(f"Stage 3 (Combined):    ρ={rs_nn:.3f}, NN={nn_nn_count}/{nc}")
print(f"  Weights: {best_nn_w:.0%} stylo + {1-best_nn_w:.0%} 4gram")

print(f"\nStage 4 (text-matcher): {len(report_pairs)} detailed pair reports generated")
print(f"  Top pairs by shared material:")
for (na, nb), info in top_tm_pairs:
    print(f"    {na}↔{nb}: {info['n_matches']} passages, {info['total_words']} words")

print(f"\nOutputs:")
print(f"  Fig JJJ: processus_figJJJ_pipeline_overview.png")
print(f"  Fig KKK: processus_figKKK_tm_detail.png")
print(f"  Fig LLL: processus_figLLL_passage_alignment.png")
print(f"  {REPORT_DIR}/: {len(report_index)} HTML pair reports")
print(f"  {REPORT_DIR}/summary.json: machine-readable summary")
