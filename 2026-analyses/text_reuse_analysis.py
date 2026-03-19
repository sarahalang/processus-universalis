#!/usr/bin/env python3
"""
Text Reuse Analysis: Longest Common Substring Matching
======================================================
Uses Jonathan Reeve's text-matcher approach (extended longest common
subsequence matching on n-gram sequences) to find text reuse between
recipe manuscripts.

Compares results to 4-gram overlap and other methods.

Key difference from 4-grams: text-matcher finds *extended* matching
passages of arbitrary length. A 30-word shared passage is far more
meaningful than six overlapping 4-grams, and this method captures that.

Produces Figures XX through BBB.
"""

import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.stats import pearsonr, spearmanr

# Use text-matcher internals but with German-appropriate settings
from text_matcher.matcher import Text, Matcher, ExtendedMatch
from difflib import SequenceMatcher
from nltk.metrics.distance import edit_distance as editDistance

# ── Configuration ──
TXT_DIR = Path("processus/processus_prev_work/processus_universalis-main/"
               "ProcessusUniversalis_relevant-files-for-2025/"
               "txt-files-lowercase_processus")
OUT_DIR = Path("processus-universalis-graphics")
OUT_DIR.mkdir(exist_ok=True)

GROUP_MAP = {
    'E2': 'I', 'E3': 'I', 'E11': 'I', 'E22': 'I', 'E35': 'I',
    'E16': 'II', 'E17': 'II', 'E19': 'II', 'E27': 'II', 'E32b': 'II',
    'E34': 'III', 'E37': 'III', 'E38': 'III', 'E39': 'III',
    'E42': 'III', 'E44': 'III', 'E45': 'III',
}
GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}


def load_texts():
    """Load all texts as raw strings and as word lists."""
    texts = {}
    for fp in sorted(TXT_DIR.glob("*.txt")):
        fname = fp.stem
        m = re.search(r'(E\d+[a-z]?)', fname)
        if not m:
            continue
        ename = m.group(1)
        if ename not in GROUP_MAP:
            continue
        raw = fp.read_text(encoding='utf-8', errors='replace')
        texts[ename] = raw
    return texts


class GermanText:
    """A Text object adapted for Early New High German.
    No stemming (Lancaster is for English), no English stopword removal.
    Simple lowercasing and tokenization only."""

    def __init__(self, raw_text, label):
        self.text = raw_text
        self.label = label
        # Simple tokenization for German/Latin
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
    """Adapted Matcher for German texts.
    Uses text-matcher's core algorithm but without English-specific processing."""

    def __init__(self, textA, textB, threshold=3, cutoff=5, ngram_size=3):
        self.textA = textA
        self.textB = textB
        self.threshold = threshold
        self.cutoff = cutoff
        self.ngram_size = ngram_size

        self.textAgrams = textA.ngrams(ngram_size)
        self.textBgrams = textB.ngrams(ngram_size)

        # Step 1: Find initial matching blocks using SequenceMatcher
        sequence = SequenceMatcher(None, self.textAgrams, self.textBgrams)
        matching_blocks = sequence.get_matching_blocks()

        # Filter by threshold
        self.initial_matches = [m for m in matching_blocks if m.size > threshold]

        # Step 2: Heal neighboring matches
        self.healed_matches = self._heal_neighbors()

        # Step 3: Extend matches with fuzzy edit-distance
        self.extended_matches = self._extend_matches()

        # Step 4: Prune short matches
        self.extended_matches = [m for m in self.extended_matches
                                  if min(m.sizeA, m.sizeB) >= cutoff]

        self.numMatches = len(self.extended_matches)

    def _heal_neighbors(self, min_distance=8):
        """Merge nearby matches into larger ones."""
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
        """Edit distance normalized by average word length."""
        distance = editDistance(wordA, wordB)
        avg_len = (len(wordA) + len(wordB)) / 2
        return distance / avg_len if avg_len > 0 else 1.0

    def _extend_matches(self, edit_cutoff=0.4):
        """Extend matches forwards and backwards using fuzzy matching."""
        for match in self.healed_matches:
            # Extend backwards
            while match.a > 0 and match.b > 0:
                wordA = self.textA.tokens[match.a - 1]
                wordB = self.textB.tokens[match.b - 1]
                if self._edit_ratio(wordA, wordB) < edit_cutoff:
                    match.a -= 1
                    match.b -= 1
                    match.sizeA += 1
                    match.sizeB += 1
                    match.extendedBackwards += 1
                else:
                    break

            # Extend forwards
            endA = match.a + match.sizeA
            endB = match.b + match.sizeB
            while endA < len(self.textA.tokens) and endB < len(self.textB.tokens):
                wordA = self.textA.tokens[endA]
                wordB = self.textB.tokens[endB]
                if self._edit_ratio(wordA, wordB) < edit_cutoff:
                    match.sizeA += 1
                    match.sizeB += 1
                    match.extendedForwards += 1
                    endA += 1
                    endB += 1
                else:
                    break

        return self.healed_matches

    def get_passage(self, text, start, length):
        """Get the original text for a match."""
        end = min(start + length, len(text.spans))
        if start >= len(text.spans) or end <= start:
            return ""
        span_start = text.spans[start][0]
        span_end = text.spans[end - 1][1]
        return text.text[span_start:span_end]

    def get_match_details(self):
        """Return structured details for all matches."""
        details = []
        for match in self.extended_matches:
            lenA = match.sizeA + self.ngram_size - 1
            lenB = match.sizeB + self.ngram_size - 1

            passageA = self.get_passage(self.textA, match.a, lenA)
            passageB = self.get_passage(self.textB, match.b, lenB)

            posA = (match.a + lenA / 2) / len(self.textA.tokens) if self.textA.tokens else 0
            posB = (match.b + lenB / 2) / len(self.textB.tokens) if self.textB.tokens else 0

            details.append({
                'passageA': passageA,
                'passageB': passageB,
                'lenA': lenA,
                'lenB': lenB,
                'posA': posA,
                'posB': posB,
                'startA': match.a,
                'startB': match.b,
                'healed': match.healed,
                'extended_back': match.extendedBackwards,
                'extended_fwd': match.extendedForwards,
            })
        return details


# ══════════════════════════════════════════════════════════════
print("Loading texts...")
raw_texts = load_texts()
text_names = sorted(raw_texts.keys(), key=lambda x: (GROUP_MAP[x], x))
print(f"Loaded {len(text_names)} texts\n")

# Create GermanText objects
text_objs = {}
for nm in text_names:
    text_objs[nm] = GermanText(raw_texts[nm], nm)
    print(f"  {nm}: {len(text_objs[nm].tokens)} tokens")

# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Running text-matcher on all pairs...")
print("=" * 70)

# Run matching on all pairs
n = len(text_names)
match_results = {}
total_pairs = n * (n - 1) // 2
pair_count = 0

# Distance matrices
tm_match_count = np.zeros((n, n))   # number of matching passages
tm_match_words = np.zeros((n, n))   # total words in matching passages
tm_match_score = np.zeros((n, n))   # similarity score (0-1)
ngram4_overlap = np.zeros((n, n))   # 4-gram Jaccard for comparison

# Collect all match details for analysis
all_matches = []

for i in range(n):
    for j in range(i + 1, n):
        nmA, nmB = text_names[i], text_names[j]
        pair_count += 1

        # text-matcher matching
        matcher = GermanMatcher(text_objs[nmA], text_objs[nmB],
                                threshold=2, cutoff=5, ngram_size=3)

        total_matched_words = sum(d['lenA'] + d['lenB']
                                  for d in matcher.get_match_details()) / 2
        total_words = len(text_objs[nmA].tokens) + len(text_objs[nmB].tokens)

        tm_match_count[i, j] = tm_match_count[j, i] = matcher.numMatches
        tm_match_words[i, j] = tm_match_words[j, i] = total_matched_words
        tm_match_score[i, j] = tm_match_score[j, i] = total_matched_words / total_words if total_words > 0 else 0

        details = matcher.get_match_details()
        for d in details:
            d['textA'] = nmA
            d['textB'] = nmB
        all_matches.extend(details)

        match_results[(nmA, nmB)] = {
            'n_matches': matcher.numMatches,
            'matched_words': total_matched_words,
            'score': total_matched_words / total_words if total_words > 0 else 0,
            'details': details,
        }

        # 4-gram overlap for comparison
        grams_a = set(text_objs[nmA].ngrams(4))
        grams_b = set(text_objs[nmB].ngrams(4))
        if len(grams_a | grams_b) > 0:
            ngram4_overlap[i, j] = ngram4_overlap[j, i] = len(grams_a & grams_b) / len(grams_a | grams_b)

        if pair_count % 20 == 0 or pair_count == total_pairs:
            print(f"  {pair_count}/{total_pairs} pairs done...")

# Convert to distance matrices (1 - similarity)
tm_dist = 1 - tm_match_score / (tm_match_score.max() + 1e-10)  # normalize
ngram4_dist = 1 - ngram4_overlap / (ngram4_overlap.max() + 1e-10)

print(f"\nTotal matches found: {len(all_matches)}")
print(f"Mean match length: {np.mean([m['lenA'] for m in all_matches]):.1f} words")
print(f"Median match length: {np.median([m['lenA'] for m in all_matches]):.1f} words")
print(f"Max match length: {max(m['lenA'] for m in all_matches)} words")
print(f"Min match length: {min(m['lenA'] for m in all_matches)} words")

# Length distribution
lengths = [m['lenA'] for m in all_matches]
print(f"\nMatch length distribution:")
for threshold in [5, 10, 15, 20, 30, 50, 100]:
    count = sum(1 for l in lengths if l >= threshold)
    print(f"  >= {threshold:3d} words: {count:4d} matches ({100*count/len(lengths):.1f}%)")


# ══════════════════════════════════════════════════════════════
# Load expert annotations for comparison
# ══════════════════════════════════════════════════════════════
print("\nLoading expert annotations for comparison...")
import xml.etree.ElementTree as ET
xml_path = Path("processus/processus_prev_work/processus_universalis-main/"
                "ProcessusUniversalis_relevant-files-for-2025/sammlung_aller_texte.xml")
tree = ET.parse(xml_path)
root = tree.getroot()

A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11',
}

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

# Expert distance matrix
common = [nm for nm in text_names if nm in anno_features]
common_idx = [text_names.index(nm) for nm in common]
expert_dist = np.zeros((len(common), len(common)))
for i in range(len(common)):
    for j in range(len(common)):
        a = anno_features[common[i]]
        b = anno_features[common[j]]
        if len(a | b) > 0:
            expert_dist[i, j] = 1 - len(a & b) / len(a | b)

def upper_tri(mat):
    return mat[np.triu_indices(len(mat), k=1)]

expert_flat = upper_tri(expert_dist)

# Align text-matcher and 4-gram distances to common texts
tm_common = tm_dist[np.ix_(common_idx, common_idx)]
ng_common = ngram4_dist[np.ix_(common_idx, common_idx)]
tm_flat = upper_tri(tm_common)
ng_flat = upper_tri(ng_common)

r_tm, _ = pearsonr(tm_flat, expert_flat)
rho_tm, _ = spearmanr(tm_flat, expert_flat)
r_ng, _ = pearsonr(ng_flat, expert_flat)
rho_ng, _ = spearmanr(ng_flat, expert_flat)

print(f"\nCorrelations with expert annotations:")
print(f"  text-matcher (longest common substring): r={r_tm:.3f}, rho={rho_tm:.3f}")
print(f"  4-gram Jaccard overlap:                  r={r_ng:.3f}, rho={rho_ng:.3f}")


# ══════════════════════════════════════════════════════════════
# Print the most significant matches (longest passages)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TOP 20 LONGEST SHARED PASSAGES (text-matcher)")
print("=" * 70)

sorted_matches = sorted(all_matches, key=lambda m: m['lenA'], reverse=True)
for i, m in enumerate(sorted_matches[:20]):
    print(f"\n  #{i+1}: {m['textA']}↔{m['textB']}, {m['lenA']} words "
          f"(pos {m['posA']:.0%}↔{m['posB']:.0%})"
          f"{'  [healed]' if m['healed'] else ''}"
          f"{'  [extended]' if m['extended_back'] or m['extended_fwd'] else ''}")
    passage = m['passageA'][:200]
    print(f"    \"{passage}{'...' if len(m['passageA']) > 200 else ''}\"")


# ══════════════════════════════════════════════════════════════
# FIGURE XX: Match length distribution + comparison
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure XX: Match length distribution...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Panel 1: Histogram of match lengths
ax1 = axes[0]
ax1.hist(lengths, bins=np.arange(5, max(lengths) + 5, 3), color='#3498db',
         alpha=0.7, edgecolor='white')
ax1.axvline(np.median(lengths), color='#e74c3c', ls='--', lw=2,
            label=f'Median = {np.median(lengths):.0f} words')
ax1.axvline(np.mean(lengths), color='#e67e22', ls=':', lw=2,
            label=f'Mean = {np.mean(lengths):.1f} words')
ax1.set_xlabel('Match length (words)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Distribution of Shared Passage Lengths\n(text-matcher)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)

# Panel 2: Scatter — text-matcher score vs 4-gram overlap
ax2 = axes[1]
for i in range(n):
    for j in range(i + 1, n):
        g_same = GROUP_MAP[text_names[i]] == GROUP_MAP[text_names[j]]
        c = '#2ecc71' if g_same else '#95a5a6'
        ax2.scatter(ngram4_overlap[i, j], tm_match_score[i, j],
                    c=c, alpha=0.5, s=30, edgecolors='white', linewidths=0.3)

# Add trend line
valid = [(ngram4_overlap[i, j], tm_match_score[i, j])
         for i in range(n) for j in range(i+1, n)]
x_vals = [v[0] for v in valid]
y_vals = [v[1] for v in valid]
r_methods, _ = pearsonr(x_vals, y_vals)
z = np.polyfit(x_vals, y_vals, 1)
p = np.poly1d(z)
x_range = np.linspace(min(x_vals), max(x_vals), 100)
ax2.plot(x_range, p(x_range), 'k--', lw=1.5, alpha=0.5)

ax2.set_xlabel('4-gram Jaccard overlap', fontsize=11)
ax2.set_ylabel('text-matcher score', fontsize=11)
ax2.set_title(f'text-matcher vs 4-gram Overlap\n(r = {r_methods:.3f})', fontsize=12, fontweight='bold')
legend_handles = [
    Patch(facecolor='#2ecc71', alpha=0.5, label='Same Gruppe'),
    Patch(facecolor='#95a5a6', alpha=0.5, label='Different Gruppe'),
]
ax2.legend(handles=legend_handles, fontsize=9)

# Panel 3: Where in the text do matches occur?
ax3 = axes[2]
for m in all_matches:
    length_cat = 'short' if m['lenA'] < 10 else ('medium' if m['lenA'] < 25 else 'long')
    color = {'short': '#95a5a6', 'medium': '#3498db', 'long': '#e74c3c'}[length_cat]
    alpha = {'short': 0.1, 'medium': 0.3, 'long': 0.7}[length_cat]
    size = {'short': 5, 'medium': 15, 'long': 40}[length_cat]
    ax3.scatter(m['posA'], m['posB'], c=color, alpha=alpha, s=size,
                edgecolors='none')

ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
ax3.set_xlabel('Position in Text A', fontsize=11)
ax3.set_ylabel('Position in Text B', fontsize=11)
ax3.set_title('Where Do Shared Passages Occur?\n(position in text A vs text B)', fontsize=12, fontweight='bold')
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=5, label='Short (5-9 words)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Medium (10-24 words)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=11, label='Long (25+ words)'),
]
ax3.legend(handles=legend_handles, fontsize=9, loc='upper left')

fig.suptitle("Text Reuse: Longest Common Substring Analysis",
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figXX_match_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig XX saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE YY: Dendrograms — text-matcher vs 4-gram vs expert
# ══════════════════════════════════════════════════════════════
print("Generating Figure YY: Dendrograms...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))

# Cophenetic correlations
for ax, dist_mat, title, names in [
    (ax1, tm_common, 'text-matcher\n(longest common substring)', common),
    (ax2, ng_common, '4-gram Jaccard', common),
    (ax3, expert_dist, 'Expert Annotations\n(reference)', common),
]:
    # Ensure diagonal is 0 and matrix is symmetric
    np.fill_diagonal(dist_mat, 0)
    dist_mat = (dist_mat + dist_mat.T) / 2

    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method='ward')
    dn = dendrogram(Z, labels=names, ax=ax, leaf_rotation=90, leaf_font_size=9)

    # Cophenetic correlation with expert
    if names == common and not np.array_equal(dist_mat, expert_dist):
        expert_condensed = squareform(expert_dist, checks=False)
        coph_dist = cophenet(Z)
        expert_Z = linkage(expert_condensed, method='ward')
        expert_coph = cophenet(expert_Z)
        coph_r, _ = pearsonr(coph_dist, expert_coph)
        r, _ = pearsonr(upper_tri(dist_mat), expert_flat)
        rho, _ = spearmanr(upper_tri(dist_mat), expert_flat)
        title += f'\nr={r:.3f}, rho={rho:.3f}'

    for lbl in ax.get_xticklabels():
        nm = lbl.get_text()
        if nm in GROUP_MAP:
            lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=10)

legend_handles = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=10)
fig.suptitle("Dendrograms: text-matcher vs 4-gram vs Expert",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.92])
plt.savefig(OUT_DIR / 'processus_figYY_tm_dendrograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig YY saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE ZZ: Heatmap of pairwise match counts + match quality
# ══════════════════════════════════════════════════════════════
print("Generating Figure ZZ: Pairwise match heatmaps...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Panel 1: Number of shared passages
im1 = ax1.imshow(tm_match_count, cmap='YlOrRd', interpolation='nearest')
ax1.set_xticks(range(n))
ax1.set_xticklabels(text_names, fontsize=8, rotation=90)
ax1.set_yticks(range(n))
ax1.set_yticklabels(text_names, fontsize=8)
for lbl in ax1.get_xticklabels():
    lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])
for lbl in ax1.get_yticklabels():
    lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])
ax1.set_title('Number of Shared Passages', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Panel 2: Total matched words
im2 = ax2.imshow(tm_match_words, cmap='YlOrRd', interpolation='nearest')
ax2.set_xticks(range(n))
ax2.set_xticklabels(text_names, fontsize=8, rotation=90)
ax2.set_yticks(range(n))
ax2.set_yticklabels(text_names, fontsize=8)
for lbl in ax2.get_xticklabels():
    lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])
for lbl in ax2.get_yticklabels():
    lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])
ax2.set_title('Total Matched Words', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

fig.suptitle("Pairwise Text Reuse: Shared Passage Count and Volume",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figZZ_match_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig ZZ saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE AAA: Where in the recipe is text reused?
#             (by match length and text position)
# ══════════════════════════════════════════════════════════════
print("Generating Figure AAA: Positional analysis of text reuse...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel 1: Density of match positions (all matches)
ax1 = axes[0, 0]
posA_all = [m['posA'] for m in all_matches]
posB_all = [m['posB'] for m in all_matches]
all_pos = posA_all + posB_all  # combine both sides
bins = np.linspace(0, 1, 21)
ax1.hist(all_pos, bins=bins, color='#3498db', alpha=0.7, edgecolor='white')
ax1.set_xlabel('Text position', fontsize=11)
ax1.set_ylabel('Number of matches', fontsize=11)
ax1.set_title('Where in Recipes Does Text Reuse Occur?\n(all matches, both sides combined)',
              fontsize=12, fontweight='bold')
ax1.axvspan(0.75, 1.0, alpha=0.08, color='grey')

# Panel 2: Match length by position
ax2 = axes[0, 1]
pos_mid = [(m['posA'] + m['posB']) / 2 for m in all_matches]
match_len = [m['lenA'] for m in all_matches]
ax2.scatter(pos_mid, match_len, alpha=0.4, s=20, c='#3498db', edgecolors='none')
# Trend
if len(pos_mid) > 10:
    # Bin and compute medians
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_medians = []
    for k in range(len(bin_edges) - 1):
        in_bin = [match_len[i] for i in range(len(pos_mid))
                  if bin_edges[k] <= pos_mid[i] < bin_edges[k+1]]
        bin_medians.append(np.median(in_bin) if in_bin else 0)
    ax2.plot(bin_centers, bin_medians, 'r-o', lw=2, markersize=6, label='Median per bin')
ax2.set_xlabel('Mean position of match', fontsize=11)
ax2.set_ylabel('Match length (words)', fontsize=11)
ax2.set_title('Do Later Matches Get Shorter or Longer?', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.axvspan(0.75, 1.0, alpha=0.08, color='grey')

# Panel 3: Long matches only (>=15 words) — where do they appear?
ax3 = axes[1, 0]
long_matches = [m for m in all_matches if m['lenA'] >= 15]
if long_matches:
    for m in long_matches:
        g_same = GROUP_MAP.get(m['textA']) == GROUP_MAP.get(m['textB'])
        c = '#2ecc71' if g_same else '#e74c3c'
        ax3.scatter(m['posA'], m['posB'], c=c, s=m['lenA'] * 2, alpha=0.6,
                    edgecolors='black', linewidths=0.5)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax3.set_xlabel('Position in Text A', fontsize=11)
    ax3.set_ylabel('Position in Text B', fontsize=11)
    ax3.set_title(f'Long Shared Passages Only (>= 15 words)\n({len(long_matches)} matches)',
                  fontsize=12, fontweight='bold')
    legend_handles = [
        Patch(facecolor='#2ecc71', alpha=0.6, label='Same Gruppe'),
        Patch(facecolor='#e74c3c', alpha=0.6, label='Different Gruppe'),
    ]
    ax3.legend(handles=legend_handles, fontsize=9)

# Panel 4: Match density by Gruppe pair type
ax4 = axes[1, 1]
pair_types = {'within I': [], 'within II': [], 'within III': [],
              'I↔II': [], 'I↔III': [], 'II↔III': []}
for i in range(n):
    for j in range(i + 1, n):
        gA, gB = GROUP_MAP[text_names[i]], GROUP_MAP[text_names[j]]
        if gA == gB:
            key = f'within {gA}'
        else:
            pair = tuple(sorted([gA, gB]))
            key = f'{pair[0]}↔{pair[1]}'
        pair_types[key].append(tm_match_score[i, j])

keys = list(pair_types.keys())
means = [np.mean(v) if v else 0 for v in pair_types.values()]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
ax4.bar(range(len(keys)), means, color=colors[:len(keys)], alpha=0.7)
ax4.set_xticks(range(len(keys)))
ax4.set_xticklabels(keys, fontsize=9, rotation=30, ha='right')
ax4.set_ylabel('Mean text-matcher score', fontsize=10)
ax4.set_title('Text Reuse by Gruppe Pairing', fontsize=12, fontweight='bold')

fig.suptitle("Positional Analysis of Text Reuse",
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_DIR / 'processus_figAAA_positional_reuse.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig AAA saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE BBB: Method comparison summary
# ══════════════════════════════════════════════════════════════
print("Generating Figure BBB: Method comparison...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Panel 1: Correlation comparison
ax1 = axes[0]
methods = ['text-matcher\n(long substr)', '4-gram\nJaccard', 'Proxy pipeline\n(1489 chars)',
           'Quadratic\nDelta 300', 'Embedding\n(full text)', 'Embedding\n(early half)']
r_vals = [r_tm, r_ng, 0.844, 0.731, 0.367, 0.621]
rho_vals = [rho_tm, rho_ng, 0.882, 0.763, 0.443, 0.689]
x = np.arange(len(methods))
w = 0.35
ax1.bar(x - w/2, r_vals, w, label='Pearson r', color='#3498db', alpha=0.7)
ax1.bar(x + w/2, rho_vals, w, label='Spearman rho', color='#e74c3c', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=8)
ax1.set_ylabel('Correlation with expert', fontsize=10)
ax1.set_title('All Methods vs Expert Annotations', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_ylim(0, 1)

# Panel 2: text-matcher distance vs expert distance scatter
ax2 = axes[1]
# Color by relationship type
for i in range(len(common)):
    for j in range(i + 1, len(common)):
        g_same = GROUP_MAP[common[i]] == GROUP_MAP[common[j]]
        c = '#2ecc71' if g_same else '#95a5a6'
        ax2.scatter(tm_common[i, j], expert_dist[i, j], c=c, alpha=0.5, s=30)

z = np.polyfit(tm_flat, expert_flat, 1)
p = np.poly1d(z)
x_range = np.linspace(min(tm_flat), max(tm_flat), 100)
ax2.plot(x_range, p(x_range), 'k--', lw=1.5, alpha=0.5)
ax2.set_xlabel('text-matcher distance', fontsize=10)
ax2.set_ylabel('Expert distance', fontsize=10)
ax2.set_title(f'text-matcher vs Expert\nr={r_tm:.3f}, rho={rho_tm:.3f}',
              fontsize=12, fontweight='bold')

# Panel 3: What text-matcher adds — example long passages
ax3 = axes[2]
ax3.axis('off')

# Top 5 longest matches
top5 = sorted_matches[:5]
text_lines = "Top 5 Longest Shared Passages\n" + "─" * 40 + "\n\n"
for i, m in enumerate(top5):
    passage = m['passageA'][:120].replace('\n', ' ')
    text_lines += (f"#{i+1}: {m['textA']}↔{m['textB']} "
                   f"({m['lenA']} words, pos {m['posA']:.0%}↔{m['posB']:.0%})\n")
    text_lines += f'  "{passage}..."\n\n'

ax3.text(0.02, 0.98, text_lines, transform=ax3.transAxes,
         fontsize=9, fontfamily='monospace', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle("Text Reuse Methods: Comparison and Context",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figBBB_method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig BBB saved.")


# ══════════════════════════════════════════════════════════════
# Print detailed summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEXT REUSE ANALYSIS SUMMARY")
print("=" * 70)

# Most connected pairs
print("\nTop 10 most-connected text pairs (by text-matcher score):")
pairs_sorted = sorted(match_results.items(), key=lambda x: x[1]['score'], reverse=True)
for (a, b), res in pairs_sorted[:10]:
    print(f"  {a}↔{b} ({GROUP_MAP[a]}/{GROUP_MAP[b]}): "
          f"{res['n_matches']} passages, {res['matched_words']:.0f} words, "
          f"score={res['score']:.4f}")

# Comparison summary
print(f"\nCorrelation with expert annotations:")
print(f"  text-matcher (long common substr): r={r_tm:.3f}, rho={rho_tm:.3f}")
print(f"  4-gram Jaccard:                    r={r_ng:.3f}, rho={rho_ng:.3f}")
print(f"  Proxy pipeline:                    r=0.844, rho=0.882")
print(f"  Quadratic Delta:                   r=0.731, rho=0.763")

print(f"\nFigures:")
print(f"  Fig XX:  processus_figXX_match_distribution.png")
print(f"  Fig YY:  processus_figYY_tm_dendrograms.png")
print(f"  Fig ZZ:  processus_figZZ_match_heatmaps.png")
print(f"  Fig AAA: processus_figAAA_positional_reuse.png")
print(f"  Fig BBB: processus_figBBB_method_comparison.png")
