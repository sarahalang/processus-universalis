"""
Text Reuse & Dependency Analysis for Processus Universalis.
Builds on the previous FLAME/phonetic normalization work.

Approach:
1. Load clean lowercased text files (from previous project)
2. Apply Cologne phonetic encoding to normalize orthographic variation
3. Compute text similarity at multiple n-gram levels
4. Build dependency trees from text-based similarity
5. Build dependency trees from annotation-based similarity
6. Compare the two: does textual proximity match chemical proximity?
"""

import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict, Counter
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, to_tree
from scipy.spatial.distance import pdist, squareform

# ── Load annotation data ──
with open('/Users/slang/claude/processus_data.json', 'r') as f:
    data = json.load(f)
texts_meta = data['texts']
categories = data['categories']

GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}
GROUP_MARKERS = {'I': 'o', 'II': 's', 'III': 'D'}

# ── Load plain text files ──
TXT_DIR = Path('processus_prev_work/processus_universalis-main/'
               'ProcessusUniversalis_relevant-files-for-2025/'
               'txt-files-lowercase_processus')

def extract_e_name(filename):
    """Extract E-name from filename like G2_E16-... or G1-E2_..."""
    m = re.search(r'E(\d+[ab]?)', filename)
    if m:
        return f'E{m.group(1)}'
    return None

plain_texts = {}
for f in sorted(TXT_DIR.iterdir()):
    if f.suffix == '.txt':
        e_name = extract_e_name(f.name)
        if e_name:
            plain_texts[e_name] = f.read_text(encoding='utf-8', errors='replace').strip()

print(f"Loaded {len(plain_texts)} text files")
# Verify alignment with annotation data
e_names_anno = {t['e_name'] for t in texts_meta}
e_names_text = set(plain_texts.keys())
print(f"  Annotation data: {sorted(e_names_anno)}")
print(f"  Text files:      {sorted(e_names_text)}")
missing = e_names_anno - e_names_text
if missing:
    print(f"  WARNING: Missing text files for: {missing}")
# Use only texts present in both
common_names = sorted(e_names_anno & e_names_text)
print(f"  Using {len(common_names)} texts present in both sources")

# ── Cologne Phonetic Encoding ──
# A phonetic algorithm for German, better suited than Soundex for
# Early Modern German orthographic variation.
def cologne_phonetic(word):
    """
    Cologne phonetic encoding (Kölner Phonetik).
    Maps German words to a phonetic code, normalizing orthographic variation.
    """
    word = word.lower().strip()
    if not word:
        return ''

    # Character mapping table
    # Special handling for context-dependent letters
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
        elif ch == 'd':
            if after in 'csz':
                c = '8'
            else:
                c = '2'
        elif ch == 't':
            if after in 'csz':
                c = '8'
            else:
                c = '2'
        elif ch in 'fvw':
            c = '3'
        elif ch in 'gkq':
            c = '4'
        elif ch == 'c':
            if before in '' and after in 'ahkloqrux':
                c = '4'
            elif after in 'ahkoqux':
                c = '4'
            else:
                c = '8'
        elif ch == 'x':
            if before in 'ckq':
                c = '8'
            else:
                c = '48'
        elif ch == 'l':
            c = '5'
        elif ch in 'mn':
            c = '6'
        elif ch == 'r':
            c = '7'
        elif ch in 'szßẞ':
            c = '8'
        # skip everything else

        # Deduplicate consecutive codes
        if c and c != prev_code:
            code.append(c)
            prev_code = c[-1] if c else ''
        elif c:
            prev_code = c[-1] if c else ''

    # Remove leading zeros (except if the whole code is zeros)
    result = ''.join(code)
    if result:
        result = result[0] + result[1:].replace('0', '')
    return result

def phonetic_normalize(text):
    """Convert text to a sequence of Cologne phonetic codes."""
    words = re.findall(r'[a-zäöüß]+', text.lower())
    return [cologne_phonetic(w) for w in words if len(w) > 1]

# ── N-gram similarity functions ──
def word_ngrams(text, n=4):
    words = text.lower().split()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def phonetic_ngrams(text, n=4):
    codes = phonetic_normalize(text)
    return set(tuple(codes[i:i+n]) for i in range(len(codes) - n + 1))

def jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0.0

# ── Compute similarity matrices ──
print("\nComputing similarity matrices...")

# 1. Raw text n-gram similarity (word 4-grams)
raw_ngrams = {name: word_ngrams(plain_texts[name], 4) for name in common_names}

# 2. Phonetically normalized n-gram similarity
phon_ngrams = {name: phonetic_ngrams(plain_texts[name], 4) for name in common_names}

# 3. Annotation-based similarity (value-level Jaccard)
def annotation_values(t):
    s = set()
    for c in categories:
        for v in t['annotations'][c]['values']:
            s.add((c, v))
    return s

anno_sets = {}
for t in texts_meta:
    if t['e_name'] in common_names:
        anno_sets[t['e_name']] = annotation_values(t)

n = len(common_names)
sim_raw = np.zeros((n, n))
sim_phon = np.zeros((n, n))
sim_anno = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            sim_raw[i,j] = sim_phon[i,j] = sim_anno[i,j] = 1.0
        elif i < j:
            sr = jaccard(raw_ngrams[common_names[i]], raw_ngrams[common_names[j]])
            sp = jaccard(phon_ngrams[common_names[i]], phon_ngrams[common_names[j]])
            sa = jaccard(anno_sets[common_names[i]], anno_sets[common_names[j]])
            sim_raw[i,j] = sim_raw[j,i] = sr
            sim_phon[i,j] = sim_phon[j,i] = sp
            sim_anno[i,j] = sim_anno[j,i] = sa

print("  Done.")

# Print comparison stats
print("\nSimilarity correlation (Mantel-like):")
upper = np.triu_indices(n, k=1)
from scipy.stats import pearsonr, spearmanr
r_raw_anno, p_raw = pearsonr(sim_raw[upper], sim_anno[upper])
r_phon_anno, p_phon = pearsonr(sim_phon[upper], sim_anno[upper])
r_raw_phon, p_rp = pearsonr(sim_raw[upper], sim_phon[upper])
print(f"  Raw text ↔ Annotations:     r={r_raw_anno:.3f}  (p={p_raw:.4f})")
print(f"  Phonetic  ↔ Annotations:    r={r_phon_anno:.3f}  (p={p_phon:.4f})")
print(f"  Raw text  ↔ Phonetic:       r={r_raw_phon:.3f}  (p={p_rp:.4f})")

# ── Helper: get metadata ──
meta_by_name = {t['e_name']: t for t in texts_meta}

def get_group(name):
    return meta_by_name[name]['new_group']

def text_label(name):
    t = meta_by_name[name]
    date_str = f" [{t['date']}]" if t['date'] else ""
    return f"{name} ({t['a_name']}){date_str}"

# ═══════════════════════════════════════════════════════════════════
# FIGURE G: Three-way similarity comparison
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

matrices = [
    (sim_raw, 'Word 4-gram\n(raw text)', 'YlOrRd', 0.35),
    (sim_phon, 'Phonetic 4-gram\n(Cologne encoding)', 'YlOrRd', 0.35),
    (sim_anno, 'Annotation values\n(expert key-values)', 'YlGnBu', 1.0),
]

for ax, (mat, title, cmap, vmax) in zip(axes, matrices):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{name}" for name in common_names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{name}" for name in common_names], fontsize=8)
    for i, name in enumerate(common_names):
        grp = get_group(name)
        ax.get_xticklabels()[i].set_color(GROUP_COLORS[grp])
        ax.get_yticklabels()[i].set_color(GROUP_COLORS[grp])
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')

fig.suptitle('Three Similarity Measures Compared\n'
             'Does phonetic normalization reveal hidden text reuse?',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figG_three_similarities.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig G saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE H: Phonetic gain — where does phonetic normalization help?
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter: raw similarity vs phonetic similarity for each pair
for i in range(n):
    for j in range(i+1, n):
        g1, g2 = get_group(common_names[i]), get_group(common_names[j])
        same_group = g1 == g2
        color = GROUP_COLORS[g1] if same_group else '#999999'
        marker = 'o' if same_group else 'x'
        ax.scatter(sim_raw[i,j], sim_phon[i,j], c=color, marker=marker,
                   s=40, alpha=0.7, zorder=2)

# Diagonal line (no phonetic gain)
lim = max(sim_raw[upper].max(), sim_phon[upper].max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Raw text 4-gram Jaccard', fontsize=12)
ax.set_ylabel('Phonetic 4-gram Jaccard', fontsize=12)
ax.set_title('Phonetic Normalization Gain\n'
             '(points above diagonal = phonetic encoding finds more overlap)',
             fontsize=13, fontweight='bold')

legend_elements = [
    mlines.Line2D([], [], color=GROUP_COLORS['I'], marker='o', linestyle='None',
                  label='Within Gruppe I'),
    mlines.Line2D([], [], color=GROUP_COLORS['II'], marker='o', linestyle='None',
                  label='Within Gruppe II'),
    mlines.Line2D([], [], color=GROUP_COLORS['III'], marker='o', linestyle='None',
                  label='Within Gruppe III'),
    mlines.Line2D([], [], color='#999', marker='x', linestyle='None',
                  label='Between groups'),
    mlines.Line2D([], [], color='k', linestyle='--', alpha=0.3, label='No gain line'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figH_phonetic_gain.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig H saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE I: Comparative dendrograms — text-based vs annotation-based
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

dendro_data = [
    (1.0 - sim_raw,  'Word 4-gram (raw text)'),
    (1.0 - sim_phon, 'Phonetic 4-gram (Cologne)'),
    (1.0 - sim_anno, 'Annotation values (expert)'),
]

leaf_orders = []
for ax, (dist_mat, title) in zip(axes, dendro_data):
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method='ward')
    labels = [text_label(name) for name in common_names]
    label_colors = {text_label(name): GROUP_COLORS[get_group(name)] for name in common_names}

    dend = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=55, leaf_font_size=9,
                      color_threshold=0)
    leaf_orders.append(dend['leaves'])

    for lbl in ax.get_xticklabels():
        lbl.set_color(label_colors.get(lbl.get_text(), 'black'))
        lbl.set_fontweight('bold')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward distance')

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g],
                  label=f'Gruppe {g}') for g in ['I', 'II', 'III']]
axes[2].legend(handles=legend_patches, loc='upper right', fontsize=10)

fig.suptitle('Comparative Dendrograms: Text-Based vs Annotation-Based Clustering\n'
             'Do texts that share wording also share chemical content?',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figI_comparative_dendrograms.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig I saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE J: Nearest-neighbour agreement
# For each text, who is its closest neighbour by text vs by annotation?
# ═══════════════════════════════════════════════════════════════════
def nearest_neighbours(sim_matrix, names):
    """Return dict: name -> (nearest_name, similarity)"""
    nn = {}
    for i, name in enumerate(names):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        j = np.argmax(sims)
        nn[name] = (names[j], sims[j])
    return nn

nn_raw = nearest_neighbours(sim_raw, common_names)
nn_phon = nearest_neighbours(sim_phon, common_names)
nn_anno = nearest_neighbours(sim_anno, common_names)

print("\nNearest-neighbour comparison:")
print(f"{'Text':<12} {'By raw text':<16} {'By phonetic':<16} {'By annotation':<16} {'Text=Anno?'}")
print("-" * 80)
agree_raw = 0
agree_phon = 0
for name in common_names:
    raw_nn = nn_raw[name][0]
    phon_nn = nn_phon[name][0]
    anno_nn = nn_anno[name][0]
    match_raw = "YES" if raw_nn == anno_nn else ""
    match_phon = "YES" if phon_nn == anno_nn else ""
    if raw_nn == anno_nn: agree_raw += 1
    if phon_nn == anno_nn: agree_phon += 1
    print(f"{name:<12} {raw_nn:<16} {phon_nn:<16} {anno_nn:<16} raw:{match_raw:<4} phon:{match_phon}")

print(f"\nAgreement rate: raw→anno {agree_raw}/{n} ({100*agree_raw/n:.0f}%),  "
      f"phon→anno {agree_phon}/{n} ({100*agree_phon/n:.0f}%)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE J: Visual nearest-neighbour comparison
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 10))

# Layout: texts as nodes in a circle
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
# Offset start so groups cluster visually
# Sort by group then name for the circular layout
sorted_names = sorted(common_names, key=lambda name: (
    {'I':0,'II':1,'III':2}[get_group(name)],
    int(re.search(r'\d+', name).group())
))
pos = {name: (np.cos(angles[i]), np.sin(angles[i])) for i, name in enumerate(sorted_names)}

# Draw nearest-neighbour edges
# Annotation edges (solid, thick)
drawn = set()
for name in sorted_names:
    partner = nn_anno[name][0]
    key = tuple(sorted([name, partner]))
    if key not in drawn:
        drawn.add(key)
        x1, y1 = pos[name]
        x2, y2 = pos[partner]
        ax.plot([x1,x2], [y1,y2], color='#2c3e50', linewidth=2.5, alpha=0.6,
                zorder=1, solid_capstyle='round')

# Phonetic edges (dashed, thinner) — only where different from annotation
drawn2 = set()
for name in sorted_names:
    partner = nn_phon[name][0]
    key = tuple(sorted([name, partner]))
    if key not in drawn2 and key not in drawn:
        drawn2.add(key)
        x1, y1 = pos[name]
        x2, y2 = pos[partner]
        ax.plot([x1,x2], [y1,y2], color='#e67e22', linewidth=1.5, alpha=0.5,
                linestyle='--', zorder=1)

# Draw nodes
for name in sorted_names:
    x, y = pos[name]
    grp = get_group(name)
    t = meta_by_name[name]
    is_dated = t['date'] is not None
    marker = 'o' if is_dated else 's'
    ax.scatter(x, y, s=300, c=GROUP_COLORS[grp], marker=marker,
              edgecolors='black' if is_dated else 'gray',
              linewidths=1.5, zorder=5)
    # Label outside the circle
    offset = 1.15
    ax.text(x*offset, y*offset, text_label(name),
            ha='center', va='center', fontsize=8, fontweight='bold',
            color=GROUP_COLORS[grp])

ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
ax.axis('off')

legend_elements = [
    mlines.Line2D([], [], color='#2c3e50', linewidth=2.5, alpha=0.6,
                  label='Nearest neighbour (annotations)'),
    mlines.Line2D([], [], color='#e67e22', linewidth=1.5, alpha=0.5,
                  linestyle='--', label='Nearest neighbour (phonetic text) — where different'),
    mpatches.Patch(color=GROUP_COLORS['I'], label='Gruppe I'),
    mpatches.Patch(color=GROUP_COLORS['II'], label='Gruppe II'),
    mpatches.Patch(color=GROUP_COLORS['III'], label='Gruppe III'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)
ax.set_title('Nearest-Neighbour Graph: Annotations vs Phonetic Text\n'
             'Do the "closest relative" assignments agree?',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figJ_nn_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig J saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE K: Cophenetic distance correlation
# More rigorous comparison of tree topologies
# ═══════════════════════════════════════════════════════════════════
from scipy.cluster.hierarchy import cophenet

dist_phon = squareform(1.0 - sim_phon)
dist_anno = squareform(1.0 - sim_anno)
dist_raw = squareform(1.0 - sim_raw)

Z_phon = linkage(dist_phon, method='ward')
Z_anno = linkage(dist_anno, method='ward')
Z_raw = linkage(dist_raw, method='ward')

_, coph_phon_cond = cophenet(Z_phon, dist_phon)
_, coph_anno_cond = cophenet(Z_anno, dist_anno)
_, coph_raw_cond = cophenet(Z_raw, dist_raw)
coph_phon = squareform(coph_phon_cond)
coph_anno = squareform(coph_anno_cond)
coph_raw = squareform(coph_raw_cond)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: raw text tree distances vs annotation tree distances
for i in range(n):
    for j in range(i+1, n):
        g1, g2 = get_group(common_names[i]), get_group(common_names[j])
        same = g1 == g2
        color = GROUP_COLORS[g1] if same else '#999'
        ax1.scatter(coph_raw[i,j], coph_anno[i,j], c=color,
                   s=25, alpha=0.6, marker='o' if same else 'x')

r_coph_raw, _ = pearsonr(coph_raw[upper], coph_anno[upper])
ax1.set_xlabel('Cophenetic distance (raw text tree)', fontsize=11)
ax1.set_ylabel('Cophenetic distance (annotation tree)', fontsize=11)
ax1.set_title(f'Raw Text vs Annotation Trees\nr = {r_coph_raw:.3f}', fontsize=12, fontweight='bold')

# Right: phonetic tree distances vs annotation tree distances
for i in range(n):
    for j in range(i+1, n):
        g1, g2 = get_group(common_names[i]), get_group(common_names[j])
        same = g1 == g2
        color = GROUP_COLORS[g1] if same else '#999'
        ax2.scatter(coph_phon[i,j], coph_anno[i,j], c=color,
                   s=25, alpha=0.6, marker='o' if same else 'x')

r_coph_phon, _ = pearsonr(coph_phon[upper], coph_anno[upper])
ax2.set_xlabel('Cophenetic distance (phonetic text tree)', fontsize=11)
ax2.set_ylabel('Cophenetic distance (annotation tree)', fontsize=11)
ax2.set_title(f'Phonetic Text vs Annotation Trees\nr = {r_coph_phon:.3f}', fontsize=12, fontweight='bold')

fig.suptitle('Tree Topology Comparison: How Well Does Text Structure Predict Annotation Structure?',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figK_cophenetic.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig K saved")


# ═══════════════════════════════════════════════════════════════════
# Print summary statistics
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nSimilarity correlations (all {n*(n-1)//2} pairs):")
print(f"  Raw text ↔ Annotations:     Pearson r = {r_raw_anno:.3f}")
print(f"  Phonetic ↔ Annotations:     Pearson r = {r_phon_anno:.3f}")
print(f"  Improvement from phonetic:  Δr = {r_phon_anno - r_raw_anno:+.3f}")

print(f"\nTree topology correlations (cophenetic distances):")
print(f"  Raw text tree ↔ Anno tree:  r = {r_coph_raw:.3f}")
print(f"  Phonetic tree ↔ Anno tree:  r = {r_coph_phon:.3f}")
print(f"  Improvement from phonetic:  Δr = {r_coph_phon - r_coph_raw:+.3f}")

print(f"\nNearest-neighbour agreement with annotations:")
print(f"  Raw text:  {agree_raw}/{n} ({100*agree_raw/n:.0f}%)")
print(f"  Phonetic:  {agree_phon}/{n} ({100*agree_phon/n:.0f}%)")

# Within-group vs between-group similarity
for label, mat in [("Raw text", sim_raw), ("Phonetic", sim_phon), ("Annotation", sim_anno)]:
    within = []
    between = []
    for i in range(n):
        for j in range(i+1, n):
            g1, g2 = get_group(common_names[i]), get_group(common_names[j])
            if g1 == g2:
                within.append(mat[i,j])
            else:
                between.append(mat[i,j])
    ratio = (np.mean(within) / np.mean(between)) if np.mean(between) > 0 else float('inf')
    print(f"\n{label} — within-group avg: {np.mean(within):.4f}, between-group avg: {np.mean(between):.4f}, ratio: {ratio:.2f}x")

print("\nAll figures saved to processus-universalis-graphics/")
