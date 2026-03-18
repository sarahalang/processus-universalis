"""
Deep-dive: Where and why does text-based similarity diverge from
annotation-based similarity in the Processus Universalis corpus?

This script produces Figures L–Q and a detailed diagnostic report.
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict, Counter
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

# ── Load data ──
with open('/Users/slang/claude/processus_data.json', 'r') as f:
    data = json.load(f)
texts_meta = data['texts']
categories = data['categories']

GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}
meta_by_name = {t['e_name']: t for t in texts_meta}

def get_group(name):
    return meta_by_name[name]['new_group']

# ── Load plain text files ──
TXT_DIR = Path('processus_prev_work/processus_universalis-main/'
               'ProcessusUniversalis_relevant-files-for-2025/'
               'txt-files-lowercase_processus')

def extract_e_name(filename):
    m = re.search(r'E(\d+[ab]?)', filename)
    return f'E{m.group(1)}' if m else None

plain_texts = {}
for f in sorted(TXT_DIR.iterdir()):
    if f.suffix == '.txt':
        e_name = extract_e_name(f.name)
        if e_name:
            plain_texts[e_name] = f.read_text(encoding='utf-8', errors='replace').strip()

e_names_anno = {t['e_name'] for t in texts_meta}
e_names_text = set(plain_texts.keys())
common_names = sorted(e_names_anno & e_names_text)
n = len(common_names)
print(f"Working with {n} texts")

# ── Cologne Phonetic Encoding ──
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
            if after in 'ahkoqux':
                c = '4'
            else:
                c = '8'
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

def phonetic_normalize(text):
    words = re.findall(r'[a-zäöüß]+', text.lower())
    return [cologne_phonetic(w) for w in words if len(w) > 1]

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
raw_ngrams = {name: word_ngrams(plain_texts[name], 4) for name in common_names}
phon_ngrams = {name: phonetic_ngrams(plain_texts[name], 4) for name in common_names}

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

sim_raw = np.zeros((n, n))
sim_phon = np.zeros((n, n))
sim_anno = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        sr = jaccard(raw_ngrams[common_names[i]], raw_ngrams[common_names[j]])
        sp = jaccard(phon_ngrams[common_names[i]], phon_ngrams[common_names[j]])
        sa = jaccard(anno_sets[common_names[i]], anno_sets[common_names[j]])
        sim_raw[i,j] = sim_raw[j,i] = sr
        sim_phon[i,j] = sim_phon[j,i] = sp
        sim_anno[i,j] = sim_anno[j,i] = sa
np.fill_diagonal(sim_raw, 1.0)
np.fill_diagonal(sim_phon, 1.0)
np.fill_diagonal(sim_anno, 1.0)

upper = np.triu_indices(n, k=1)

# ── Nearest-neighbour computation ──
def nearest_neighbours(sim_matrix, names):
    nn = {}
    for i, name in enumerate(names):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        j = np.argmax(sims)
        nn[name] = (names[j], sims[j])
    return nn

nn_raw = nearest_neighbours(sim_raw, common_names)
nn_phon = nearest_neighbours(sim_phon, common_names)
nn_anno = nearest_neighbours(sim_anno, common_names)


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Residual analysis — which pairs diverge most?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 1: Pairs with largest text↔annotation divergence")
print("="*70)

# Compute normalized residuals
# First normalize both to [0,1] range for comparable residuals
raw_vals = sim_raw[upper]
anno_vals = sim_anno[upper]

# Residuals: annotation - (scaled raw text)
# Use rank-based approach to find pairs that disagree most
pairs = []
for idx in range(len(raw_vals)):
    i, j = upper[0][idx], upper[1][idx]
    pairs.append({
        'name_i': common_names[i],
        'name_j': common_names[j],
        'sim_raw': raw_vals[idx],
        'sim_phon': sim_phon[i, j],
        'sim_anno': anno_vals[idx],
        'group_i': get_group(common_names[i]),
        'group_j': get_group(common_names[j]),
    })

# Rank each pair by both metrics
for metric in ['sim_raw', 'sim_anno']:
    sorted_pairs = sorted(pairs, key=lambda p: p[metric], reverse=True)
    for rank, p in enumerate(sorted_pairs):
        p[f'rank_{metric}'] = rank

# Compute rank difference
for p in pairs:
    p['rank_diff'] = p['rank_sim_anno'] - p['rank_sim_raw']  # positive = higher anno rank than text rank

# Sort by absolute rank difference
pairs_by_divergence = sorted(pairs, key=lambda p: abs(p['rank_diff']), reverse=True)

print("\nTop 15 pairs where text similarity and annotation similarity disagree most:")
print(f"{'Pair':<20} {'Text sim':<10} {'Anno sim':<10} {'Text rank':<10} {'Anno rank':<10} {'Rank Δ':<8} {'Type'}")
print("-" * 90)
for p in pairs_by_divergence[:15]:
    pair_str = f"{p['name_i']}-{p['name_j']}"
    same = "within" if p['group_i'] == p['group_j'] else "between"
    direction = "anno>text" if p['rank_diff'] < 0 else "text>anno"
    print(f"{pair_str:<20} {p['sim_raw']:.4f}    {p['sim_anno']:.4f}    "
          f"{p['rank_sim_raw']:<10} {p['rank_sim_anno']:<10} {p['rank_diff']:<+8} {same} ({direction})")


# ═══════════════════════════════════════════════════════════════════
# FIGURE L: Divergence scatter — text vs annotation similarity
# with the most divergent pairs labelled
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))

for p in pairs:
    g1, g2 = p['group_i'], p['group_j']
    same = g1 == g2
    color = GROUP_COLORS[g1] if same else '#aaa'
    marker = 'o' if same else 'x'
    ax.scatter(p['sim_raw'], p['sim_anno'], c=color, marker=marker,
               s=50, alpha=0.6, zorder=2)

# Label the top divergent pairs
top_divergent = pairs_by_divergence[:10]
for p in top_divergent:
    label = f"{p['name_i']}–{p['name_j']}"
    # Determine arrow direction for visual clarity
    ax.annotate(label, (p['sim_raw'], p['sim_anno']),
                textcoords="offset points", xytext=(8, 6),
                fontsize=7, fontweight='bold', color='#333',
                arrowprops=dict(arrowstyle='-', color='#999', lw=0.5))

# Add trend line
z = np.polyfit(raw_vals, anno_vals, 1)
x_line = np.linspace(0, max(raw_vals)*1.1, 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.3, linewidth=1,
        label=f'Linear fit (r={pearsonr(raw_vals, anno_vals)[0]:.3f})')

ax.set_xlabel('Raw Text 4-gram Jaccard Similarity', fontsize=12)
ax.set_ylabel('Annotation Value Jaccard Similarity', fontsize=12)
ax.set_title('Text Similarity vs Annotation Similarity\n'
             'Labelled pairs show the largest disagreements',
             fontsize=14, fontweight='bold')

legend_elements = [
    mlines.Line2D([], [], color=GROUP_COLORS['I'], marker='o', linestyle='None', label='Within Gruppe I'),
    mlines.Line2D([], [], color=GROUP_COLORS['II'], marker='o', linestyle='None', label='Within Gruppe II'),
    mlines.Line2D([], [], color=GROUP_COLORS['III'], marker='o', linestyle='None', label='Within Gruppe III'),
    mlines.Line2D([], [], color='#aaa', marker='x', linestyle='None', label='Between groups'),
    mlines.Line2D([], [], color='k', linestyle='--', alpha=0.3, label=f'Linear fit'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figL_divergence_scatter.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nFig L saved")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Per-category text predictability
# For each annotation category, can text similarity predict whether
# two texts agree on that category?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 2: Per-category text predictability")
print("="*70)

# Recipe phases (from visualize_evolution.py)
PHASES = [
    ('Preface', 0, 2),
    ('Earth & Sampling', 2, 9),
    ('Extraction & Salt Work', 9, 16),
    ('Recombination & Gold Work', 16, 22),
    ("Philosopher's Stone & Projection", 22, 30),
]

def get_phase(cat_idx):
    for name, start, end in PHASES:
        if start <= cat_idx < end:
            return name
    return "Unknown"

cat_results = []
for c_idx, cat in enumerate(categories):
    # For each pair, compute:
    # 1. Do they agree on this category's values? (category-specific Jaccard)
    # 2. What's their text similarity?
    cat_sims = []
    text_sims = []
    for i in range(n):
        for j in range(i+1, n):
            t_i = meta_by_name[common_names[i]]
            t_j = meta_by_name[common_names[j]]
            vals_i = set(t_i['annotations'][cat]['values'])
            vals_j = set(t_j['annotations'][cat]['values'])
            # Category-specific Jaccard
            inter = len(vals_i & vals_j)
            union = len(vals_i | vals_j)
            cat_jac = inter / union if union else 0.0
            cat_sims.append(cat_jac)
            text_sims.append(sim_raw[i,j])

    r, p = pearsonr(text_sims, cat_sims)
    present_count = sum(1 for name in common_names
                        if meta_by_name[name]['annotations'][cat]['present'])
    avg_cat_sim = np.mean(cat_sims)
    cat_results.append({
        'category': cat,
        'cat_idx': c_idx,
        'phase': get_phase(c_idx),
        'r': r,
        'p': p,
        'present_count': present_count,
        'avg_cat_sim': avg_cat_sim,
        'n_distinct_values': len(set(v for name in common_names
                                     for v in meta_by_name[name]['annotations'][cat]['values']
                                     if v != 'FEHLT')),
    })

cat_results.sort(key=lambda x: -abs(x['r']))
print(f"\n{'Category':<55} {'r':<8} {'p':<10} {'#Texts':<7} {'#Vals':<6} {'Phase'}")
print("-" * 130)
for cr in cat_results:
    sig = "***" if cr['p'] < 0.001 else "**" if cr['p'] < 0.01 else "*" if cr['p'] < 0.05 else ""
    print(f"{cr['category'][:54]:<55} {cr['r']:+.3f}   {cr['p']:.4f}{sig:<4} {cr['present_count']:<7} "
          f"{cr['n_distinct_values']:<6} {cr['phase']}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE M: Per-category text predictability
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 10))

# Color by phase
PHASE_COLORS = {
    'Preface': '#9b59b6',
    'Earth & Sampling': '#8B4513',
    'Extraction & Salt Work': '#1abc9c',
    'Recombination & Gold Work': '#e67e22',
    "Philosopher's Stone & Projection": '#e74c3c',
}

# Sort categories by recipe order for the chart
cat_results_ordered = sorted(cat_results, key=lambda x: x['cat_idx'])
y_pos = range(len(cat_results_ordered))

bars = ax.barh(y_pos,
               [cr['r'] for cr in cat_results_ordered],
               color=[PHASE_COLORS[cr['phase']] for cr in cat_results_ordered],
               height=0.7, edgecolor='white', linewidth=0.5)

# Mark significance
for i, cr in enumerate(cat_results_ordered):
    if cr['p'] < 0.05:
        x_pos = cr['r'] + (0.01 if cr['r'] >= 0 else -0.01)
        ha = 'left' if cr['r'] >= 0 else 'right'
        stars = "***" if cr['p'] < 0.001 else "**" if cr['p'] < 0.01 else "*"
        ax.text(x_pos, i, stars, va='center', ha=ha, fontsize=8, color='#333')

ax.set_yticks(y_pos)
ax.set_yticklabels([cr['category'][:60] for cr in cat_results_ordered], fontsize=8)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Pearson r (text 4-gram similarity ↔ category-level Jaccard)', fontsize=11)
ax.set_title('Per-Category Text Predictability\n'
             'Which annotation categories can be predicted from text similarity alone?',
             fontsize=14, fontweight='bold')

legend_patches = [mpatches.Patch(color=PHASE_COLORS[p], label=p) for p in PHASE_COLORS]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, title='Recipe Phase')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figM_category_predictability.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig M saved")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 3: The 5 disagreement cases — deep dive
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 3: Nearest-neighbour disagreements — detailed")
print("="*70)

disagreement_details = []
for name in common_names:
    raw_nn_name, raw_nn_sim = nn_raw[name]
    anno_nn_name, anno_nn_sim = nn_anno[name]
    if raw_nn_name != anno_nn_name:
        # Get index for similarity lookup
        idx_self = common_names.index(name)
        idx_raw = common_names.index(raw_nn_name)
        idx_anno = common_names.index(anno_nn_name)

        # What's the text sim to the anno-nn? And anno sim to the text-nn?
        text_sim_to_anno_nn = sim_raw[idx_self, idx_anno]
        anno_sim_to_text_nn = sim_anno[idx_self, idx_raw]

        # How close was the second-best in each metric?
        raw_sims = sim_raw[idx_self].copy()
        raw_sims[idx_self] = -1
        raw_sorted_idx = np.argsort(raw_sims)[::-1]
        raw_2nd = common_names[raw_sorted_idx[1]]
        raw_2nd_sim = raw_sims[raw_sorted_idx[1]]

        anno_sims = sim_anno[idx_self].copy()
        anno_sims[idx_self] = -1
        anno_sorted_idx = np.argsort(anno_sims)[::-1]
        anno_2nd = common_names[anno_sorted_idx[1]]
        anno_2nd_sim = anno_sims[anno_sorted_idx[1]]

        # Shared annotation values between self↔text_nn vs self↔anno_nn
        self_anno = anno_sets[name]
        text_nn_anno = anno_sets[raw_nn_name]
        anno_nn_anno = anno_sets[anno_nn_name]

        shared_text_nn = self_anno & text_nn_anno
        shared_anno_nn = self_anno & anno_nn_anno
        unique_to_text_nn = shared_text_nn - shared_anno_nn
        unique_to_anno_nn = shared_anno_nn - shared_text_nn

        detail = {
            'name': name,
            'group': get_group(name),
            'text_nn': raw_nn_name,
            'text_nn_group': get_group(raw_nn_name),
            'text_nn_sim': raw_nn_sim,
            'anno_nn': anno_nn_name,
            'anno_nn_group': get_group(anno_nn_name),
            'anno_nn_sim': anno_nn_sim,
            'text_sim_to_anno_nn': text_sim_to_anno_nn,
            'anno_sim_to_text_nn': anno_sim_to_text_nn,
            'gap_text': raw_nn_sim - raw_sims[idx_anno],  # margin at text level
            'gap_anno': anno_nn_sim - anno_sims[idx_raw],  # margin at anno level
            'n_shared_text_nn': len(shared_text_nn),
            'n_shared_anno_nn': len(shared_anno_nn),
            'n_unique_to_text_nn': len(unique_to_text_nn),
            'n_unique_to_anno_nn': len(unique_to_anno_nn),
            'unique_cats_text_nn': set(c for c, v in unique_to_text_nn),
            'unique_cats_anno_nn': set(c for c, v in unique_to_anno_nn),
        }
        disagreement_details.append(detail)

        print(f"\n  {name} (Gruppe {detail['group']}):")
        print(f"    Text says nearest = {raw_nn_name} (Gr.{detail['text_nn_group']}, "
              f"text sim={raw_nn_sim:.4f})")
        print(f"    Anno says nearest = {anno_nn_name} (Gr.{detail['anno_nn_group']}, "
              f"anno sim={anno_nn_sim:.4f})")
        print(f"    Text sim to anno-nn:  {text_sim_to_anno_nn:.4f} "
              f"(gap: {detail['gap_text']:.4f})")
        print(f"    Anno sim to text-nn:  {anno_sim_to_text_nn:.4f} "
              f"(gap: {detail['gap_anno']:.4f})")
        print(f"    Shared anno values with text-nn:  {detail['n_shared_text_nn']}")
        print(f"    Shared anno values with anno-nn:  {detail['n_shared_anno_nn']}")
        print(f"    Values shared with text-nn but NOT anno-nn ({detail['n_unique_to_text_nn']}):")
        if detail['unique_cats_text_nn']:
            for cat in sorted(detail['unique_cats_text_nn']):
                vals = [v for c, v in unique_to_text_nn if c == cat]
                print(f"      - {cat}: {vals}")
        print(f"    Values shared with anno-nn but NOT text-nn ({detail['n_unique_to_anno_nn']}):")
        if detail['unique_cats_anno_nn']:
            for cat in sorted(detail['unique_cats_anno_nn']):
                vals = [v for c, v in unique_to_anno_nn if c == cat]
                print(f"      - {cat}: {vals}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE N: Disagreement case profiles
# For each disagreeing text, show its similarity to all others by
# both metrics, highlighting where the methods disagree
# ═══════════════════════════════════════════════════════════════════
disagree_names = [d['name'] for d in disagreement_details]
n_disagree = len(disagree_names)

fig, axes = plt.subplots(n_disagree, 1, figsize=(14, 3.5 * n_disagree))
if n_disagree == 1:
    axes = [axes]

for ax, d in zip(axes, disagreement_details):
    name = d['name']
    idx_self = common_names.index(name)
    others = [cn for cn in common_names if cn != name]

    # Get similarities to all others
    text_sims = []
    anno_sims_list = []
    for other in others:
        idx_other = common_names.index(other)
        text_sims.append(sim_raw[idx_self, idx_other])
        anno_sims_list.append(sim_anno[idx_self, idx_other])

    x = np.arange(len(others))
    width = 0.35

    # Normalize text sims for visual comparison (scale to annotation range)
    text_max = max(text_sims) if max(text_sims) > 0 else 1
    anno_max = max(anno_sims_list) if max(anno_sims_list) > 0 else 1
    text_scaled = [t * (anno_max / text_max) for t in text_sims]

    bars1 = ax.bar(x - width/2, text_scaled, width, label='Text sim (scaled)',
                   color='#3498db', alpha=0.7)
    bars2 = ax.bar(x + width/2, anno_sims_list, width, label='Anno sim',
                   color='#e74c3c', alpha=0.7)

    # Highlight the two disagreeing nearest neighbours
    for i, other in enumerate(others):
        if other == d['text_nn']:
            bars1[i].set_edgecolor('blue')
            bars1[i].set_linewidth(3)
        if other == d['anno_nn']:
            bars2[i].set_edgecolor('red')
            bars2[i].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(others, rotation=45, ha='right', fontsize=8)
    for i, other in enumerate(others):
        ax.get_xticklabels()[i].set_color(GROUP_COLORS[get_group(other)])
        ax.get_xticklabels()[i].set_fontweight('bold')

    ax.set_title(f"{name} (Gruppe {d['group']}): text→{d['text_nn']}, "
                 f"anno→{d['anno_nn']}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('Similarity')

plt.suptitle('Nearest-Neighbour Disagreement Profiles\n'
             'Blue border = text nearest-neighbour, Red border = annotation nearest-neighbour',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figN_disagreement_profiles.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig N saved")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 4: Per-phase correlation — does text better predict
# annotations in early vs late recipe phases?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 4: Per-phase text↔annotation correlation")
print("="*70)

phase_results = []
for phase_name, start, end in PHASES:
    phase_cats = categories[start:end]
    # Compute phase-specific annotation Jaccard
    phase_sims = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            t_i = meta_by_name[common_names[i]]
            t_j = meta_by_name[common_names[j]]
            vals_i = set()
            vals_j = set()
            for cat in phase_cats:
                for v in t_i['annotations'][cat]['values']:
                    vals_i.add((cat, v))
                for v in t_j['annotations'][cat]['values']:
                    vals_j.add((cat, v))
            inter = len(vals_i & vals_j)
            union = len(vals_i | vals_j)
            jac = inter / union if union else 0.0
            phase_sims[i,j] = phase_sims[j,i] = jac

    r, p = pearsonr(sim_raw[upper], phase_sims[upper])
    r_phon, p_phon = pearsonr(sim_phon[upper], phase_sims[upper])

    phase_results.append({
        'phase': phase_name,
        'n_cats': end - start,
        'r_raw': r,
        'r_phon': r_phon,
        'p_raw': p,
        'p_phon': p_phon,
        'phase_sims': phase_sims,
    })
    print(f"  {phase_name:<40} r_raw={r:.3f} (p={p:.4f})  r_phon={r_phon:.3f} (p={p_phon:.4f})")


# ═══════════════════════════════════════════════════════════════════
# FIGURE O: Per-phase correlation comparison
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(phase_results))
width = 0.35

bars1 = ax.bar(x - width/2, [pr['r_raw'] for pr in phase_results], width,
               label='Raw text ↔ Phase annotations', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, [pr['r_phon'] for pr in phase_results], width,
               label='Phonetic text ↔ Phase annotations', color='#2ecc71', alpha=0.8)

# Add significance markers
for i, pr in enumerate(phase_results):
    for offset, p_val, bar in [(-width/2, pr['p_raw'], bars1[i]),
                                (width/2, pr['p_phon'], bars2[i])]:
        stars = ""
        if p_val < 0.001: stars = "***"
        elif p_val < 0.01: stars = "**"
        elif p_val < 0.05: stars = "*"
        if stars:
            ax.text(i + offset, bar.get_height() + 0.02, stars,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([pr['phase'] for pr in phase_results], fontsize=10, rotation=15, ha='right')
ax.set_ylabel('Pearson r', fontsize=12)
ax.set_title('Text↔Annotation Correlation by Recipe Phase\n'
             'In which parts of the recipe does text best predict expert annotations?',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylim(-0.1, max(pr['r_raw'] for pr in phase_results) + 0.15)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figO_phase_correlation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig O saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE P: Per-phase scatter grid — text vs annotation similarity
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

for ax, pr in zip(axes, phase_results):
    phase_sims = pr['phase_sims']
    for i in range(n):
        for j in range(i+1, n):
            g1, g2 = get_group(common_names[i]), get_group(common_names[j])
            same = g1 == g2
            color = GROUP_COLORS[g1] if same else '#ccc'
            marker = 'o' if same else 'x'
            ax.scatter(sim_raw[i,j], phase_sims[i,j], c=color, marker=marker,
                       s=20, alpha=0.5)

    ax.set_xlabel('Text 4-gram sim', fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel('Phase annotation sim', fontsize=10)
    ax.set_title(f"{pr['phase']}\n(r={pr['r_raw']:.3f})", fontsize=10, fontweight='bold')

fig.suptitle('Text vs Annotation Similarity by Recipe Phase',
             fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figP_phase_scatter.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig P saved")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 5: Shared n-gram content in disagreement pairs
# What specific text passages are shared?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 5: Shared text passages in key pairs")
print("="*70)

# For the most interesting pairs, show what 4-grams they share
interesting_pairs = [
    # High text sim, low annotation sim
    ('E34', 'E35'),  # Text says nearest but they're in different groups
    ('E37', 'E38'),  # Strong text + annotation match
    # Disagreement cases
    ('E11', 'E38'),  # Text says E38, annotations say E22
    ('E11', 'E22'),  # Annotations say E22
    ('E45', 'E34'),  # Text says E34
    ('E45', 'E44'),  # Annotations say E44
    ('E32b', 'E22'),  # Text says E22
    ('E32b', 'E19'),  # Annotations say E19
]

for name_i, name_j in interesting_pairs:
    if name_i not in raw_ngrams or name_j not in raw_ngrams:
        continue
    shared = raw_ngrams[name_i] & raw_ngrams[name_j]
    idx_i = common_names.index(name_i)
    idx_j = common_names.index(name_j)

    print(f"\n  {name_i} (Gr.{get_group(name_i)}) ↔ {name_j} (Gr.{get_group(name_j)})")
    print(f"    Text sim: {sim_raw[idx_i,idx_j]:.4f}, Anno sim: {sim_anno[idx_i,idx_j]:.4f}")
    print(f"    Shared 4-grams: {len(shared)}")
    print(f"    {name_i} total 4-grams: {len(raw_ngrams[name_i])}")
    print(f"    {name_j} total 4-grams: {len(raw_ngrams[name_j])}")
    if shared:
        # Show a sample
        sample = sorted(shared, key=lambda x: ' '.join(x))[:10]
        print(f"    Sample shared 4-grams:")
        for gram in sample:
            print(f"      \"{' '.join(gram)}\"")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 6: Text length effects
# Does text length confound the comparison?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 6: Text length effects")
print("="*70)

word_counts = {}
for name in common_names:
    word_counts[name] = len(plain_texts[name].split())

print(f"\n{'Text':<12} {'Words':<8} {'4-grams':<10} {'Phon 4-grams':<14} {'Anno values':<12} {'Group'}")
print("-" * 70)
for name in common_names:
    t = meta_by_name[name]
    print(f"{name:<12} {word_counts[name]:<8} {len(raw_ngrams[name]):<10} "
          f"{len(phon_ngrams[name]):<14} {len(anno_sets[name]):<12} Gr.{t['new_group']}")

# Correlation between text length and n-gram count
lengths = [word_counts[name] for name in common_names]
ngram_counts = [len(raw_ngrams[name]) for name in common_names]
anno_counts = [len(anno_sets[name]) for name in common_names]
r_len_ngram, _ = pearsonr(lengths, ngram_counts)
r_len_anno, _ = pearsonr(lengths, anno_counts)
print(f"\n  Word count ↔ 4-gram count:    r = {r_len_ngram:.3f}")
print(f"  Word count ↔ Anno value count: r = {r_len_anno:.3f}")

# Do longer texts have higher similarity? (Length bias)
mean_text_sim = [np.mean([sim_raw[common_names.index(name), common_names.index(other)]
                          for other in common_names if other != name])
                 for name in common_names]
mean_anno_sim = [np.mean([sim_anno[common_names.index(name), common_names.index(other)]
                          for other in common_names if other != name])
                 for name in common_names]
r_len_textsim, _ = pearsonr(lengths, mean_text_sim)
r_len_annosim, _ = pearsonr(lengths, mean_anno_sim)
print(f"  Word count ↔ Mean text sim:   r = {r_len_textsim:.3f}")
print(f"  Word count ↔ Mean anno sim:   r = {r_len_annosim:.3f}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE Q: Text length bias analysis
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: word count vs mean text similarity
ax = axes[0]
for name in common_names:
    idx = common_names.index(name)
    grp = get_group(name)
    ax.scatter(word_counts[name], mean_text_sim[idx], c=GROUP_COLORS[grp],
               s=60, zorder=3)
    ax.annotate(name, (word_counts[name], mean_text_sim[idx]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
                color=GROUP_COLORS[grp])
ax.set_xlabel('Word count', fontsize=11)
ax.set_ylabel('Mean text similarity to all others', fontsize=10)
ax.set_title(f'Text Length vs Text Similarity\n(r={r_len_textsim:.3f})',
             fontsize=11, fontweight='bold')

# Centre: word count vs mean annotation similarity
ax = axes[1]
for name in common_names:
    idx = common_names.index(name)
    grp = get_group(name)
    ax.scatter(word_counts[name], mean_anno_sim[idx], c=GROUP_COLORS[grp],
               s=60, zorder=3)
    ax.annotate(name, (word_counts[name], mean_anno_sim[idx]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
                color=GROUP_COLORS[grp])
ax.set_xlabel('Word count', fontsize=11)
ax.set_ylabel('Mean annotation similarity to all others', fontsize=10)
ax.set_title(f'Text Length vs Annotation Similarity\n(r={r_len_annosim:.3f})',
             fontsize=11, fontweight='bold')

# Right: 4-gram count vs annotation value count
ax = axes[2]
for name in common_names:
    idx = common_names.index(name)
    grp = get_group(name)
    ax.scatter(len(raw_ngrams[name]), len(anno_sets[name]), c=GROUP_COLORS[grp],
               s=60, zorder=3)
    ax.annotate(name, (len(raw_ngrams[name]), len(anno_sets[name])),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
                color=GROUP_COLORS[grp])
ax.set_xlabel('Number of 4-grams', fontsize=11)
ax.set_ylabel('Number of annotation values', fontsize=10)
ax.set_title(f'Feature Counts: Text vs Annotation\n(r={r_len_anno:.3f})',
             fontsize=11, fontweight='bold')

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Gruppe {g}')
                  for g in ['I', 'II', 'III']]
axes[2].legend(handles=legend_patches, loc='lower right', fontsize=9)

fig.suptitle('Does Text Length Bias the Analysis?', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('processus-universalis-graphics/processus_figQ_length_bias.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig Q saved")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 7: What makes annotation similarity "extra"?
# Pairs with high annotation sim but low text sim — what do they
# share in annotations that isn't captured by text?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 7: 'Annotation-extra' pairs — high anno sim, low text sim")
print("="*70)

# Find pairs in the top quartile of annotation sim but bottom half of text sim
anno_threshold = np.percentile(sim_anno[upper], 75)
text_threshold = np.median(sim_raw[upper])

print(f"  Thresholds: anno > {anno_threshold:.4f} (75th pct), text < {text_threshold:.4f} (median)")

anno_extra = []
for i in range(n):
    for j in range(i+1, n):
        if sim_anno[i,j] >= anno_threshold and sim_raw[i,j] <= text_threshold:
            anno_extra.append({
                'name_i': common_names[i],
                'name_j': common_names[j],
                'sim_anno': sim_anno[i,j],
                'sim_raw': sim_raw[i,j],
                'group_i': get_group(common_names[i]),
                'group_j': get_group(common_names[j]),
            })

anno_extra.sort(key=lambda x: -x['sim_anno'])
print(f"\n  Found {len(anno_extra)} 'annotation-extra' pairs:")
for p in anno_extra:
    print(f"    {p['name_i']} (Gr.{p['group_i']}) ↔ {p['name_j']} (Gr.{p['group_j']}):  "
          f"anno={p['sim_anno']:.4f}, text={p['sim_raw']:.4f}")

    # What categories do they agree on?
    t_i = meta_by_name[p['name_i']]
    t_j = meta_by_name[p['name_j']]
    shared_vals = anno_sets[p['name_i']] & anno_sets[p['name_j']]
    shared_cats = set(c for c, v in shared_vals)
    only_i = anno_sets[p['name_i']] - anno_sets[p['name_j']]
    only_j = anno_sets[p['name_j']] - anno_sets[p['name_i']]
    print(f"      Shared annotation values: {len(shared_vals)} across {len(shared_cats)} categories")
    print(f"      Unique to {p['name_i']}: {len(only_i)}, unique to {p['name_j']}: {len(only_j)}")


print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nFigures saved:")
print("  L: Divergence scatter (text vs annotation, labelled outliers)")
print("  M: Per-category text predictability")
print("  N: Disagreement profiles (the 5 nearest-neighbour disagreements)")
print("  O: Per-phase correlation comparison")
print("  P: Per-phase scatter grid")
print("  Q: Text length bias analysis")
