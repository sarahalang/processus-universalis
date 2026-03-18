"""
Processus Universalis — Recipe evolution visualizations.
Focus: chronological recipe flow, group divergence, text reuse, stemmatic relationships.
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import Counter, defaultdict
from adjustText import adjust_text

with open('/Users/slang/claude/processus_data.json', 'r') as f:
    data = json.load(f)

texts = data['texts']
categories = data['categories']
N_CAT = len(categories)
N_TXT = len(texts)

GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}
GROUP_LABELS = {'I': 'Gruppe I', 'II': 'Gruppe II', 'III': 'Gruppe III'}

def text_label(t):
    date_str = f" ({t['date']})" if t['date'] else ""
    return f"{t['e_name']}{date_str}"

def text_label_full(t):
    date_str = f" [{t['date']}]" if t['date'] else ""
    return f"{t['e_name']} ({t['a_name']}){date_str}"

# ═══════════════════════════════════════════════════════════════════
# FIGURE A: Process flow — group agreement at each recipe step
# Shows where in the recipe the three groups start diverging
# ═══════════════════════════════════════════════════════════════════
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 10),
                                      gridspec_kw={'height_ratios': [3, 2]},
                                      sharex=True)

# Top panel: presence rate per group at each step
x = np.arange(N_CAT)
width = 0.25
for gi, g in enumerate(['I', 'II', 'III']):
    g_texts = [t for t in texts if t['new_group'] == g]
    rates = []
    for c in categories:
        rate = sum(1 for t in g_texts if t['annotations'][c]['present']) / len(g_texts)
        rates.append(rate)
    ax_top.bar(x + (gi - 1) * width, rates, width, color=GROUP_COLORS[g],
               label=GROUP_LABELS[g], alpha=0.85)

ax_top.set_ylabel('Fraction of texts including this step', fontsize=11)
ax_top.set_ylim(0, 1.15)
ax_top.legend(fontsize=11, loc='upper right')
ax_top.set_title('Recipe Process Flow: Group Agreement at Each Step',
                 fontsize=14, fontweight='bold')

# Add phase annotations
phases = [
    (0, 2, 'Preface'),
    (2, 9, 'Earth & Sampling'),
    (9, 16, 'Extraction &\nSalt Work'),
    (16, 22, 'Recombination &\nGold Work'),
    (22, 30, 'Philosopher\'s Stone\n& Projection'),
]
for start, end, label in phases:
    mid = (start + end) / 2 - 0.5
    ax_top.axvspan(start - 0.5, end - 0.5, alpha=0.06, color='gray')
    ax_top.text(mid, 1.08, label, ha='center', va='bottom', fontsize=8,
                fontstyle='italic', color='#555')

# Bottom panel: inter-group divergence at each step
# Measured as max difference in presence rate between any two groups
divergences = []
for ci, c in enumerate(categories):
    rates = {}
    for g in ['I', 'II', 'III']:
        g_texts = [t for t in texts if t['new_group'] == g]
        rates[g] = sum(1 for t in g_texts if t['annotations'][c]['present']) / len(g_texts)
    # Max pairwise difference
    vals = list(rates.values())
    divergence = max(vals) - min(vals)
    divergences.append(divergence)

colors = ['#2ecc71' if d < 0.3 else '#f39c12' if d < 0.6 else '#e74c3c' for d in divergences]
ax_bot.bar(x, divergences, color=colors, alpha=0.85, width=0.7)
ax_bot.set_ylabel('Max group divergence', fontsize=11)
ax_bot.set_ylim(0, 1.1)
ax_bot.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

# Add a smoothed trend line
from numpy.polynomial.polynomial import polyfit, polyval
coeffs = polyfit(x, divergences, 3)
x_smooth = np.linspace(0, N_CAT - 1, 100)
y_smooth = polyval(x_smooth, coeffs)
ax_bot.plot(x_smooth, y_smooth, color='black', linewidth=2, alpha=0.6, linestyle='-',
            label='Trend (cubic fit)')
ax_bot.legend(fontsize=10)

# Phase shading on bottom too
for start, end, label in phases:
    ax_bot.axvspan(start - 0.5, end - 0.5, alpha=0.06, color='gray')

ax_bot.set_xticks(x)
ax_bot.set_xticklabels([c.replace('Weiterverarbeitung der Mischung von Spiritus und Sal volatile',
                                   'Weiterverarb. Spiritus+Sal vol.')
                         for c in categories],
                        rotation=55, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_figA_flow_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig A: Flow divergence saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE B: Cumulative divergence — at which recipe phase do groups
# lose coherence? Rolling average of disagreement.
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

# For each pair of groups, compute rolling presence agreement
from itertools import combinations
group_pairs = [('I', 'II'), ('I', 'III'), ('II', 'III')]
pair_colors = {'I-II': '#8e44ad', 'I-III': '#e67e22', 'II-III': '#16a085'}

window = 3  # rolling window

for g1, g2 in group_pairs:
    pair_key = f"{g1}-{g2}"
    g1_texts = [t for t in texts if t['new_group'] == g1]
    g2_texts = [t for t in texts if t['new_group'] == g2]

    step_diffs = []
    for c in categories:
        r1 = sum(1 for t in g1_texts if t['annotations'][c]['present']) / len(g1_texts)
        r2 = sum(1 for t in g2_texts if t['annotations'][c]['present']) / len(g2_texts)
        step_diffs.append(abs(r1 - r2))

    # Rolling average
    rolling = np.convolve(step_diffs, np.ones(window)/window, mode='valid')
    x_roll = np.arange(window//2, window//2 + len(rolling))

    ax.plot(x_roll, rolling, color=pair_colors[pair_key], linewidth=2.5, alpha=0.85,
            label=f'Gruppe {g1} vs {g2}')
    # Also plot raw as faint dots
    ax.scatter(range(N_CAT), step_diffs, color=pair_colors[pair_key], alpha=0.25, s=20, zorder=1)

# Phase shading
for start, end, label in phases:
    ax.axvspan(start - 0.5, end - 0.5, alpha=0.06, color='gray')
    mid = (start + end) / 2 - 0.5
    ax.text(mid, 0.95, label, ha='center', va='top', fontsize=8, fontstyle='italic', color='#555')

ax.set_xticks(range(N_CAT))
ax.set_xticklabels([c.replace('Weiterverarbeitung der Mischung von Spiritus und Sal volatile',
                               'Weiterverarb. Spiritus+Sal vol.')
                     for c in categories],
                    rotation=55, ha='right', fontsize=8)
ax.set_ylabel('Presence disagreement (rolling avg, window=3)', fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title('Pairwise Group Divergence across Recipe Flow',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_figB_pairwise_divergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig B: Pairwise divergence saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE C: Per-text "recipe coverage" — where does each text stop
# or have gaps? Shows which texts are truncated vs complete.
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9))

# Sort texts: by group, then by completeness descending
def sort_key_c(t):
    grp_order = {'I': 0, 'II': 1, 'III': 2}
    present = sum(1 for c in categories if t['annotations'][c]['present'])
    return (grp_order.get(t['new_group'], 9), -present)
texts_c = sorted(texts, key=sort_key_c)

# Build matrix: 1=present, 0=absent, with color by group
cmap_i = LinearSegmentedColormap.from_list('gi', ['#fce4e4', '#c0392b'])
cmap_ii = LinearSegmentedColormap.from_list('gii', ['#dbeafe', '#2471a3'])
cmap_iii = LinearSegmentedColormap.from_list('giii', ['#d5f5e3', '#1e8449'])
group_cmaps = {'I': cmap_i, 'II': cmap_ii, 'III': cmap_iii}

for yi, t in enumerate(texts_c):
    for xi, c in enumerate(categories):
        present = t['annotations'][c]['present']
        if present:
            color = GROUP_COLORS[t['new_group']]
            ax.add_patch(plt.Rectangle((xi - 0.4, yi - 0.4), 0.8, 0.8,
                                        facecolor=color, edgecolor='white',
                                        linewidth=0.5, alpha=0.85))
        else:
            ax.add_patch(plt.Rectangle((xi - 0.4, yi - 0.4), 0.8, 0.8,
                                        facecolor='#f5f5f5', edgecolor='#e0e0e0',
                                        linewidth=0.3))

ax.set_xlim(-0.5, N_CAT - 0.5)
ax.set_ylim(-0.5, len(texts_c) - 0.5)
ax.set_xticks(range(N_CAT))
ax.set_xticklabels([c.replace('Weiterverarbeitung der Mischung von Spiritus und Sal volatile',
                               'Weiterverarb. Spiritus+Sal vol.')
                     for c in categories],
                    rotation=55, ha='right', fontsize=8)
ax.set_yticks(range(len(texts_c)))
ylabels = [text_label_full(t) for t in texts_c]
ax.set_yticklabels(ylabels, fontsize=9)
for i, t in enumerate(texts_c):
    ax.get_yticklabels()[i].set_color(GROUP_COLORS[t['new_group']])
    ax.get_yticklabels()[i].set_fontweight('bold')

ax.invert_yaxis()

# Phase lines
for start, end, label in phases:
    ax.axvline(x=start - 0.5, color='gray', linewidth=0.5, alpha=0.4)

# Group separators
prev_group = None
for i, t in enumerate(texts_c):
    if prev_group and t['new_group'] != prev_group:
        ax.axhline(y=i - 0.5, color='black', linewidth=2)
    prev_group = t['new_group']

# Mark "last present step" for each text with an arrow
for yi, t in enumerate(texts_c):
    last_present = -1
    for xi, c in enumerate(categories):
        if t['annotations'][c]['present']:
            last_present = xi
    if last_present < N_CAT - 1:
        ax.annotate('', xy=(last_present + 0.6, yi), xytext=(last_present + 1.2, yi),
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5, alpha=0.6))

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in ['I', 'II', 'III']]
legend_patches.append(mpatches.Patch(color='#f5f5f5', edgecolor='#ccc', label='Absent'))
arr_legend = mlines.Line2D([], [], color='red', marker='>', linestyle='-', alpha=0.6,
                           label='Last attested step')
legend_patches.append(arr_legend)
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.95)

ax.set_title('Recipe Coverage per Text — Where Do Texts End or Have Gaps?',
             fontsize=14, fontweight='bold')
# Phase labels at top
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([(s + e) / 2 - 0.5 for s, e, _ in phases])
ax2.set_xticklabels([l.replace('\n', ' ') for _, _, l in phases], fontsize=9,
                     fontstyle='italic', color='#555')

plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_figC_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig C: Coverage saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE D: Text reuse / dependency network
# Based on shared annotation *values* (not just presence).
# Weighted edges show how much specific procedural detail is shared.
# Approximate dates shown on a timeline axis where available.
# ═══════════════════════════════════════════════════════════════════

# Compute value-level Jaccard for all pairs
def value_sets(t):
    s = set()
    for c in categories:
        for v in t['annotations'][c]['values']:
            s.add((c, v))
    return s

all_value_sets = {t['e_name']: value_sets(t) for t in texts}

pair_sims = {}
for i, t1 in enumerate(texts):
    for j, t2 in enumerate(texts):
        if i >= j:
            continue
        s1 = all_value_sets[t1['e_name']]
        s2 = all_value_sets[t2['e_name']]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        sim = inter / union if union else 0
        pair_sims[(t1['e_name'], t2['e_name'])] = sim

# Build a minimum spanning tree-like structure: for each text, find its
# nearest neighbor (highest similarity). This approximates a stemma.
nearest = {}
for t in texts:
    best_sim = -1
    best_partner = None
    for t2 in texts:
        if t2['e_name'] == t['e_name']:
            continue
        key = tuple(sorted([t['e_name'], t2['e_name']]))
        sim = pair_sims.get(key, 0)
        if sim > best_sim:
            best_sim = sim
            best_partner = t2
    nearest[t['e_name']] = (best_partner['e_name'], best_sim)

fig, ax = plt.subplots(figsize=(20, 12))

# Layout: Y = group lanes, X = date where known, spread out otherwise
# Assign x positions
dated_texts = [t for t in texts if t['date']]
undated_texts = [t for t in texts if not t['date']]

# X positions based on date or interpolated
x_pos = {}
y_pos = {}
group_y = {'I': 4, 'II': 2.5, 'III': 1}

# For dated texts, use actual year
for t in dated_texts:
    x_pos[t['e_name']] = int(t['date'])

# For undated texts, place near their nearest dated neighbor
for t in undated_texts:
    partner, sim = nearest[t['e_name']]
    if partner in x_pos:
        # Offset slightly to avoid overlap
        x_pos[t['e_name']] = x_pos[partner]
    else:
        # Find any high-sim partner that has a date
        best_dated = None
        best_sim = -1
        for t2 in dated_texts:
            key = tuple(sorted([t['e_name'], t2['e_name']]))
            s = pair_sims.get(key, 0)
            if s > best_sim:
                best_sim = s
                best_dated = t2
        if best_dated:
            x_pos[t['e_name']] = int(best_dated['date'])
        else:
            x_pos[t['e_name']] = 1680  # fallback

# Jitter undated texts slightly to avoid overlap
group_x_counts = defaultdict(list)
for t in texts:
    group_x_counts[(t['new_group'], x_pos[t['e_name']])].append(t['e_name'])

for key, names in group_x_counts.items():
    if len(names) > 1:
        spread = 15
        offsets = np.linspace(-spread * (len(names)-1)/2, spread * (len(names)-1)/2, len(names))
        for name, offset in zip(names, offsets):
            x_pos[name] += offset

for t in texts:
    y_pos[t['e_name']] = group_y[t['new_group']]

# Draw edges (nearest-neighbor connections)
# Use thickness/opacity proportional to similarity
drawn_edges = set()
for t in texts:
    partner, sim = nearest[t['e_name']]
    edge_key = tuple(sorted([t['e_name'], partner]))
    if edge_key in drawn_edges:
        continue
    drawn_edges.add(edge_key)

    x1, y1 = x_pos[t['e_name']], y_pos[t['e_name']]
    x2, y2 = x_pos[partner], y_pos[partner]
    lw = sim * 4
    alpha = max(0.2, sim * 0.8)
    ax.plot([x1, x2], [y1, y2], color='#888', linewidth=lw, alpha=alpha, zorder=1)
    # Label similarity on edge midpoint
    mx, my = (x1+x2)/2, (y1+y2)/2
    if sim > 0.6:
        ax.text(mx, my, f'{sim:.2f}', fontsize=7, ha='center', va='center',
                color='#666', alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))

# Also draw high-similarity edges that aren't nearest-neighbor
for (n1, n2), sim in pair_sims.items():
    edge_key = tuple(sorted([n1, n2]))
    if sim > 0.7 and edge_key not in drawn_edges:
        drawn_edges.add(edge_key)
        x1, y1 = x_pos[n1], y_pos[n1]
        x2, y2 = x_pos[n2], y_pos[n2]
        ax.plot([x1, x2], [y1, y2], color='#bbb', linewidth=sim * 3,
                alpha=0.3, linestyle='--', zorder=0)

# Draw nodes
for t in texts:
    x, y = x_pos[t['e_name']], y_pos[t['e_name']]
    present = sum(1 for c in categories if t['annotations'][c]['present'])
    size = 150 + present * 20  # larger = more complete
    is_dated = t['date'] is not None
    marker = 'o' if is_dated else 's'  # circle=dated, square=undated
    edgecolor = 'black' if is_dated else 'gray'
    ax.scatter(x, y, s=size, c=GROUP_COLORS[t['new_group']], marker=marker,
               edgecolors=edgecolor, linewidths=1.5 if is_dated else 1, zorder=5)
    # Labels collected below for adjustText

# Add labels with manual staggering to avoid overlaps
# Sort texts by x position within each group, alternate above/below
from itertools import groupby
texts_by_group = defaultdict(list)
for t in texts:
    texts_by_group[t['new_group']].append(t)

for g, g_texts in texts_by_group.items():
    g_texts.sort(key=lambda t: x_pos[t['e_name']])
    # Detect clusters of close-together nodes (within 20 x-units)
    # and stagger their labels at different y-offsets
    offsets = [0.28, -0.28, 0.50, -0.50, 0.72, -0.72]
    placed = []  # (x_center, y_label, label_width_approx) of placed labels
    for t in g_texts:
        x = x_pos[t['e_name']]
        y = y_pos[t['e_name']]
        label = text_label_full(t)
        label_half_width = len(label) * 1.8  # approx half-width in x-units
        best_off = offsets[0]
        best_min_dist = -1
        for off in offsets:
            y_cand = y + off
            min_dist = float('inf')
            for px, py, pw in placed:
                # Check if labels would horizontally overlap
                x_overlap = max(0, (label_half_width + pw) - abs(x - px))
                if x_overlap > 0:
                    # They overlap horizontally, so vertical separation matters
                    dy = abs(y_cand - py)
                    min_dist = min(min_dist, dy)
                # If no horizontal overlap, they can't collide
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_off = off
        y_label = y + best_off
        placed.append((x, y_label, label_half_width))
        va = 'bottom' if best_off > 0 else 'top'
        ax.annotate(label, xy=(x, y), xytext=(x, y_label),
                    fontsize=7, ha='center', va=va,
                    fontweight='bold', color=GROUP_COLORS[g],
                    arrowprops=dict(arrowstyle='-', color='#ccc', lw=0.5, shrinkA=0, shrinkB=3))

# Group lane labels
for g, y in group_y.items():
    ax.text(1607, y, f'Gruppe {g}', fontsize=12, fontweight='bold',
            color=GROUP_COLORS[g], va='center', ha='right')

# Axis formatting
ax.set_xlim(1598, 1770)
ax.set_ylim(0.0, 5.2)
ax.set_xlabel('Approximate Date (where known)', fontsize=12)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Legend
legend_elements = [
    mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10,
                  markeredgecolor='black', label='Dated text'),
    mlines.Line2D([], [], color='gray', marker='s', linestyle='None', markersize=10,
                  markeredgecolor='gray', label='Undated text (estimated position)'),
    mlines.Line2D([], [], color='#888', linewidth=2, label='Nearest-neighbor link'),
    mlines.Line2D([], [], color='#bbb', linewidth=2, linestyle='--', label='High similarity (>0.7)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
ax.set_title('Text Relationship Network on Approximate Timeline\n(node size ∝ completeness, edge weight ∝ annotation similarity)',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_figD_network_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig D: Network timeline saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE E: "Shared vs unique" annotation values by recipe phase
# For each phase, how many annotation values are shared by all groups,
# shared by two, or unique to one group?
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(phases), figsize=(18, 6), sharey=True)

phase_cats = []
for start, end, label in phases:
    phase_cats.append((label, categories[start:end]))

for ax, (phase_label, cats) in zip(axes, phase_cats):
    # Collect all values per group for this phase
    group_vals = defaultdict(set)
    for t in texts:
        for c in cats:
            if t['annotations'][c]['present']:
                for v in t['annotations'][c]['values']:
                    group_vals[t['new_group']].add((c, v))

    g1 = group_vals.get('I', set())
    g2 = group_vals.get('II', set())
    g3 = group_vals.get('III', set())
    all_vals = g1 | g2 | g3

    # Shared by all three
    shared_all = g1 & g2 & g3
    # Shared by exactly two
    shared_12 = (g1 & g2) - g3
    shared_13 = (g1 & g3) - g2
    shared_23 = (g2 & g3) - g1
    # Unique to one
    only_1 = g1 - g2 - g3
    only_2 = g2 - g1 - g3
    only_3 = g3 - g1 - g2

    counts = [len(shared_all), len(shared_12), len(shared_13), len(shared_23),
              len(only_1), len(only_2), len(only_3)]
    labels_bar = ['All 3', 'I∩II', 'I∩III', 'II∩III', 'Only I', 'Only II', 'Only III']
    colors_bar = ['#95a5a6', '#8e44ad', '#e67e22', '#16a085',
                  GROUP_COLORS['I'], GROUP_COLORS['II'], GROUP_COLORS['III']]

    bars = ax.bar(range(len(counts)), counts, color=colors_bar, alpha=0.85)
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels(labels_bar, rotation=45, ha='right', fontsize=8)
    ax.set_title(phase_label.replace('\n', ' '), fontsize=10, fontweight='bold')

    total = len(all_vals) if all_vals else 1
    shared_pct = len(shared_all) / total * 100
    ax.text(0.5, 0.95, f'{len(all_vals)} values\n{shared_pct:.0f}% shared by all',
            transform=ax.transAxes, ha='center', va='top', fontsize=8, color='#555')

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=8)

axes[0].set_ylabel('Number of distinct annotation values', fontsize=10)
fig.suptitle('Shared vs Group-Unique Annotation Values by Recipe Phase',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_figE_shared_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig E: Shared values saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE F: Text reuse heatmap — n-gram overlap between plain texts
# ═══════════════════════════════════════════════════════════════════
import xml.etree.ElementTree as ET

tree = ET.parse('/Users/slang/claude/processus-sammlung_aller_texte.xml')
root = tree.getroot()

# Extract plain text per div
plain_texts = {}
for div in root.findall('.//div'):
    text_id = div.get('type', '')
    # Find the matching text entry
    match = [t for t in texts if t['text_id'] == text_id]
    if not match:
        continue
    t = match[0]
    raw = ''.join(div.itertext()).strip()
    raw = re.sub(r'\s+', ' ', raw).lower()
    # Remove common stop-like words to focus on content
    plain_texts[t['e_name']] = raw

def ngram_set(text, n=4):
    words = text.split()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

# Compute word 4-gram overlap (Jaccard on word n-grams)
ngram_sets = {name: ngram_set(txt, 4) for name, txt in plain_texts.items()}

texts_for_ngram = [t for t in texts if t['e_name'] in ngram_sets]
texts_for_ngram.sort(key=lambda t: ({'I':0,'II':1,'III':2}.get(t['new_group'],9),
                                     int(''.join(c for c in t['e_name'] if c.isdigit()) or '0')))
text_names_ordered = [t['e_name'] for t in texts_for_ngram]

n_texts = len(text_names_ordered)
ngram_sim = np.zeros((n_texts, n_texts))
for i, n1 in enumerate(text_names_ordered):
    for j, n2 in enumerate(text_names_ordered):
        if i == j:
            ngram_sim[i, j] = 1.0
        else:
            s1, s2 = ngram_sets[n1], ngram_sets[n2]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            ngram_sim[i, j] = inter / union if union else 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Left: text reuse (n-gram)
im1 = ax1.imshow(ngram_sim, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='equal')
ax1.set_xticks(range(n_texts))
full_labels = [f"{name} ({[t for t in texts if t['e_name']==name][0]['a_name']})"
               for name in text_names_ordered]
ax1.set_xticklabels(full_labels, rotation=45, ha='right', fontsize=9)
ax1.set_yticks(range(n_texts))
ax1.set_yticklabels(full_labels, fontsize=9)
for i, name in enumerate(text_names_ordered):
    grp = [t for t in texts if t['e_name'] == name][0]['new_group']
    ax1.get_xticklabels()[i].set_color(GROUP_COLORS[grp])
    ax1.get_yticklabels()[i].set_color(GROUP_COLORS[grp])
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('4-gram Jaccard overlap', fontsize=10)
ax1.set_title('Text Reuse (word 4-gram overlap)', fontsize=13, fontweight='bold')

# Right: annotation similarity (for comparison)
ann_sim = np.zeros((n_texts, n_texts))
for i, n1 in enumerate(text_names_ordered):
    for j, n2 in enumerate(text_names_ordered):
        if i == j:
            ann_sim[i, j] = 1.0
        else:
            key = tuple(sorted([n1, n2]))
            ann_sim[i, j] = pair_sims.get(key, 0)

im2 = ax2.imshow(ann_sim, cmap='YlGnBu', vmin=0, vmax=1, aspect='equal')
ax2.set_xticks(range(n_texts))
ax2.set_xticklabels(full_labels, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(n_texts))
ax2.set_yticklabels(full_labels, fontsize=9)
for i, name in enumerate(text_names_ordered):
    grp = [t for t in texts if t['e_name'] == name][0]['new_group']
    ax2.get_xticklabels()[i].set_color(GROUP_COLORS[grp])
    ax2.get_yticklabels()[i].set_color(GROUP_COLORS[grp])
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Annotation value Jaccard', fontsize=10)
ax2.set_title('Annotation Similarity (expert key-values)', fontsize=13, fontweight='bold')

fig.suptitle('Text Reuse vs Annotation Similarity — Do Shared Words Mean Shared Chemistry?',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_figF_text_reuse.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig F: Text reuse saved")

print("\nAll evolution figures saved.")
