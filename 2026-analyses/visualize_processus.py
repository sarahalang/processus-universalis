"""
Visualizations for the Processus Universalis annotated corpus.
Uses the current (E-name / Gruppe I-III) nomenclature throughout.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

# ── Load data ──
with open('/Users/slang/claude/processus_data.json', 'r') as f:
    data = json.load(f)

texts = data['texts']
categories = data['categories']

# Group colors (new nomenclature)
GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}
GROUP_LABELS = {'I': 'Gruppe I (old G3)', 'II': 'Gruppe II (old G1)', 'III': 'Gruppe III (old G2)'}

def text_label(t):
    return f"{t['e_name']} ({t['a_name']})"

def text_label_short(t):
    return t['e_name']

# Sort texts by new group then e_name
def sort_key(t):
    grp_order = {'I': 0, 'II': 1, 'III': 2}
    e_num = int(''.join(c for c in t['e_name'] if c.isdigit()) or '0')
    return (grp_order.get(t['new_group'], 9), e_num)

texts_sorted = sorted(texts, key=sort_key)

# Build presence matrix
matrix = np.array([
    [1 if t['annotations'][c]['present'] else 0 for c in categories]
    for t in texts_sorted
])

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Presence/Absence Heatmap (grouped by Gruppe)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 10))

# Custom colormap: light gray for absent, dark teal for present
cmap = ListedColormap(['#f0f0f0', '#2c7fb8'])
im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')

# Y-axis: text labels with group coloring
y_labels = [text_label(t) for t in texts_sorted]
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=10)
for i, t in enumerate(texts_sorted):
    ax.get_yticklabels()[i].set_color(GROUP_COLORS[t['new_group']])
    ax.get_yticklabels()[i].set_fontweight('bold')

# X-axis: category labels
short_cats = [c.replace('Weiterverarbeitung der Mischung von Spiritus und Sal volatile',
                         'Weiterverarb. Spiritus+Sal vol.')
              for c in categories]
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(short_cats, rotation=55, ha='right', fontsize=8)

# Group separators
group_boundaries = []
prev_group = None
for i, t in enumerate(texts_sorted):
    if prev_group and t['new_group'] != prev_group:
        group_boundaries.append(i - 0.5)
    prev_group = t['new_group']
for b in group_boundaries:
    ax.axhline(y=b, color='black', linewidth=2)

# Legend
legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
                  for g in ['I', 'II', 'III']]
legend_patches += [mpatches.Patch(color='#2c7fb8', label='Present'),
                   mpatches.Patch(color='#f0f0f0', label='Absent (FEHLT)')]
ax.legend(handles=legend_patches, loc='upper right', fontsize=9, framealpha=0.9)

ax.set_title('Processus Universalis — Process Step Presence across Texts', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_fig1_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1: Heatmap saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Category presence bar chart (sorted, colored by group breakdown)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 9))

# Count presence per category per group
cat_data = []
for c in categories:
    counts = {}
    for g in ['I', 'II', 'III']:
        g_texts = [t for t in texts if t['new_group'] == g]
        counts[g] = sum(1 for t in g_texts if t['annotations'][c]['present'])
    cat_data.append((c, counts, sum(counts.values())))

cat_data.sort(key=lambda x: -x[2])

y_pos = range(len(cat_data))
left = np.zeros(len(cat_data))
for g in ['I', 'II', 'III']:
    widths = [cd[1][g] for cd in cat_data]
    ax.barh(y_pos, widths, left=left, color=GROUP_COLORS[g], label=GROUP_LABELS[g], height=0.7)
    left += widths

ax.set_yticks(y_pos)
ax.set_yticklabels([cd[0] for cd in cat_data], fontsize=9)
ax.set_xlabel('Number of texts containing this step', fontsize=11)
ax.set_title('Process Step Frequency (by Gruppe)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 19)
ax.invert_yaxis()

# Add count labels
for i, cd in enumerate(cat_data):
    ax.text(cd[2] + 0.3, i, str(cd[2]), va='center', fontsize=9, color='#333')

plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_fig2_category_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2: Category bars saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Hierarchical clustering dendrogram (value-level Jaccard)
# ═══════════════════════════════════════════════════════════════════
def value_jaccard_dist(t1, t2):
    s1 = set()
    s2 = set()
    for c in categories:
        for v in t1['annotations'][c]['values']:
            s1.add((c, v))
        for v in t2['annotations'][c]['values']:
            s2.add((c, v))
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return 1 - (inter / union) if union else 1

n = len(texts_sorted)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = value_jaccard_dist(texts_sorted[i], texts_sorted[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

condensed = squareform(dist_matrix)
Z = linkage(condensed, method='ward')

fig, ax = plt.subplots(figsize=(14, 7))
labels = [text_label(t) for t in texts_sorted]
label_colors = {text_label(t): GROUP_COLORS[t['new_group']] for t in texts_sorted}

dend = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=10,
                  color_threshold=0)

# Color the leaf labels by group
xlbls = ax.get_xticklabels()
for lbl in xlbls:
    lbl.set_color(label_colors.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
                  for g in ['I', 'II', 'III']]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
ax.set_ylabel('Distance (Ward linkage, value-level Jaccard)', fontsize=11)
ax.set_title('Hierarchical Clustering of Texts by Annotation Similarity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_fig3_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3: Dendrogram saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Similarity heatmap (value-level Jaccard)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))

# Reorder by dendrogram leaf order
leaf_order = dend['leaves']
ordered_texts = [texts_sorted[i] for i in leaf_order]
ordered_labels = [text_label(t) for t in ordered_texts]

# Rebuild similarity matrix in dendrogram order
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            sim_matrix[i, j] = 1.0
        else:
            oi, oj = leaf_order[i], leaf_order[j]
            sim_matrix[i, j] = 1 - dist_matrix[oi, oj]

im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
ax.set_xticks(range(n))
ax.set_xticklabels(ordered_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(n))
ax.set_yticklabels(ordered_labels, fontsize=9)

for i, t in enumerate(ordered_texts):
    ax.get_xticklabels()[i].set_color(GROUP_COLORS[t['new_group']])
    ax.get_yticklabels()[i].set_color(GROUP_COLORS[t['new_group']])

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Jaccard similarity (annotation values)', fontsize=10)

ax.set_title('Pairwise Text Similarity (Value-Level Jaccard)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_fig4_similarity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4: Similarity heatmap saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Text completeness + word count (grouped)
# ═══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Sort by completeness
completeness = [(t, sum(1 for c in categories if t['annotations'][c]['present']))
                for t in texts]
completeness.sort(key=lambda x: -x[1])

# Left: completeness
y_pos = range(len(completeness))
bars = ax1.barh(y_pos,
                [c[1] for c in completeness],
                color=[GROUP_COLORS[c[0]['new_group']] for c in completeness],
                height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([text_label(c[0]) for c in completeness], fontsize=10)
ax1.set_xlabel('Categories present (out of 30)', fontsize=11)
ax1.set_title('Text Completeness', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 32)
for i, c in enumerate(completeness):
    ax1.text(c[1] + 0.3, i, str(c[1]), va='center', fontsize=9)
ax1.invert_yaxis()

# Right: word count
wc_sorted = sorted(texts, key=lambda t: -t['word_count'])
y_pos2 = range(len(wc_sorted))
ax2.barh(y_pos2,
         [t['word_count'] for t in wc_sorted],
         color=[GROUP_COLORS[t['new_group']] for t in wc_sorted],
         height=0.7)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels([text_label(t) for t in wc_sorted], fontsize=10)
ax2.set_xlabel('Word count', fontsize=11)
ax2.set_title('Text Length', fontsize=13, fontweight='bold')
for i, t in enumerate(wc_sorted):
    ax2.text(t['word_count'] + 30, i, str(t['word_count']), va='center', fontsize=9)
ax2.invert_yaxis()

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
                  for g in ['I', 'II', 'III']]
ax2.legend(handles=legend_patches, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_fig5_completeness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5: Completeness saved")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Group-distinctive categories (diverging bar chart)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)

for idx, g in enumerate(['I', 'II', 'III']):
    ax = axes[idx]
    g_texts = [t for t in texts if t['new_group'] == g]
    other_texts = [t for t in texts if t['new_group'] != g]

    diffs = []
    for c in categories:
        g_rate = sum(1 for t in g_texts if t['annotations'][c]['present']) / len(g_texts)
        o_rate = sum(1 for t in other_texts if t['annotations'][c]['present']) / len(other_texts)
        diffs.append((c, g_rate - o_rate, g_rate, o_rate))

    diffs.sort(key=lambda x: x[1])
    cats = [d[0] for d in diffs]
    vals = [d[1] for d in diffs]

    colors = [GROUP_COLORS[g] if v > 0 else '#999999' for v in vals]
    ax.barh(range(len(cats)), vals, color=colors, height=0.7)
    ax.set_yticks(range(len(cats)))
    if idx == 0:
        ax.set_yticklabels(cats, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Difference from other groups', fontsize=10)
    ax.set_title(f'Gruppe {g}\n({len(g_texts)} texts)', fontsize=12,
                 fontweight='bold', color=GROUP_COLORS[g])
    ax.set_xlim(-1, 1)

fig.suptitle('Group-Distinctive Process Steps\n(positive = more common in this group than others)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/slang/claude/processus-universalis-graphics/processus_fig6_group_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6: Group profiles saved")

print("\nAll figures saved to /Users/slang/claude/processus-universalis-graphics/processus_fig*.png")
