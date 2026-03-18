"""
Automated Pipeline: Text → Step Segmentation → Network Analysis
Compared with manual annotations and stylometric analysis.

Pipeline stages:
1. Load raw texts
2. Automatic step segmentation using keyword detection
3. Build similarity networks (text-based, step-based, stylometric)
4. Compare all approaches with annotation-based ground truth
5. Visualise where and why they diverge
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
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'processus-universalis-graphics'

# ── Load data ──
with open('/Users/slang/claude/processus_data.json', 'r') as f:
    data = json.load(f)
texts_meta = data['texts']
categories = data['categories']
meta_by_name = {t['e_name']: t for t in texts_meta}

GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}
GROUP_NAMES = {'I': 'Gruppe I', 'II': 'Gruppe II', 'III': 'Gruppe III'}

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

common_names = sorted(set(meta_by_name.keys()) & set(plain_texts.keys()))
n = len(common_names)
print(f"Pipeline: {n} texts loaded")


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: AUTOMATIC STEP SEGMENTATION
# ═══════════════════════════════════════════════════════════════════
# These keyword patterns are derived from the 30 annotation category
# names and common Early Modern German recipe vocabulary.
# The pipeline detects which "steps" a text covers based on keyword
# presence, producing a binary feature vector comparable to the
# manual presence/absence annotations.

STEP_KEYWORDS = {
    'preface_attribution': [
        'sendivog', 'becher', 'beuther', 'sethon', 'adept',
        'processus', 'vorschrift', 'geheimnuß', 'secretum',
    ],
    'philosophical_intro': [
        'philosophi', 'centrum', 'natur', 'element',
        'subjectum', 'principi', 'fundament', 'geheimn',
    ],
    'earth_sampling_time': [
        'martio', 'aries', 'ariete', 'frühling',
        'april', 'may', 'himmel', 'sonn', 'mond', 'stern',
        'morgen',
    ],
    'earth_sampling_place': [
        'wiese', 'feld', 'acker', 'garten', 'wald',
        'berg', 'böhei', 'böhm',
    ],
    'earth_type': [
        'fette', 'schwartz', 'jungfräu', 'virgin',
        'thon', 'lehm', 'bolus', 'erde', 'erdt',
    ],
    'earth_sampling_method': [
        'grab', 'grabe', 'steche', 'stich', 'schaufel',
        'spaten', 'nimm', 'nehme',
    ],
    'earth_division': [
        'theile', 'zwey', 'zwei', 'gleiche', 'theil',
        'divide', 'partes', 'aequales',
    ],
    'earth_location': [
        'fundort', 'ort', 'gegen', 'land',
    ],
    'earth_impregnation': [
        'imprägnir', 'impraegnir', 'impregnir',
        'geschwänger', 'geschwaenger', 'penetr',
        'gestirn', 'magnet',
    ],
    'extraction': [
        'extrah', 'extrac', 'auslaug', 'auszieh',
        'sied', 'koch', 'wasser', 'faß', 'zapf',
        'filtr', 'laug',
    ],
    'evaporation': [
        'eindampf', 'evapor', 'eingesott',
        'abdampf', 'einsied', 'kristall', 'schieß',
        'kessel', 'pfann',
    ],
    'salt_naming': [
        'sal nitri', 'salpeter', 'nitrum', 'nitri',
        'sal terrae', 'salz der erde',
        'hauptschlüssel', 'secretum',
    ],
    'earth_salt_processing': [
        'calcinir', 'calcin', 'glüh', 'durchglüh',
        'kugel', 'retort', 'destillir', 'destill',
        'spiritus', 'gradus',
    ],
    'spiritus_sal_volatile_processing': [
        'sal volatile', 'flüchtig', 'sublim',
        'übersublim', 'rektifiz', 'rectific',
        'phlegma', 'wasserbad', 'sandbad', 'aschebad',
    ],
    'sal_volatile_extraction': [
        'sal volatile', 'flüchtig salz',
        'extra gewin',
    ],
    'sal_fixum_extraction': [
        'sal fixum', 'fix salz', 'fixes salz',
        'regenwasser', 'coagul', 'auskristall',
    ],
    'wet_dry_path': [
        'nasser weg', 'trockener weg', 'nassen weg',
        'trockenen weg', 'via humida', 'via sicca',
    ],
    'two_principles_joining': [
        'zwey principi', 'zwei principi',
        'zusammenfüg',
    ],
    'three_principles_joining': [
        'drey principi', 'drei principi',
        'zusammen setz', 'zusammensetz',
        'sal volatile.*sal fixum.*spiritus',
        'menstru', 'universal',
    ],
    'solvent_naming': [
        'menstruum', 'lösungsmittel', 'menstru',
        'aurum potabile', 'aqua', 'schlüssel',
    ],
    'gold_silver_melting': [
        'zusammenschmelz', 'gold.*silber.*schmelz',
        'silber.*gold.*schmelz',
    ],
    'gold_dissolution': [
        'gold', 'auflös', 'digest', 'phiole',
        'sigill', 'hermetice', 'gefeil',
        'geschlagen',
    ],
    'athanor_description': [
        'athanor', 'ofen', 'ofens',
    ],
    'ruby_grain_production': [
        'rubinkorn', 'rubin', 'roth', 'gelb',
        'schwartz', 'weiß', 'farb',
        'sandbad', 'aschebad',
    ],
    'ruby_grain_naming': [
        'rubinkorn', 'stein der weisen',
        'lapis', 'quintessenz', 'tinctur',
    ],
    'further_synthesis': [
        'weiter', 'synthesis', 'synthetis',
    ],
    'intermediate_analysis': [
        'prob', 'prüf', 'test', 'analyse',
        'versuch', 'nagel',
    ],
    'multiplication': [
        'multiplic', 'vermehr', 'augment',
    ],
    'fermentation': [
        'ferment', 'gähr', 'verdau',
    ],
    'projection': [
        'project', 'tingir', 'tingier',
        'transmut', 'verwandl', 'tropf',
        'bley', 'kupfer', 'zinn',
    ],
}

# Map auto-detected steps to the manual category indices
STEP_TO_CATEGORY_IDX = {
    'preface_attribution': 0,
    'philosophical_intro': 1,
    'earth_sampling_time': 2,
    'earth_sampling_place': 3,
    'earth_type': 4,
    'earth_sampling_method': 5,
    'earth_division': 6,
    'earth_location': 7,
    'earth_impregnation': 8,
    'extraction': 9,
    'evaporation': 10,
    'salt_naming': 11,
    'earth_salt_processing': 12,
    'spiritus_sal_volatile_processing': 13,
    'sal_volatile_extraction': 14,
    'sal_fixum_extraction': 15,
    'wet_dry_path': 16,
    'two_principles_joining': 17,
    'three_principles_joining': 18,
    'solvent_naming': 19,
    'gold_silver_melting': 20,
    'gold_dissolution': 21,
    'athanor_description': 22,
    'ruby_grain_production': 23,
    'ruby_grain_naming': 24,
    'further_synthesis': 25,
    'intermediate_analysis': 26,
    'multiplication': 27,
    'fermentation': 28,
    'projection': 29,
}

def detect_steps(text):
    """Detect which recipe steps are present in a text using keyword matching."""
    text_lower = text.lower()
    detected = {}
    for step_name, keywords in STEP_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if '.*' in kw:
                if re.search(kw, text_lower):
                    score += 1
            else:
                score += text_lower.count(kw)
        # Threshold: at least 2 keyword hits to count as "present"
        detected[step_name] = score >= 2
    return detected


def segment_text(text, window_size=100):
    """
    Segment text into regions associated with each step.
    Uses a sliding window to find which parts of the text
    correspond to which recipe steps.
    Returns: dict of step_name -> list of (start_word_idx, end_word_idx) spans
    """
    words = text.lower().split()
    n_words = len(words)
    step_spans = defaultdict(list)

    for step_name, keywords in STEP_KEYWORDS.items():
        for start in range(0, n_words - window_size + 1, window_size // 2):
            end = min(start + window_size, n_words)
            window_text = ' '.join(words[start:end])
            score = 0
            for kw in keywords:
                if '.*' in kw:
                    if re.search(kw, window_text):
                        score += 1
                else:
                    score += window_text.count(kw)
            if score >= 2:
                step_spans[step_name].append((start, end))

    return step_spans


# Detect steps for all texts
print("\n── Stage 1: Automatic Step Detection ──")
auto_steps = {}
auto_segments = {}
for name in common_names:
    auto_steps[name] = detect_steps(plain_texts[name])
    auto_segments[name] = segment_text(plain_texts[name])

# Build auto-detected presence matrix (comparable to manual)
auto_matrix = np.zeros((n, 30))
for i, name in enumerate(common_names):
    for step_name, present in auto_steps[name].items():
        cat_idx = STEP_TO_CATEGORY_IDX[step_name]
        auto_matrix[i, cat_idx] = 1 if present else 0

# Build manual presence matrix for comparison
manual_matrix = np.zeros((n, 30))
for i, name in enumerate(common_names):
    t = meta_by_name[name]
    for j, cat in enumerate(categories):
        manual_matrix[i, j] = 1 if t['annotations'][cat]['present'] else 0

# Compare auto vs manual detection
agreement = (auto_matrix == manual_matrix).mean()
per_cat_agreement = (auto_matrix == manual_matrix).mean(axis=0)
per_text_agreement = (auto_matrix == manual_matrix).mean(axis=1)
print(f"  Overall agreement: {agreement:.1%}")
print(f"  Per-category agreement range: {per_cat_agreement.min():.1%} - {per_cat_agreement.max():.1%}")
print(f"  Per-text agreement range: {per_text_agreement.min():.1%} - {per_text_agreement.max():.1%}")

# Confusion matrix
tp = ((auto_matrix == 1) & (manual_matrix == 1)).sum()
fp = ((auto_matrix == 1) & (manual_matrix == 0)).sum()
fn = ((auto_matrix == 0) & (manual_matrix == 1)).sum()
tn = ((auto_matrix == 0) & (manual_matrix == 0)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: STYLOMETRIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════
# Implements Burrows' Delta and Cosine Delta — the standard
# stylometric distance measures used by tools like Stylo.

print("\n── Stage 2: Stylometric Analysis ──")

def tokenize(text):
    """Simple tokenization for stylometric analysis."""
    return re.findall(r'[a-zäöüß]+', text.lower())

# Build vocabulary from all texts
all_tokens = []
text_tokens = {}
for name in common_names:
    tokens = tokenize(plain_texts[name])
    text_tokens[name] = tokens
    all_tokens.extend(tokens)

# Most Frequent Words (MFW) — the core of stylometry
vocab_counts = Counter(all_tokens)
# Use different MFW sizes for robustness
MFW_SIZES = [100, 200, 500]

def compute_mfw_features(tokens, mfw_list):
    """Compute relative frequencies of MFW in a text."""
    total = len(tokens)
    if total == 0:
        return np.zeros(len(mfw_list))
    counts = Counter(tokens)
    return np.array([counts.get(w, 0) / total for w in mfw_list])

def burrows_delta(features_matrix):
    """
    Burrows' Delta distance.
    1. Z-score normalize each feature across texts
    2. Distance = mean absolute difference of z-scores
    """
    # Z-score normalize columns (features)
    means = features_matrix.mean(axis=0)
    stds = features_matrix.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero
    z_scores = (features_matrix - means) / stds

    n_texts = features_matrix.shape[0]
    dist = np.zeros((n_texts, n_texts))
    for i in range(n_texts):
        for j in range(i+1, n_texts):
            d = np.mean(np.abs(z_scores[i] - z_scores[j]))
            dist[i, j] = dist[j, i] = d
    return dist

def cosine_delta(features_matrix):
    """
    Cosine Delta (Eders' Delta).
    1. Z-score normalize each feature
    2. Distance = 1 - cosine similarity of z-score vectors
    """
    means = features_matrix.mean(axis=0)
    stds = features_matrix.std(axis=0)
    stds[stds == 0] = 1
    z_scores = (features_matrix - means) / stds

    n_texts = features_matrix.shape[0]
    dist = np.zeros((n_texts, n_texts))
    for i in range(n_texts):
        for j in range(i+1, n_texts):
            dot = np.dot(z_scores[i], z_scores[j])
            norm_i = np.linalg.norm(z_scores[i])
            norm_j = np.linalg.norm(z_scores[j])
            cos_sim = dot / (norm_i * norm_j) if (norm_i * norm_j) > 0 else 0
            dist[i, j] = dist[j, i] = 1 - cos_sim
    return dist

# Compute stylometric distances for multiple MFW sizes
stylo_distances = {}
for mfw_size in MFW_SIZES:
    mfw_list = [w for w, _ in vocab_counts.most_common(mfw_size)]
    features = np.array([compute_mfw_features(text_tokens[name], mfw_list)
                         for name in common_names])

    dist_burrows = burrows_delta(features)
    dist_cosine = cosine_delta(features)
    stylo_distances[f'burrows_{mfw_size}'] = dist_burrows
    stylo_distances[f'cosine_{mfw_size}'] = dist_cosine

    # Convert to similarity for correlation
    sim_burrows = 1 - (dist_burrows / dist_burrows.max())
    sim_cosine = 1 - (dist_cosine / dist_cosine.max())

    print(f"  MFW={mfw_size}: Burrows' Δ range [{dist_burrows[dist_burrows>0].min():.3f}, "
          f"{dist_burrows.max():.3f}], Cosine Δ range [{dist_cosine[dist_cosine>0].min():.3f}, "
          f"{dist_cosine.max():.3f}]")


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: COMPUTE ALL SIMILARITY/DISTANCE MATRICES
# ═══════════════════════════════════════════════════════════════════

print("\n── Stage 3: Building comparison matrices ──")

def jaccard_sim(set1, set2):
    if not set1 and not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0.0

# 1. Manual annotation similarity (ground truth)
def annotation_values(t):
    s = set()
    for c in categories:
        for v in t['annotations'][c]['values']:
            s.add((c, v))
    return s

anno_sets = {name: annotation_values(meta_by_name[name]) for name in common_names}
sim_anno = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        s = jaccard_sim(anno_sets[common_names[i]], anno_sets[common_names[j]])
        sim_anno[i,j] = sim_anno[j,i] = s
np.fill_diagonal(sim_anno, 1.0)

# 2. Auto-detected step similarity (Jaccard on detected step sets)
sim_auto_step = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        steps_i = set(s for s, v in auto_steps[common_names[i]].items() if v)
        steps_j = set(s for s, v in auto_steps[common_names[j]].items() if v)
        s = jaccard_sim(steps_i, steps_j)
        sim_auto_step[i,j] = sim_auto_step[j,i] = s
np.fill_diagonal(sim_auto_step, 1.0)

# 3. Word 4-gram text reuse similarity
def word_ngrams(text, ng=4):
    words = text.lower().split()
    return set(tuple(words[i:i+ng]) for i in range(len(words) - ng + 1))

raw_ngrams = {name: word_ngrams(plain_texts[name]) for name in common_names}
sim_text = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        s = jaccard_sim(raw_ngrams[common_names[i]], raw_ngrams[common_names[j]])
        sim_text[i,j] = sim_text[j,i] = s
np.fill_diagonal(sim_text, 1.0)

# 4. Stylometric similarities (use Cosine Delta with 200 MFW as primary)
primary_stylo_dist = stylo_distances['cosine_200']
sim_stylo = 1 - (primary_stylo_dist / primary_stylo_dist.max())
np.fill_diagonal(sim_stylo, 1.0)

# Also compute Burrows Delta 200 for comparison
burrows_200_dist = stylo_distances['burrows_200']
sim_burrows = 1 - (burrows_200_dist / burrows_200_dist.max())
np.fill_diagonal(sim_burrows, 1.0)

# 5. Annotation presence-level (binary) similarity — closer to auto-detected
sim_anno_presence = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        pres_i = set(c for c in categories if meta_by_name[common_names[i]]['annotations'][c]['present'])
        pres_j = set(c for c in categories if meta_by_name[common_names[j]]['annotations'][c]['present'])
        s = jaccard_sim(pres_i, pres_j)
        sim_anno_presence[i,j] = sim_anno_presence[j,i] = s
np.fill_diagonal(sim_anno_presence, 1.0)

upper = np.triu_indices(n, k=1)

# Compute all pairwise correlations
methods = {
    'Text 4-gram': sim_text,
    'Auto steps': sim_auto_step,
    'Cosine Delta (200 MFW)': sim_stylo,
    "Burrows' Delta (200 MFW)": sim_burrows,
    'Anno values (manual)': sim_anno,
    'Anno presence (manual)': sim_anno_presence,
}

print("\n  Pairwise correlations:")
print(f"  {'Method A':<28} {'Method B':<28} {'Pearson r':<10}")
print("  " + "-" * 70)
method_names = list(methods.keys())
for i_m in range(len(method_names)):
    for j_m in range(i_m+1, len(method_names)):
        m_a, m_b = method_names[i_m], method_names[j_m]
        r, p = pearsonr(methods[m_a][upper], methods[m_b][upper])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {m_a:<28} {m_b:<28} {r:+.3f} {sig}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE R: Step Detection Agreement Heatmap
# Auto-detected vs manually annotated step presence
# ═══════════════════════════════════════════════════════════════════
print("\n── Generating figures ──")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Left: Manual presence matrix
ax = axes[0]
im = ax.imshow(manual_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
ax.set_yticks(range(n))
ax.set_yticklabels(common_names, fontsize=8)
for i, name in enumerate(common_names):
    ax.get_yticklabels()[i].set_color(GROUP_COLORS[get_group(name)])
ax.set_xticks(range(30))
ax.set_xticklabels(range(1, 31), fontsize=7)
ax.set_xlabel('Category index', fontsize=10)
ax.set_title('Manual Annotations\n(expert)', fontsize=12, fontweight='bold')

# Centre: Auto-detected matrix
ax = axes[1]
im = ax.imshow(auto_matrix, cmap='Oranges', aspect='auto', interpolation='nearest')
ax.set_yticks(range(n))
ax.set_yticklabels(common_names, fontsize=8)
for i, name in enumerate(common_names):
    ax.get_yticklabels()[i].set_color(GROUP_COLORS[get_group(name)])
ax.set_xticks(range(30))
ax.set_xticklabels(range(1, 31), fontsize=7)
ax.set_xlabel('Category index', fontsize=10)
ax.set_title('Auto-Detected Steps\n(keyword-based)', fontsize=12, fontweight='bold')

# Right: Agreement/disagreement
# Green = both agree present, blue = both agree absent,
# red = false positive (auto says yes, manual says no),
# orange = false negative (auto says no, manual says yes)
agreement_matrix = np.zeros((n, 30))
from matplotlib.colors import ListedColormap
for i in range(n):
    for j in range(30):
        if auto_matrix[i,j] == 1 and manual_matrix[i,j] == 1:
            agreement_matrix[i,j] = 3  # true positive
        elif auto_matrix[i,j] == 0 and manual_matrix[i,j] == 0:
            agreement_matrix[i,j] = 0  # true negative
        elif auto_matrix[i,j] == 1 and manual_matrix[i,j] == 0:
            agreement_matrix[i,j] = 1  # false positive
        else:
            agreement_matrix[i,j] = 2  # false negative

ax = axes[2]
cmap_agree = ListedColormap(['#f0f0f0', '#e74c3c', '#e67e22', '#2ecc71'])
im = ax.imshow(agreement_matrix, cmap=cmap_agree, aspect='auto',
               interpolation='nearest', vmin=0, vmax=3)
ax.set_yticks(range(n))
ax.set_yticklabels(common_names, fontsize=8)
for i, name in enumerate(common_names):
    ax.get_yticklabels()[i].set_color(GROUP_COLORS[get_group(name)])
ax.set_xticks(range(30))
ax.set_xticklabels(range(1, 31), fontsize=7)
ax.set_xlabel('Category index', fontsize=10)
ax.set_title(f'Agreement\n(overall: {agreement:.1%})', fontsize=12, fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#2ecc71', label=f'True Positive ({tp})'),
    mpatches.Patch(color='#f0f0f0', label=f'True Negative ({tn})'),
    mpatches.Patch(color='#e74c3c', label=f'False Positive ({fp})'),
    mpatches.Patch(color='#e67e22', label=f'False Negative ({fn})'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

fig.suptitle('Automatic Step Detection vs Expert Annotations\n'
             f'Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figR_step_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig R saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE S: Four-way Dendrogram Comparison
# Text reuse, auto steps, stylometric, manual annotations
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

dendro_configs = [
    (axes[0,0], 1 - sim_text,   'Text Reuse\n(word 4-gram Jaccard)', 'text'),
    (axes[0,1], 1 - sim_auto_step, 'Auto-Detected Steps\n(keyword Jaccard)', 'auto'),
    (axes[1,0], primary_stylo_dist, 'Stylometric\n(Cosine Delta, 200 MFW)', 'stylo'),
    (axes[1,1], 1 - sim_anno,  'Expert Annotations\n(value Jaccard)', 'anno'),
]

dendro_results = {}
for ax, dist_mat, title, key in dendro_configs:
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method='ward')
    labels = [f"{name} ({get_group(name)})" for name in common_names]
    label_colors = {f"{name} ({get_group(name)})": GROUP_COLORS[get_group(name)]
                    for name in common_names}

    dend = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
                      color_threshold=0)
    dendro_results[key] = {'Z': Z, 'leaves': dend['leaves'], 'dist': dist_mat}

    for lbl in ax.get_xticklabels():
        lbl.set_color(label_colors.get(lbl.get_text(), 'black'))
        lbl.set_fontweight('bold')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=10)

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Gruppe {g}')
                  for g in ['I', 'II', 'III']]
axes[0,0].legend(handles=legend_patches, loc='upper left', fontsize=9)

fig.suptitle('Four Methods of Clustering: Which Agrees with Expert Annotations?',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figS_four_dendrograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig S saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE T: Correlation matrix of all methods
# ═══════════════════════════════════════════════════════════════════
# Include more variants for a thorough comparison
all_methods = {
    'Text 4-gram': sim_text,
    'Auto steps': sim_auto_step,
    'Anno presence': sim_anno_presence,
    'Cosine Δ 100': 1 - stylo_distances['cosine_100'] / stylo_distances['cosine_100'].max(),
    'Cosine Δ 200': sim_stylo,
    'Cosine Δ 500': 1 - stylo_distances['cosine_500'] / stylo_distances['cosine_500'].max(),
    'Burrows Δ 100': 1 - stylo_distances['burrows_100'] / stylo_distances['burrows_100'].max(),
    'Burrows Δ 200': sim_burrows,
    'Burrows Δ 500': 1 - stylo_distances['burrows_500'] / stylo_distances['burrows_500'].max(),
    'Anno values': sim_anno,
}

method_keys = list(all_methods.keys())
n_methods = len(method_keys)
corr_matrix = np.zeros((n_methods, n_methods))
for i_m in range(n_methods):
    for j_m in range(n_methods):
        if i_m == j_m:
            corr_matrix[i_m, j_m] = 1.0
        else:
            r, _ = pearsonr(all_methods[method_keys[i_m]][upper],
                            all_methods[method_keys[j_m]][upper])
            corr_matrix[i_m, j_m] = r

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
ax.set_xticks(range(n_methods))
ax.set_xticklabels(method_keys, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(n_methods))
ax.set_yticklabels(method_keys, fontsize=10)

# Annotate cells with r values
for i_m in range(n_methods):
    for j_m in range(n_methods):
        val = corr_matrix[i_m, j_m]
        color = 'white' if val < 0.4 or val > 0.85 else 'black'
        ax.text(j_m, i_m, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Pearson r', fontsize=11)
ax.set_title('Method Correlation Matrix\nHow well do different approaches agree?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figT_method_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig T saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE U: Network Visualisation
# Spring-layout network showing all relationships above threshold
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

network_configs = [
    (axes[0], sim_text, 'Text Reuse Network\n(4-gram edges)', 0.005),
    (axes[1], sim_stylo, 'Stylometric Network\n(Cosine Delta edges)', 0.5),
    (axes[2], sim_anno, 'Annotation Network\n(value Jaccard edges)', 0.3),
]

# Use same layout across all networks for comparability
# Compute layout from annotation similarity (the ground truth)
G_layout = nx.Graph()
for name in common_names:
    G_layout.add_node(name)
for i in range(n):
    for j in range(i+1, n):
        if sim_anno[i,j] > 0.15:
            G_layout.add_edge(common_names[i], common_names[j],
                              weight=sim_anno[i,j])
pos = nx.spring_layout(G_layout, k=2.5, iterations=100, seed=42)

for ax, sim_mat, title, threshold in network_configs:
    G = nx.Graph()
    for name in common_names:
        G.add_node(name, group=get_group(name))

    edge_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if sim_mat[i,j] > threshold:
                G.add_edge(common_names[i], common_names[j],
                          weight=sim_mat[i,j])
                edge_count += 1

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        max_weight = max(d['weight'] for _, _, d in edges) if edges else 1
        for u, v, d in edges:
            w = d['weight']
            alpha = 0.15 + 0.6 * (w / max_weight)
            width = 0.5 + 3 * (w / max_weight)
            # Color edge by whether it's within or between groups
            if get_group(u) == get_group(v):
                color = GROUP_COLORS[get_group(u)]
            else:
                color = '#aaa'
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=color, alpha=alpha, linewidth=width, zorder=1)

    # Draw nodes
    for name in common_names:
        x, y = pos[name]
        grp = get_group(name)
        node_size = 200
        ax.scatter(x, y, s=node_size, c=GROUP_COLORS[grp],
                  edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7,
                    fontweight='bold', color=GROUP_COLORS[grp])

    ax.set_title(f'{title}\n({edge_count} edges)', fontsize=12, fontweight='bold')
    ax.axis('off')

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Gruppe {g}')
                  for g in ['I', 'II', 'III']]
axes[2].legend(handles=legend_patches, loc='lower right', fontsize=10)
fig.suptitle('Network Comparison: Three Lenses on Text Relationships\n'
             '(same node layout, different edge criteria)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figU_networks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig U saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE V: Cophenetic correlation comparison across all methods
# ═══════════════════════════════════════════════════════════════════

# Compute dendrograms and cophenetic matrices for all key methods
key_methods = {
    'Text 4-gram': 1 - sim_text,
    'Auto steps': 1 - sim_auto_step,
    'Cosine Δ 200': primary_stylo_dist,
    "Burrows' Δ 200": burrows_200_dist,
    'Anno values': 1 - sim_anno,
}

coph_matrices = {}
for label, dist_mat in key_methods.items():
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method='ward')
    _, coph_cond = cophenet(Z, condensed)
    coph_matrices[label] = squareform(coph_cond)

# Compare each method's tree topology to annotation tree
ref_coph = coph_matrices['Anno values']
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

compare_methods = ['Text 4-gram', 'Auto steps', 'Cosine Δ 200', "Burrows' Δ 200"]
for ax, method in zip(axes, compare_methods):
    method_coph = coph_matrices[method]
    for i in range(n):
        for j in range(i+1, n):
            g1, g2 = get_group(common_names[i]), get_group(common_names[j])
            same = g1 == g2
            color = GROUP_COLORS[g1] if same else '#ccc'
            marker = 'o' if same else 'x'
            ax.scatter(method_coph[i,j], ref_coph[i,j], c=color, marker=marker,
                       s=15, alpha=0.5)
    r_coph, _ = pearsonr(method_coph[upper], ref_coph[upper])
    ax.set_xlabel(f'Cophenetic dist ({method})', fontsize=9)
    ax.set_ylabel('Cophenetic dist (Annotations)', fontsize=9)
    ax.set_title(f'{method}\nr = {r_coph:.3f}', fontsize=11, fontweight='bold')

fig.suptitle('Tree Topology Comparison: Which Method Best Matches the Annotation Tree?',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figV_cophenetic_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig V saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE W: Nearest-neighbour agreement across all methods
# ═══════════════════════════════════════════════════════════════════

def get_nearest_neighbours(sim_matrix, names):
    nn = {}
    for i, name in enumerate(names):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        j = np.argmax(sims)
        nn[name] = names[j]
    return nn

nn_methods = {}
nn_agreement = {}
nn_anno = get_nearest_neighbours(sim_anno, common_names)

for label, sim_mat in [('Text 4-gram', sim_text),
                        ('Auto steps', sim_auto_step),
                        ('Cosine Δ 200', sim_stylo),
                        ("Burrows' Δ 200", sim_burrows),
                        ('Anno presence', sim_anno_presence)]:
    nn = get_nearest_neighbours(sim_mat, common_names)
    nn_methods[label] = nn
    agree = sum(1 for name in common_names if nn[name] == nn_anno[name])
    nn_agreement[label] = agree
    print(f"  NN agreement {label}: {agree}/{n} ({100*agree/n:.0f}%)")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
labels_sorted = sorted(nn_agreement.keys(), key=lambda k: -nn_agreement[k])
y_pos = range(len(labels_sorted))
values = [nn_agreement[l] for l in labels_sorted]
colors_bar = ['#3498db' if 'Anno' not in l else '#2ecc71' for l in labels_sorted]
bars = ax.barh(y_pos, values, color=colors_bar, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_sorted, fontsize=11)
ax.set_xlabel(f'Texts with matching nearest neighbour (out of {n})', fontsize=12)
for i, v in enumerate(values):
    ax.text(v + 0.3, i, f'{v}/{n} ({100*v/n:.0f}%)', va='center', fontsize=10)
ax.set_xlim(0, n + 2)
ax.set_title('Nearest-Neighbour Agreement with Expert Annotations\nWhich automated method best identifies the closest relative?',
             fontsize=13, fontweight='bold')
ax.axvline(n, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figW_nn_agreement.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig W saved")


# ═══════════════════════════════════════════════════════════════════
# FIGURE X: Detailed NN comparison table (visual)
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')

methods_for_table = ['Text 4-gram', 'Auto steps', 'Cosine Δ 200', "Burrows' Δ 200"]
col_labels = ['Text'] + methods_for_table + ['Expert Anno']
n_cols = len(col_labels)

# Header
for j, label in enumerate(col_labels):
    ax.text(j / (n_cols - 1), 1.0, label, ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            transform=ax.transAxes)

for i, name in enumerate(common_names):
    y = 1.0 - (i + 1.5) / (n + 2)
    # Text name
    ax.text(0 / (n_cols - 1), y, name,
            ha='center', va='center', fontsize=9,
            color=GROUP_COLORS[get_group(name)], fontweight='bold',
            transform=ax.transAxes)

    # Each method's NN
    expert_nn = nn_anno[name]
    for j, method in enumerate(methods_for_table):
        nn_name = nn_methods[method][name]
        match = nn_name == expert_nn
        color = '#2ecc71' if match else '#e74c3c'
        ax.text((j + 1) / (n_cols - 1), y, nn_name,
                ha='center', va='center', fontsize=8,
                color=color, fontweight='bold' if match else 'normal',
                transform=ax.transAxes)

    # Expert annotation NN
    ax.text((n_cols - 1) / (n_cols - 1), y, expert_nn,
            ha='center', va='center', fontsize=9,
            color='black', fontweight='bold',
            transform=ax.transAxes)

ax.set_title('Nearest-Neighbour Assignments: All Methods vs Expert\n'
             'Green = agrees with expert, Red = disagrees',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figX_nn_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig X saved")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS: Where stylometric analysis diverges from text reuse
# and from annotations — and why
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DIVERGENCE ANALYSIS: Stylometry vs Text Reuse vs Annotations")
print("="*70)

# 1. Cases where stylometry agrees with annotations but text reuse doesn't
print("\n  Cases where Cosine Delta agrees with annotations but 4-gram doesn't:")
nn_text = get_nearest_neighbours(sim_text, common_names)
nn_stylo = get_nearest_neighbours(sim_stylo, common_names)

for name in common_names:
    expert = nn_anno[name]
    text_nn = nn_text[name]
    stylo_nn = nn_stylo[name]

    if stylo_nn == expert and text_nn != expert:
        print(f"    {name}: stylo→{stylo_nn} (=expert), text→{text_nn} (wrong)")

print("\n  Cases where 4-gram agrees with annotations but Cosine Delta doesn't:")
for name in common_names:
    expert = nn_anno[name]
    text_nn = nn_text[name]
    stylo_nn = nn_stylo[name]

    if text_nn == expert and stylo_nn != expert:
        print(f"    {name}: text→{text_nn} (=expert), stylo→{stylo_nn} (wrong)")

print("\n  Cases where all automated methods disagree with annotations:")
for name in common_names:
    expert = nn_anno[name]
    text_nn = nn_text[name]
    stylo_nn = nn_stylo[name]
    auto_nn = nn_methods['Auto steps'][name]

    if text_nn != expert and stylo_nn != expert and auto_nn != expert:
        print(f"    {name}: expert→{expert}, text→{text_nn}, stylo→{stylo_nn}, auto→{auto_nn}")


# 2. Stylometric clustering quality — Adjusted Rand Index equivalent
# Compare group membership recovery
print("\n  Group recovery — which method best separates the three groups?")
for label, sim_mat in [('Text 4-gram', sim_text),
                        ('Auto steps', sim_auto_step),
                        ('Cosine Δ 200', sim_stylo),
                        ("Burrows' Δ 200", sim_burrows),
                        ('Anno values', sim_anno)]:
    within = []
    between = []
    for i in range(n):
        for j in range(i+1, n):
            if get_group(common_names[i]) == get_group(common_names[j]):
                within.append(sim_mat[i,j])
            else:
                between.append(sim_mat[i,j])
    ratio = np.mean(within) / np.mean(between) if np.mean(between) > 0 else float('inf')
    print(f"    {label:<28} within/between ratio: {ratio:.2f}x  "
          f"(within={np.mean(within):.4f}, between={np.mean(between):.4f})")


# 3. What drives stylometric distance that's different from text reuse?
print("\n  Top 10 MFW (200) driving stylometric distances:")
mfw_200 = [w for w, _ in vocab_counts.most_common(200)]
features_200 = np.array([compute_mfw_features(text_tokens[name], mfw_200)
                         for name in common_names])
# Compute feature variance (high variance = high discriminating power)
feature_variance = features_200.std(axis=0)
top_features = sorted(zip(mfw_200, feature_variance), key=lambda x: -x[1])
print(f"    {'Word':<20} {'Std dev of rel. freq':<25}")
for word, var in top_features[:20]:
    print(f"    {word:<20} {var:.5f}")


# 4. What MFW size works best?
print("\n  Effect of MFW size on correlation with annotations:")
for mfw_size in [50, 100, 150, 200, 300, 500]:
    mfw_list = [w for w, _ in vocab_counts.most_common(mfw_size)]
    features = np.array([compute_mfw_features(text_tokens[name], mfw_list)
                         for name in common_names])
    dist = cosine_delta(features)
    sim = 1 - dist / dist.max()
    r, _ = pearsonr(sim[upper], sim_anno[upper])
    r_p, _ = pearsonr(sim[upper], sim_anno_presence[upper])
    print(f"    MFW={mfw_size:<5} ↔ Anno values: r={r:.3f}, ↔ Anno presence: r={r_p:.3f}")


print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print(f"\nFigures saved to {OUT_DIR}/processus_fig[R-X]*.png")
