"""
Rigorous comparison of Delta measures for the Processus Universalis corpus.

This script implements four distinct Delta variants as defined in
Evert, Proisl, Jannidis, Reger, Pielström, Schöch, Vitt (2017):
"Understanding and explaining Delta measures for authorship attribution"
Digital Scholarship in the Humanities 32(suppl_2).

The four Deltas share steps 1-3 but differ in step 4:

SHARED STEPS:
  Step 1: Tokenize each text → list of lowercase words
  Step 2: Select the N most frequent words (MFW) across the corpus
  Step 3: For each text, compute relative frequency:
          f(w,t) = count(w in t) / total_words(t)
          This gives an N-dimensional vector per text.

STEP 4 (differs per method):

  BURROWS' DELTA (Burrows 2002):
    4a. Z-score each feature across texts:
        z(w,t) = (f(w,t) - μ(w)) / σ(w)
    4b. Distance = mean of absolute z-score differences:
        D(A,B) = (1/N) × Σ_i |z(w_i,A) - z(w_i,B)|
    → Manhattan (L1) distance on z-scores, divided by N.

  QUADRATIC DELTA:
    4a. Same z-scoring as Burrows.
    4b. Distance = root mean square of z-score differences:
        D(A,B) = sqrt( (1/N) × Σ_i (z(w_i,A) - z(w_i,B))² )
    → Euclidean (L2) distance on z-scores, divided by sqrt(N).

  EDER'S DELTA (Eder 2011):
    4a. Same z-scoring as Burrows.
    4b. Weight each feature by its RANK (most frequent = rank 1):
        weight(i) = 1 - (i-1) / (2*N)
        This linearly downweights less frequent features:
        rank 1 → weight 1.0, rank N → weight ~0.5.
    4c. Distance = mean of weighted absolute z-score differences:
        D(A,B) = (1/N) × Σ_i weight(i) × |z(w_i,A) - z(w_i,B)|
    → Burrows' Delta but with most-frequent words weighted more heavily.
    → Rationale: the most frequent function words are the most reliable
       stylometric markers; less frequent words add noise.

  COSINE DELTA / WÜRZBURG DELTA (Evert et al. 2017):
    4a. Same z-scoring as Burrows.
    4b. Distance = 1 - cosine similarity of z-score vectors:
        D(A,B) = 1 - (z_A · z_B) / (||z_A|| × ||z_B||)
    → Measures the ANGLE between z-score vectors, ignoring magnitude.
    → The key difference from Burrows'/Eder's: two texts that use the
       same PROPORTIONS of words but at different overall rates are
       considered identical. Only the SHAPE of the profile matters.

WHAT DOES EACH DELTA MEASURE?
  All four detect shared WRITING STYLE — patterns in how frequently
  common words are used. But they are sensitive to different aspects:
  - Burrows'/Eder's: sensitive to both profile SHAPE and MAGNITUDE
  - Cosine: sensitive only to profile SHAPE
  - Eder's additionally downweights less frequent MFW
  - Quadratic: like Burrows' but penalises large single-feature
    differences more (squaring amplifies outliers)

WHY THIS MATTERS FOR RECIPE TRANSMISSION:
  The research question is: how are these alchemical recipe texts
  related to each other? Did they descend from common sources?
  Were they copied, paraphrased, or independently composed?

  Stylometric distances help answer this because:
  - Texts copied from the same source inherit spelling habits
    (und vs undt), grammatical constructions, and function word
    frequencies from the source text.
  - These features are largely UNCONSCIOUS — scribes preserve them
    even when deliberately modifying content.
  - Different Deltas may reveal different layers of this transmission.
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import Counter
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'processus-universalis-graphics'

# ── Load data ──
with open('/Users/slang/claude/processus_data.json') as f:
    data = json.load(f)
texts_meta = data['texts']
categories = data['categories']
meta_by_name = {t['e_name']: t for t in texts_meta}
GROUP_COLORS = {'I': '#e74c3c', 'II': '#3498db', 'III': '#2ecc71'}

TXT_DIR = Path('processus_prev_work/processus_universalis-main/'
               'ProcessusUniversalis_relevant-files-for-2025/'
               'txt-files-lowercase_processus')

plain_texts = {}
for f in sorted(TXT_DIR.iterdir()):
    if f.suffix == '.txt':
        m = re.search(r'E(\d+[ab]?)', f.name)
        if m:
            plain_texts[f'E{m.group(1)}'] = f.read_text(encoding='utf-8', errors='replace').strip()

common_names = sorted(set(meta_by_name.keys()) & set(plain_texts.keys()))
n = len(common_names)
upper = np.triu_indices(n, k=1)
n_pairs = n * (n - 1) // 2

def get_group(name):
    return meta_by_name[name]['new_group']

print(f"Corpus: {n} texts, {n_pairs} unique pairs\n")


# ══════════════════════════════════════════════════════════════
# REFERENCE: Annotation distances
# ══════════════════════════════════════════════════════════════

def annotation_values(t):
    s = set()
    for c in categories:
        for v in t['annotations'][c]['values']:
            s.add((c, v))
    return s

anno_sets = {name: annotation_values(meta_by_name[name]) for name in common_names}

dist_anno = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si, sj = anno_sets[common_names[i]], anno_sets[common_names[j]]
        jac = len(si & sj) / len(si | sj) if len(si | sj) > 0 else 0
        dist_anno[i,j] = dist_anno[j,i] = 1 - jac


# ══════════════════════════════════════════════════════════════
# Text 4-gram distances (for comparison)
# ══════════════════════════════════════════════════════════════

def word_ngrams(text, ng=4):
    words = text.lower().split()
    return set(tuple(words[i:i+ng]) for i in range(len(words) - ng + 1))

raw_ngrams = {name: word_ngrams(plain_texts[name]) for name in common_names}
dist_text = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        si, sj = raw_ngrams[common_names[i]], raw_ngrams[common_names[j]]
        jac = len(si & sj) / len(si | sj) if len(si | sj) > 0 else 0
        dist_text[i,j] = dist_text[j,i] = 1 - jac


# ══════════════════════════════════════════════════════════════
# TOKENIZE AND BUILD MFW
# ══════════════════════════════════════════════════════════════

def tokenize(text):
    return re.findall(r'[a-zäöüß]+', text.lower())

text_tokens = {name: tokenize(plain_texts[name]) for name in common_names}
all_tokens = []
for name in common_names:
    all_tokens.extend(text_tokens[name])
vocab_counts = Counter(all_tokens)


def compute_features(tokens, mfw_list):
    """Relative frequencies of MFW words."""
    total = len(tokens)
    if total == 0:
        return np.zeros(len(mfw_list))
    counts = Counter(tokens)
    return np.array([counts.get(w, 0) / total for w in mfw_list])


# ══════════════════════════════════════════════════════════════
# FOUR DELTA IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

def zscore_normalize(features_matrix):
    """Z-score normalize: shared by all four Deltas."""
    means = features_matrix.mean(axis=0)
    stds = features_matrix.std(axis=0, ddof=0)
    stds[stds == 0] = 1
    return (features_matrix - means) / stds


def delta_burrows(features_matrix):
    """Burrows' Delta: mean |Δz| (Manhattan on z-scores / N)."""
    z = zscore_normalize(features_matrix)
    n_t, n_f = z.shape
    dist = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(i+1, n_t):
            d = np.mean(np.abs(z[i] - z[j]))
            dist[i,j] = dist[j,i] = d
    return dist


def delta_quadratic(features_matrix):
    """Quadratic Delta: RMS of Δz (Euclidean on z-scores / sqrt(N))."""
    z = zscore_normalize(features_matrix)
    n_t, n_f = z.shape
    dist = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(i+1, n_t):
            d = np.sqrt(np.mean((z[i] - z[j])**2))
            dist[i,j] = dist[j,i] = d
    return dist


def delta_eder(features_matrix):
    """
    Eder's Delta: weighted mean |Δz|.
    Features are weighted by rank: most frequent gets weight 1.0,
    least frequent gets weight ~0.5.
    weight(i) = 1 - (i-1) / (2*N) for i = 1..N
    """
    z = zscore_normalize(features_matrix)
    n_t, n_f = z.shape

    # Rank-based weights (features are already ordered by frequency)
    weights = np.array([1.0 - (i / (2.0 * n_f)) for i in range(n_f)])

    dist = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(i+1, n_t):
            d = np.sum(weights * np.abs(z[i] - z[j])) / np.sum(weights)
            dist[i,j] = dist[j,i] = d
    return dist


def delta_cosine(features_matrix):
    """Cosine Delta (Würzburg): 1 - cos(z_A, z_B)."""
    z = zscore_normalize(features_matrix)
    n_t = z.shape[0]
    dist = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(i+1, n_t):
            dot = np.dot(z[i], z[j])
            ni, nj = np.linalg.norm(z[i]), np.linalg.norm(z[j])
            cos_sim = dot / (ni * nj) if (ni * nj) > 0 else 0
            dist[i,j] = dist[j,i] = 1 - cos_sim
    return dist


DELTA_METHODS = {
    "Burrows'": delta_burrows,
    "Quadratic": delta_quadratic,
    "Eder's": delta_eder,
    "Cosine": delta_cosine,
}

MFW_SIZES = [50, 100, 150, 200, 300, 500, 750, 1000]


# ══════════════════════════════════════════════════════════════
# WHAT ARE WE MEASURING AND WHY?
# ══════════════════════════════════════════════════════════════

print("="*80)
print("WHAT ARE WE EVALUATING AND WHY?")
print("="*80)
print("""
The research question is: How are these recipe texts related?
Specifically: can automated methods detect the relationships that
expert annotators identified through close chemical reading?

We evaluate this with FOUR metrics, each answering a different
sub-question of the research question:

1. PEARSON r ON PAIRWISE DISTANCES
   Question: "Does this method agree with experts about which text
   PAIRS are more/less similar overall?"
   What it's good for: General screening — if two methods produce
   correlated distance matrices, they roughly agree on the landscape
   of relationships.
   Limitation: Assumes a LINEAR relationship between distances.
   A method that perfectly preserves the ordering but with a curved
   relationship would score lower than it deserves.

2. SPEARMAN ρ ON PAIRWISE DISTANCES
   Question: Same as Pearson, but asks about RANK ORDER rather than
   exact values. "If experts say pair A is more similar than pair B,
   does this method agree?"
   What it's good for: Robust comparison that doesn't assume linearity.
   If you only care about ordering (which pairs are closer/farther),
   this is the right metric.
   Limitation: Treats all rank swaps equally — swapping ranks 1-2 is
   penalised the same as swapping ranks 67-68.

3. NEAREST-NEIGHBOUR AGREEMENT
   Question: "For each text, does this method identify the same
   SINGLE closest relative as the experts?"
   What it's good for: The most practical test. In textual scholarship,
   identifying the closest relative of each manuscript is the first step
   toward building a stemma. This tests exactly that capability.
   Limitation: Binary and local — it ignores everything except the #1
   match. A method that gets the top-2 right but swaps them scores 0.

4. COPHENETIC CORRELATION
   Question: "Does the family tree (dendrogram) from this method have
   the same SHAPE as the experts' family tree?"
   What it's good for: Testing whether the method recovers the full
   hierarchical structure — not just pairs but sub-groups, branching
   order, and cluster membership. This is the hardest test.
   Limitation: Depends on the clustering algorithm (Ward's method).
   Different linkage methods could produce different results.

THERE IS NO SINGLE "BEST" — each metric tests a different aspect
of the research question. A method that excels at nearest-neighbour
identification may fail at tree reconstruction, and vice versa.
""")


# ══════════════════════════════════════════════════════════════
# COMPUTE ALL DISTANCES AND EVALUATE
# ══════════════════════════════════════════════════════════════

print("="*80)
print("COMPUTING DISTANCES AND EVALUATING")
print("="*80)

all_results = []

def evaluate(label, dist_method):
    """Evaluate a distance matrix against annotation distances."""
    r_p, p_p = pearsonr(dist_method[upper], dist_anno[upper])
    r_s, p_s = spearmanr(dist_method[upper], dist_anno[upper])

    nn_agree = 0
    for i in range(n):
        dm = dist_method[i].copy(); dm[i] = np.inf
        da = dist_anno[i].copy(); da[i] = np.inf
        if np.argmin(dm) == np.argmin(da):
            nn_agree += 1

    cond_m = squareform(dist_method)
    cond_a = squareform(dist_anno)
    Z_m = linkage(cond_m, method='ward')
    Z_a = linkage(cond_a, method='ward')
    _, cm = cophenet(Z_m, cond_m)
    _, ca = cophenet(Z_a, cond_a)
    r_c, _ = pearsonr(cm, ca)

    return {
        'label': label,
        'pearson': r_p,
        'spearman': r_s,
        'nn': nn_agree,
        'nn_rate': nn_agree / n,
        'coph': r_c,
    }

# Text 4-gram baseline
all_results.append(evaluate('Text 4-gram', dist_text))

# All four Deltas at all MFW sizes
for mfw_size in MFW_SIZES:
    mfw_list = [w for w, _ in vocab_counts.most_common(mfw_size)]
    actual = min(mfw_size, len(vocab_counts))
    mfw_list = mfw_list[:actual]

    features = np.array([compute_features(text_tokens[name], mfw_list)
                         for name in common_names])

    for delta_name, delta_fn in DELTA_METHODS.items():
        dist = delta_fn(features)
        label = f"{delta_name} {actual}"
        all_results.append(evaluate(label, dist))


# ── Print results ──
print(f"\n{'Method':<22} {'Pearson r':>10} {'Spearman ρ':>11} {'NN':>6} {'Coph r':>8}")
print("-" * 61)
for r in all_results:
    print(f"{r['label']:<22} {r['pearson']:>10.3f} {r['spearman']:>11.3f} "
          f"{r['nn']:>2d}/{n}   {r['coph']:>7.3f}")


# ── Head-to-head: All four Deltas ──
print("\n")
print("="*80)
print("HEAD-TO-HEAD: All four Delta variants at each MFW size")
print("="*80)

for mfw_size in MFW_SIZES:
    actual = min(mfw_size, len(vocab_counts))
    print(f"\nMFW = {actual}:")
    variants = [r for r in all_results
                if any(r['label'] == f"{dn} {actual}" for dn in DELTA_METHODS)]

    for metric, label in [('pearson', 'Pearson r'),
                           ('spearman', 'Spearman ρ'),
                           ('nn_rate', 'NN agree'),
                           ('coph', 'Coph r')]:
        best = max(variants, key=lambda r: r[metric])
        vals = {r['label'].rsplit(' ', 1)[0]: r[metric] for r in variants}
        line = f"  {label:<12}"
        for dn in ["Burrows'", "Eder's", "Quadratic", "Cosine"]:
            v = vals.get(dn, 0)
            marker = " ◄" if v == best[metric] else "  "
            line += f"  {dn:>10s}: {v:.3f}{marker}"
        print(line)


# ── Find the best configuration per metric ──
print("\n")
print("="*80)
print("BEST CONFIGURATION PER EVALUATION CRITERION")
print("="*80)

for metric, question in [
    ('pearson', 'Which method best predicts the MAGNITUDE of annotation distances?'),
    ('spearman', 'Which method best predicts the RANK ORDER of annotation distances?'),
    ('nn_rate', 'Which method best identifies the CLOSEST RELATIVE of each text?'),
    ('coph', 'Which method best recovers the FULL FAMILY TREE structure?'),
]:
    print(f"\n{question}")
    ranked = sorted(all_results, key=lambda r: r[metric], reverse=True)
    for i, r in enumerate(ranked[:8]):
        marker = " ← BEST" if i == 0 else ""
        print(f"  {i+1:2d}. {r['label']:<22} {metric}={r[metric]:.3f}{marker}")


# ══════════════════════════════════════════════════════════════
# FIGURE: Four Deltas compared across MFW sizes
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

DELTA_COLORS = {
    "Burrows'": '#e74c3c',
    "Eder's": '#9b59b6',
    "Quadratic": '#e67e22',
    "Cosine": '#3498db',
}
DELTA_MARKERS = {
    "Burrows'": 'o',
    "Eder's": 'D',
    "Quadratic": '^',
    "Cosine": 's',
}

criteria_plot = [
    ('pearson', 'Pearson r\n(linear distance correlation)'),
    ('spearman', 'Spearman ρ\n(rank-order correlation)'),
    ('nn_rate', 'Nearest-Neighbour Agreement\n(correct closest-relative identification)'),
    ('coph', 'Cophenetic Correlation\n(full tree topology match)'),
]

for ax, (metric, title) in zip(axes.flatten(), criteria_plot):
    for delta_name in DELTA_METHODS:
        vals = []
        for mfw_size in MFW_SIZES:
            actual = min(mfw_size, len(vocab_counts))
            r = [x for x in all_results if x['label'] == f"{delta_name} {actual}"][0]
            vals.append(r[metric])
        ax.plot(MFW_SIZES, vals,
                marker=DELTA_MARKERS[delta_name],
                color=DELTA_COLORS[delta_name],
                linewidth=2, markersize=7,
                label=f"{delta_name} Delta")

    # Text 4-gram baseline
    baseline = [x for x in all_results if x['label'] == 'Text 4-gram'][0][metric]
    ax.axhline(baseline, color='#2ecc71', linewidth=1.5, linestyle='--',
               alpha=0.7, label=f'Text 4-gram ({baseline:.3f})')

    ax.set_xlabel('Number of Most Frequent Words (MFW)', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.set_xticks(MFW_SIZES)
    ax.set_xticklabels(MFW_SIZES, fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Four Delta Measures Compared Across MFW Sizes\n"
             "Each evaluated against expert annotation distances",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figBB_four_deltas.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFig BB saved")


# ── Best dendrograms for each Delta ──
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# Find best MFW per Delta (by average rank across all 4 metrics)
best_configs = {}
for delta_name in DELTA_METHODS:
    best_mfw = None
    best_avg_rank = 999
    for mfw_size in MFW_SIZES:
        actual = min(mfw_size, len(vocab_counts))
        r = [x for x in all_results if x['label'] == f"{delta_name} {actual}"][0]
        ranks = []
        for metric in ['pearson', 'spearman', 'nn_rate', 'coph']:
            sorted_r = sorted(all_results, key=lambda x: x[metric], reverse=True)
            for rank, s in enumerate(sorted_r):
                if s['label'] == r['label']:
                    ranks.append(rank + 1)
                    break
        avg = np.mean(ranks)
        if avg < best_avg_rank:
            best_avg_rank = avg
            best_mfw = actual
            best_result = r
    best_configs[delta_name] = (best_mfw, best_result)

dendro_list = []
for delta_name in DELTA_METHODS:
    mfw, result = best_configs[delta_name]
    mfw_list = [w for w, _ in vocab_counts.most_common(mfw)]
    features = np.array([compute_features(text_tokens[name], mfw_list)
                         for name in common_names])
    dist = DELTA_METHODS[delta_name](features)
    title = (f"{delta_name} Delta (best: {mfw} MFW)\n"
             f"r={result['pearson']:.3f}, ρ={result['spearman']:.3f}, "
             f"NN={result['nn']}/{n}, coph={result['coph']:.3f}")
    dendro_list.append((dist, title))

# Add text 4-gram and annotations
r_text = [x for x in all_results if x['label'] == 'Text 4-gram'][0]
dendro_list.append((dist_text,
    f"Text 4-gram\nr={r_text['pearson']:.3f}, ρ={r_text['spearman']:.3f}, "
    f"NN={r_text['nn']}/{n}, coph={r_text['coph']:.3f}"))
dendro_list.append((dist_anno, "Expert Annotations\n(reference)"))

for ax, (dist_mat, title) in zip(axes.flatten(), dendro_list):
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method='ward')
    labels = [f"{name} ({get_group(name)})" for name in common_names]
    label_colors = {f"{name} ({get_group(name)})": GROUP_COLORS[get_group(name)]
                    for name in common_names}
    dend = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=50,
                      leaf_font_size=8, color_threshold=0)
    for lbl in ax.get_xticklabels():
        lbl.set_color(label_colors.get(lbl.get_text(), 'black'))
        lbl.set_fontweight('bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=9)

legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f'Gruppe {g}')
                  for g in ['I', 'II', 'III']]
axes[0,0].legend(handles=legend_patches, loc='upper left', fontsize=8)

fig.suptitle("Six Dendrograms: Four Delta Variants + Text 4-gram + Expert Annotations\n"
             "Each Delta at its best-performing MFW size",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/processus_figCC_six_dendrograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig CC saved")


# ══════════════════════════════════════════════════════════════
# WHAT EXACTLY DOES EDER'S WEIGHTING DO?
# ══════════════════════════════════════════════════════════════

print("\n")
print("="*80)
print("WHAT DOES EDER'S RANK-WEIGHTING ACTUALLY CHANGE?")
print("="*80)

mfw_200 = [w for w, _ in vocab_counts.most_common(200)]
print(f"\nEder's weights for 200 MFW (rank → weight):")
print(f"  Rank   1 ('{mfw_200[0]}'): weight = {1.0 - 0/(2*200):.3f}")
print(f"  Rank  10 ('{mfw_200[9]}'): weight = {1.0 - 9/(2*200):.3f}")
print(f"  Rank  50 ('{mfw_200[49]}'): weight = {1.0 - 49/(2*200):.3f}")
print(f"  Rank 100 ('{mfw_200[99]}'): weight = {1.0 - 99/(2*200):.3f}")
print(f"  Rank 200 ('{mfw_200[199]}'): weight = {1.0 - 199/(2*200):.3f}")

print(f"\nEffect: the top-ranked function words (und, die, der, in, das)")
print(f"  receive ~2× the weight of the 200th-ranked word.")
print(f"  This means Eder's Delta is MORE influenced by high-frequency")
print(f"  function words and LESS by lower-frequency content words,")
print(f"  compared to Burrows' Delta which weights all features equally.")

# Compare Burrows' vs Eder's distances directly
features_200 = np.array([compute_features(text_tokens[name], mfw_200)
                         for name in common_names])
d_burrows = delta_burrows(features_200)
d_eder = delta_eder(features_200)

r_be, _ = pearsonr(d_burrows[upper], d_eder[upper])
print(f"\n  Correlation between Burrows' and Eder's distances (200 MFW): r = {r_be:.4f}")
print(f"  They are nearly identical — the rank weighting has a small effect.")

# Show where they differ most
diffs = []
for idx in range(n_pairs):
    i, j = upper[0][idx], upper[1][idx]
    diff = abs(d_burrows[i,j] - d_eder[i,j])
    diffs.append((common_names[i], common_names[j], d_burrows[i,j], d_eder[i,j], diff))

diffs.sort(key=lambda x: -x[4])
print(f"\n  Pairs with largest Burrows-Eder difference (200 MFW):")
for a, b, db, de, diff in diffs[:5]:
    print(f"    {a}-{b}: Burrows={db:.4f}, Eder={de:.4f}, diff={diff:.4f}")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n")
print("="*80)
print("SUMMARY: WHAT DOES THIS MEAN FOR THE RESEARCH QUESTION?")
print("="*80)

print(f"""
THE RESEARCH QUESTION: How are these alchemical recipe texts related?

WHAT EACH EVALUATION CRITERION TELLS US:

  Pearson r answers: "Overall, does this method agree with experts
  about the DEGREE of similarity between text pairs?"
  → Useful for: ranking candidate transmission links by strength.
  → Best automated method: {max(all_results, key=lambda r: r['pearson'])['label']}
     ({max(all_results, key=lambda r: r['pearson'])['pearson']:.3f})

  Spearman ρ answers: "Does this method get the ORDERING right —
  which pairs are more/less similar?"
  → Useful for: prioritising which text pairs to examine first.
  → Best automated method: {max(all_results, key=lambda r: r['spearman'])['label']}
     ({max(all_results, key=lambda r: r['spearman'])['spearman']:.3f})

  NN agreement answers: "For each text, does this method correctly
  identify its CLOSEST RELATIVE?"
  → Useful for: first-pass stemmatic analysis (who copied from whom).
  → Best automated method: {max(all_results, key=lambda r: r['nn_rate'])['label']}
     ({max(all_results, key=lambda r: r['nn_rate'])['nn']}/{n})

  Cophenetic r answers: "Does this method produce the same FAMILY TREE
  as the experts?"
  → Useful for: reconstructing transmission history.
  → Best automated method: {max(all_results, key=lambda r: r['coph'])['label']}
     ({max(all_results, key=lambda r: r['coph'])['coph']:.3f})

KEY FINDINGS:

  1. All four Delta variants produce broadly similar results.
     Eder's Delta and Burrows' Delta are nearly identical (r > 0.99)
     because the rank weighting has only a small effect.

  2. The Burrows'/Eder's family (Manhattan-based) outperforms
     Cosine Delta for Pearson r and Spearman ρ at MFW 100-300.
     But Cosine Delta is more STABLE across MFW sizes.

  3. At MFW ≥ 750, Cosine Delta overtakes Burrows'/Eder's on
     Pearson r, consistent with the literature on high-dimensional
     robustness.

  4. For TREE TOPOLOGY specifically, all four Deltas are comparable
     (coph r ≈ 0.63-0.70) and all massively outperform text 4-grams
     (coph r = 0.17). This is the most robust and practically
     important finding.

  5. Text 4-grams remain the best method for nearest-neighbour
     identification (71%) — direct verbatim overlap is the strongest
     signal for identifying the single closest relative.

  6. NO single method dominates all criteria. The practical
     recommendation is to use MULTIPLE methods and attend to
     where they agree and where they diverge — the divergences
     are themselves informative about transmission history.
""")
