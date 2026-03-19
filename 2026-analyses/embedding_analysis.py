#!/usr/bin/env python3
"""
Embedding-Based Semantic Analysis
==================================
Bridges the gap between surface word matching and actual text meaning
using multilingual sentence embeddings.

Key idea: instead of classifying individual words into fixed categories,
embed text *passages* into a semantic space and measure their similarity
to reference concepts. This captures meaning-in-context rather than
relying on exact word matches.

Produces Figures SS through WW.
"""

import re
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cosine as cosine_dist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

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

# ── Load model ──
print("Loading multilingual sentence-transformer model...")
print("(paraphrase-multilingual-MiniLM-L12-v2 — trained on 50+ languages incl. German)")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.\n")


# ── Reference concept passages ──
# These anchor passages define the two semantic poles.
# They are written in a mix of modern German and the same Early New High German
# register as the texts, to give the model fair reference points.

PRACTICAL_ANCHORS = [
    # Laboratory equipment and operations
    "Nimm eine Retorte und destillire den Spiritus durch Feuer.",
    "Filtrire die Lauge und evaporire das Wasser.",
    "Calcinire die Erde im offenen Feuer zu Aschen.",
    "Sublimire das flüchtige Saltz im Kolben.",
    "Gib starck Feuer bis die Spiritus alle herüber sein.",
    "Setze den Kolben in Balneum Mariae zu destilliren.",
    "Nimm 6 Pfund Erde in eine verlutirte Retorte.",
    "Solvire und coagulire das Saltz bis es crystallisch wird.",
    "Rectificire den Spiritum in Arena sechsmal.",
    "Lauge die Erde aus mit destillirtem Regenwasser.",
    "Schlag 2 Pfund Wasser in den Recipienten vor.",
    "Distill the spirit of niter through a retort.",
    "Filter the solution and evaporate to crystallization.",
    "Calcine the earth in an open fire to ash.",
]

THEORETICAL_ANCHORS = [
    # Philosophical, cosmological, and transmutation claims
    "Die Tinctur verwandelt alle unedle Metalle in das edelste Gold.",
    "Lapis Philosophorum, der Stein der Weisen.",
    "Multiplicatio: ein Theil auf zehen, dann auf hundert, dann auf tausend.",
    "Die himmlischen Einflüsse schwängern die jungfräuliche Erde.",
    "Der Spiritus Mundi, der unsichtbare Geist der Natur.",
    "Soli Deo Gloria. Amen.",
    "Die Tinctur heilet alle Kranckheiten des menschlichen Leibes.",
    "Das Menstruum Universale schliesset alle Metallen und Edelgesteine auf.",
    "Die Natur scheidet das allerreinste Quintum Esse.",
    "Projectio: wirf ein Gran auf tausend Theil geschmolzenes Blei.",
    "Fermentatio des rothen Pulvers mit feinem Gold.",
    "The philosopher's stone transmutes base metals into gold.",
    "The universal medicine cures all diseases.",
    "Multiplication of the tincture without end.",
]

# A THIRD pole: the cosmological/nature-philosophy preamble
# (distinct from transmutation claims — describes the earth's nature)
COSMOLOGICAL_ANCHORS = [
    "Die Erde ist das Subjectum aller himmlischen Strahlen und Einflüsse.",
    "Sie wird von den Elementen und Himmeln geschwängert.",
    "In ihrem Centro ist eine jungfräuliche Erde verborgen.",
    "Die Erde ist das Centrum und Fundamentum aller Dinge.",
    "Der erstgeborne himmlische Geist der Natur ist in ihr verborgen.",
    "In der Erde sind drey unterschiedliche Salia verborgen.",
    "Die Sonne und Sterne schwängern die Erde mit ihren Strahlen.",
    "The earth is the mother of all things and contains all seeds.",
]

# A FOURTH pole: color-stage / opus magnum process descriptions
COLOR_STAGE_ANCHORS = [
    "Es wird sich die Schwärtze erzeigen nach vierzig Tagen.",
    "Mancherley Farben werden sich erzeigen und letzlich eine grüne.",
    "Es wird in dreissig Tagen eine weisse Farbe erscheinen.",
    "Letzlich wird das Pulver roth und durchsichtig.",
    "In der Mitten ein rubinfarbenes Körnlein, einer Linsen gross.",
    "Die Putrefaction währet vierzig oder fünfundvierzig Tage.",
    "Fahre fort bis sich die gelbe Farbe erzeiget.",
    "The blackening appears after forty days of putrefaction.",
    "Colors appear: first green, then white, yellow, and finally red.",
]


def load_texts():
    """Load all texts, return dict of {name: raw_text}."""
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


def chunk_text(text, chunk_size=80, overlap=20):
    """Split text into overlapping word chunks. Returns list of (mid_position, chunk_text)."""
    words = text.split()
    n = len(words)
    if n < chunk_size:
        return [(0.5, text)]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, n - chunk_size + 1, step):
        chunk_words = words[i:i + chunk_size]
        mid_pos = (i + chunk_size / 2) / n
        chunks.append((mid_pos, ' '.join(chunk_words)))

    # Make sure we include the very end
    if chunks and chunks[-1][0] < 0.95:
        chunk_words = words[-chunk_size:]
        mid_pos = (n - chunk_size / 2) / n
        chunks.append((mid_pos, ' '.join(chunk_words)))

    return chunks


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return 1.0 - cosine_dist(a, b)


# ══════════════════════════════════════════════════════════════
print("Step 1: Embedding reference anchors...")
# ══════════════════════════════════════════════════════════════

prac_embeddings = model.encode(PRACTICAL_ANCHORS)
theo_embeddings = model.encode(THEORETICAL_ANCHORS)
cosmo_embeddings = model.encode(COSMOLOGICAL_ANCHORS)
color_embeddings = model.encode(COLOR_STAGE_ANCHORS)

# Compute centroid for each pole
prac_centroid = np.mean(prac_embeddings, axis=0)
theo_centroid = np.mean(theo_embeddings, axis=0)
cosmo_centroid = np.mean(cosmo_embeddings, axis=0)
color_centroid = np.mean(color_embeddings, axis=0)

# Verify anchors are semantically distinct
print(f"  Practical ↔ Theoretical cosine sim: {cosine_sim(prac_centroid, theo_centroid):.3f}")
print(f"  Practical ↔ Cosmological cosine sim: {cosine_sim(prac_centroid, cosmo_centroid):.3f}")
print(f"  Theoretical ↔ Cosmological cosine sim: {cosine_sim(theo_centroid, cosmo_centroid):.3f}")
print(f"  Practical ↔ Color stages cosine sim: {cosine_sim(prac_centroid, color_centroid):.3f}")
print(f"  Theoretical ↔ Color stages cosine sim: {cosine_sim(theo_centroid, color_centroid):.3f}")
print(f"  Cosmological ↔ Color stages cosine sim: {cosine_sim(cosmo_centroid, color_centroid):.3f}")

# ══════════════════════════════════════════════════════════════
print("\nStep 2: Loading and chunking texts...")
# ══════════════════════════════════════════════════════════════

texts_raw = load_texts()
text_names = sorted(texts_raw.keys(), key=lambda x: (GROUP_MAP[x], x))
print(f"  Loaded {len(text_names)} texts")

# Chunk each text
text_chunks = {}
for nm in text_names:
    chunks = chunk_text(texts_raw[nm], chunk_size=80, overlap=20)
    text_chunks[nm] = chunks
    print(f"  {nm}: {len(chunks)} chunks")

# ══════════════════════════════════════════════════════════════
print("\nStep 3: Embedding all text chunks...")
# ══════════════════════════════════════════════════════════════

text_chunk_embeddings = {}
for nm in text_names:
    chunk_texts = [c[1] for c in text_chunks[nm]]
    embeddings = model.encode(chunk_texts, show_progress_bar=False)
    text_chunk_embeddings[nm] = embeddings
    print(f"  {nm}: {len(embeddings)} embeddings computed")

# ══════════════════════════════════════════════════════════════
print("\nStep 4: Computing semantic similarity to each pole...")
# ══════════════════════════════════════════════════════════════

# For each chunk, compute similarity to each pole centroid
text_profiles = {}
for nm in text_names:
    chunks = text_chunks[nm]
    embeddings = text_chunk_embeddings[nm]

    profile = []
    for i, (pos, _) in enumerate(chunks):
        emb = embeddings[i]
        profile.append({
            'pos': pos,
            'practical': cosine_sim(emb, prac_centroid),
            'theoretical': cosine_sim(emb, theo_centroid),
            'cosmological': cosine_sim(emb, cosmo_centroid),
            'color_stage': cosine_sim(emb, color_centroid),
        })
    text_profiles[nm] = profile

# Print summary
print("\n  Per-text mean similarities:")
print(f"  {'Text':>5s}  {'Practical':>10s}  {'Theoretical':>12s}  {'Cosmological':>13s}  {'Color Stage':>12s}")
for nm in text_names:
    p = text_profiles[nm]
    mp = np.mean([x['practical'] for x in p])
    mt = np.mean([x['theoretical'] for x in p])
    mc = np.mean([x['cosmological'] for x in p])
    mcs = np.mean([x['color_stage'] for x in p])
    print(f"  {nm:>5s}  {mp:>10.3f}  {mt:>12.3f}  {mc:>13.3f}  {mcs:>12.3f}")


# ══════════════════════════════════════════════════════════════
# FIGURE SS: Embedding-based semantic trajectories (per text)
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure SS: Semantic trajectories...")

fig, axes = plt.subplots(4, 5, figsize=(26, 20), sharey=False)
axes_flat = axes.flatten()

for ax in axes_flat[len(text_names):]:
    ax.set_visible(False)

for idx, nm in enumerate(text_names):
    ax = axes_flat[idx]
    profile = text_profiles[nm]
    positions = [p['pos'] for p in profile]
    sim_prac = [p['practical'] for p in profile]
    sim_theo = [p['theoretical'] for p in profile]
    sim_cosmo = [p['cosmological'] for p in profile]
    sim_color = [p['color_stage'] for p in profile]

    ax.plot(positions, sim_prac, color='#3498db', lw=2, label='Practical')
    ax.plot(positions, sim_theo, color='#e74c3c', lw=2, label='Theoretical')
    ax.plot(positions, sim_cosmo, color='#e67e22', lw=1.5, ls='--', label='Cosmological')
    ax.plot(positions, sim_color, color='#9b59b6', lw=1.5, ls=':', label='Color stages')

    ax.axvspan(0.75, 1.0, alpha=0.08, color='grey')

    g = GROUP_MAP[nm]
    ax.set_title(f"{nm} (Gruppe {g})", fontsize=11, fontweight='bold',
                 color=GROUP_COLORS[g])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Text position', fontsize=8)
    if idx % 5 == 0:
        ax.set_ylabel('Cosine similarity', fontsize=9)
    ax.tick_params(labelsize=8)

handles = [
    Line2D([0], [0], color='#3498db', lw=2, label='Practical chemistry'),
    Line2D([0], [0], color='#e74c3c', lw=2, label='Transmutation/philosophical'),
    Line2D([0], [0], color='#e67e22', lw=1.5, ls='--', label='Cosmological preamble'),
    Line2D([0], [0], color='#9b59b6', lw=1.5, ls=':', label='Color stages'),
    Patch(facecolor='grey', alpha=0.15, label='Last 25% of text'),
]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=11,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Embedding-Based Semantic Trajectories\n"
             "(cosine similarity of each passage to four reference concept poles)",
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUT_DIR / 'processus_figSS_embedding_trajectories.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig SS saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE TT: Comparison — word-list vs embedding classification
# ══════════════════════════════════════════════════════════════
print("Generating Figure TT: Word-list vs embedding comparison...")

# For each text, compute:
#   - Word-list practical/theoretical ratio per quintile (from language_chemistry_divergence.py logic)
#   - Embedding practical/theoretical similarity per quintile

# Recreate the word lists for comparison
from language_chemistry_divergence import CHEM_PRACTICAL, CHEM_THEORETICAL, classify_word

N_SEG = 5
seg_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

fig, axes = plt.subplots(4, 5, figsize=(26, 20))
axes_flat = axes.flatten()
for ax in axes_flat[len(text_names):]:
    ax.set_visible(False)

for idx, nm in enumerate(text_names):
    ax = axes_flat[idx]

    # Word-list method: classify words by quintile
    raw_words = re.findall(r'[a-zäöüß\-]+', texts_raw[nm].lower())
    n_words = len(raw_words)
    seg_size = n_words // N_SEG

    wl_balance = []
    for s in range(N_SEG):
        start = s * seg_size
        end = (s + 1) * seg_size if s < N_SEG - 1 else n_words
        chunk = raw_words[start:end]
        counts = Counter(classify_word(w) for w in chunk)
        prac = counts.get('practical', 0)
        theo = counts.get('theoretical', 0)
        if prac + theo > 0:
            wl_balance.append(prac / (prac + theo))
        else:
            wl_balance.append(0.5)

    # Embedding method: average similarity per quintile
    emb_balance = []
    profile = text_profiles[nm]
    for s in range(N_SEG):
        seg_start = s / N_SEG
        seg_end = (s + 1) / N_SEG
        seg_chunks = [p for p in profile if seg_start <= p['pos'] < seg_end]
        if not seg_chunks:
            # Fall back to nearest
            seg_chunks = [min(profile, key=lambda p: abs(p['pos'] - (seg_start + seg_end) / 2))]
        mean_prac = np.mean([p['practical'] for p in seg_chunks])
        mean_theo = np.mean([p['theoretical'] for p in seg_chunks])
        emb_balance.append(mean_prac / (mean_prac + mean_theo))

    x = np.arange(N_SEG)
    w = 0.35
    ax.bar(x - w/2, wl_balance, w, color='#3498db', alpha=0.7, label='Word-list')
    ax.bar(x + w/2, emb_balance, w, color='#e74c3c', alpha=0.7, label='Embedding')
    ax.axhline(0.5, color='black', ls='--', lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels, fontsize=7, rotation=30)
    ax.set_ylim(0.3, 0.75)

    g = GROUP_MAP[nm]
    ax.set_title(f"{nm} ({g})", fontsize=10, fontweight='bold', color=GROUP_COLORS[g])
    if idx % 5 == 0:
        ax.set_ylabel('Practical fraction\n(> 0.5 = more practical)', fontsize=8)

axes_flat[0].legend(fontsize=9, loc='upper right')
fig.suptitle("Word-List vs Embedding Classification: Side-by-Side by Quintile\n"
             "(blue = word-list method, red = embedding method; "
             "above 0.5 = more practical, below = more theoretical)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_DIR / 'processus_figTT_wordlist_vs_embedding.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig TT saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE UU: Embedding-based distance matrix vs word-list and expert
# ══════════════════════════════════════════════════════════════
print("Generating Figure UU: Embedding distance comparison...")

# Compute a text-level embedding by averaging all chunk embeddings
text_embeddings = {}
for nm in text_names:
    text_embeddings[nm] = np.mean(text_chunk_embeddings[nm], axis=0)

# Embedding distance matrix
n = len(text_names)
emb_dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        emb_dist[i, j] = cosine_dist(text_embeddings[text_names[i]],
                                      text_embeddings[text_names[j]])

# Load expert distance matrix from XML annotations
anno_dir = Path("processus/processus_prev_work/processus_universalis-main/"
                "ProcessusUniversalis_relevant-files-for-2025")
xml_path = anno_dir / "sammlung_aller_texte.xml"

import xml.etree.ElementTree as ET
tree = ET.parse(xml_path)
root = tree.getroot()

# A-number (XML) to E-number (txt) mapping, matched by title
A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2',  'a11': 'E27', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',  # A16=Alexander Sethonius = E27
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11',
}

# Extract annotation keywords per text
anno_features = {}
for div in root.findall('div'):
    dtype = div.get('type', '')
    # dtype is like 'g1a1', 'g2a2', etc.
    m = re.search(r'a(\d+)', dtype)
    if not m:
        continue
    a_key = 'a' + m.group(1)
    ename = A_TO_E.get(a_key)
    if not ename or ename not in GROUP_MAP:
        continue

    features = set()
    for keys_el in div.findall('.//keys'):
        ktype = keys_el.get('type', '')
        kvals = keys_el.get('n', '')
        if kvals and 'FEHLT' not in kvals:
            for val in kvals.split(';'):
                val = val.strip()
                if val:
                    features.add(f"{ktype}::{val}")
    if features:
        anno_features[ename] = features

print(f"  Expert annotations loaded for {len(anno_features)} texts: {sorted(anno_features.keys())}")

# Build expert Jaccard distance
common = [nm for nm in text_names if nm in anno_features]
expert_dist = np.zeros((len(common), len(common)))
for i in range(len(common)):
    for j in range(len(common)):
        a = anno_features[common[i]]
        b = anno_features[common[j]]
        if len(a | b) > 0:
            expert_dist[i, j] = 1.0 - len(a & b) / len(a | b)

# Word-list based: compute Jaccard on word-list classified vocab per text
wl_features = {}
for nm in text_names:
    raw_words = re.findall(r'[a-zäöüß\-]+', texts_raw[nm].lower())
    prac_set = set(w for w in raw_words if w in CHEM_PRACTICAL)
    theo_set = set(w for w in raw_words if w in CHEM_THEORETICAL)
    wl_features[nm] = prac_set | theo_set

wl_dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a = wl_features[text_names[i]]
        b = wl_features[text_names[j]]
        if len(a | b) > 0:
            wl_dist[i, j] = 1.0 - len(a & b) / len(a | b)

# Correlations
from scipy.stats import pearsonr, spearmanr

# Flatten upper triangles
def upper_tri(mat):
    idx = np.triu_indices(len(mat), k=1)
    return mat[idx]

emb_flat = upper_tri(emb_dist)
expert_flat = upper_tri(expert_dist[:len(common), :len(common)])
# Need to align — use only common texts
common_idx = [text_names.index(nm) for nm in common]
emb_common = np.zeros((len(common), len(common)))
for i, ci in enumerate(common_idx):
    for j, cj in enumerate(common_idx):
        emb_common[i, j] = emb_dist[ci, cj]
emb_flat = upper_tri(emb_common)

r_emb_expert, _ = pearsonr(emb_flat, expert_flat)
rho_emb_expert, _ = spearmanr(emb_flat, expert_flat)

print(f"  Embedding vs Expert: r={r_emb_expert:.3f}, rho={rho_emb_expert:.3f}")

# Also compare embedding distances from early vs late halves
# Embed first half and second half separately
print("\n  Computing early-half vs late-half embedding distances...")
early_embeddings = {}
late_embeddings = {}
for nm in text_names:
    chunks = text_chunks[nm]
    embs = text_chunk_embeddings[nm]
    mid = len(chunks) // 2
    if mid > 0:
        early_embeddings[nm] = np.mean(embs[:mid], axis=0)
        late_embeddings[nm] = np.mean(embs[mid:], axis=0)
    else:
        early_embeddings[nm] = embs[0]
        late_embeddings[nm] = embs[-1]

early_dist = np.zeros((n, n))
late_dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        early_dist[i, j] = cosine_dist(early_embeddings[text_names[i]],
                                        early_embeddings[text_names[j]])
        late_dist[i, j] = cosine_dist(late_embeddings[text_names[i]],
                                       late_embeddings[text_names[j]])

# Plot: three dendrograms (embedding full, embedding late, expert)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8))

for ax, dist_mat, title, names in [
    (ax1, emb_common, f'Embedding (full text)\nr={r_emb_expert:.3f}, rho={rho_emb_expert:.3f}', common),
    (ax2, late_dist[np.ix_(common_idx, common_idx)],
     'Embedding (late half only)', common),
    (ax3, expert_dist, 'Expert Annotations\n(reference)', common),
]:
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method='ward')
    dn = dendrogram(Z, labels=names, ax=ax, leaf_rotation=90, leaf_font_size=9)

    # Color labels
    for lbl in ax.get_xticklabels():
        nm = lbl.get_text()
        if nm in GROUP_MAP:
            lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward distance', fontsize=10)

legend_handles = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc='upper right', fontsize=10)
fig.suptitle("Embedding-Based Dendrograms vs Expert Annotations\n"
             "(do semantic embeddings recover the expert-defined text relationships?)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.95, 0.92])
plt.savefig(OUT_DIR / 'processus_figUU_embedding_dendrograms.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig UU saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE VV: The ambiguous words — how embeddings handle them
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure VV: Ambiguous word contexts...")

# For the words that were excluded from word lists (gold, wasser, feuer, saltz, erde, geist),
# find all passages containing each word and embed them.
# Then measure whether the embedding is closer to practical or theoretical pole
# depending on WHERE in the text the word appears.

AMBIGUOUS_WORDS = ['gold', 'wasser', 'feuer', 'saltz', 'erde', 'geist']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for w_idx, target_word in enumerate(AMBIGUOUS_WORDS):
    ax = axes[w_idx // 3, w_idx % 3]

    all_positions = []
    all_prac_sims = []
    all_theo_sims = []
    all_groups = []

    for nm in text_names:
        raw_lower = texts_raw[nm].lower()
        words = raw_lower.split()
        n_words = len(words)

        # Find all occurrences
        for i, w in enumerate(words):
            # Clean word for matching
            w_clean = re.sub(r'[^a-zäöüß]', '', w)
            if w_clean != target_word:
                continue

            # Get context window (±30 words)
            start = max(0, i - 30)
            end = min(n_words, i + 31)
            context = ' '.join(words[start:end])

            # Embed context
            emb = model.encode([context])[0]

            pos = i / n_words
            all_positions.append(pos)
            all_prac_sims.append(cosine_sim(emb, prac_centroid))
            all_theo_sims.append(cosine_sim(emb, theo_centroid))
            all_groups.append(GROUP_MAP[nm])

    if not all_positions:
        ax.text(0.5, 0.5, f'"{target_word}" not found', ha='center', va='center',
                transform=ax.transAxes)
        continue

    # Compute practical-theoretical balance
    balance = [p / (p + t) for p, t in zip(all_prac_sims, all_theo_sims)]

    colors = [GROUP_COLORS[g] for g in all_groups]
    ax.scatter(all_positions, balance, c=colors, alpha=0.5, s=30, edgecolors='white',
               linewidths=0.3)

    # Trend line
    if len(all_positions) > 5:
        z = np.polyfit(all_positions, balance, 2)
        p_line = np.poly1d(z)
        x_smooth = np.linspace(0, 1, 100)
        ax.plot(x_smooth, p_line(x_smooth), color='black', lw=2, ls='--', alpha=0.7)

    ax.axhline(0.5, color='grey', ls=':', lw=1, alpha=0.5)
    ax.text(0.02, 0.52, 'practical', fontsize=7, color='#3498db', transform=ax.transAxes)
    ax.text(0.02, 0.47, 'theoretical', fontsize=7, color='#e74c3c', transform=ax.transAxes)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('Text position', fontsize=10)
    ax.set_ylabel('Practical ← balance → Theoretical', fontsize=9)
    ax.set_title(f'"{target_word}" ({len(all_positions)} occurrences)',
                 fontsize=12, fontweight='bold')

legend_handles = [Patch(facecolor=c, label=f'Gruppe {g}') for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("How Embeddings See Ambiguous Words in Context\n"
             "(same word, different meaning depending on position? "
             "dots above 0.5 = practical context, below = theoretical)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figVV_ambiguous_words.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig VV saved.")


# ══════════════════════════════════════════════════════════════
# FIGURE WW: Aggregate embedding trajectory by group + quantitative
#             comparison of methods
# ══════════════════════════════════════════════════════════════
print("Generating Figure WW: Group trajectories + method comparison...")

fig = plt.figure(figsize=(22, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Top row: group-averaged embedding trajectories (like Fig NN but with embeddings)
for g_idx, gruppe in enumerate(['I', 'II', 'III']):
    ax = fig.add_subplot(gs[0, g_idx])
    g_texts = [nm for nm in text_names if GROUP_MAP[nm] == gruppe]

    common_pos = np.linspace(0.05, 0.95, 40)
    all_prac = []
    all_theo = []
    all_cosmo = []
    all_color = []

    for nm in g_texts:
        profile = text_profiles[nm]
        pos = np.array([p['pos'] for p in profile])
        prac = np.array([p['practical'] for p in profile])
        theo = np.array([p['theoretical'] for p in profile])
        cosmo = np.array([p['cosmological'] for p in profile])
        color = np.array([p['color_stage'] for p in profile])

        all_prac.append(np.interp(common_pos, pos, prac))
        all_theo.append(np.interp(common_pos, pos, theo))
        all_cosmo.append(np.interp(common_pos, pos, cosmo))
        all_color.append(np.interp(common_pos, pos, color))

        ax.plot(pos, prac, color='#3498db', lw=0.5, alpha=0.2)
        ax.plot(pos, theo, color='#e74c3c', lw=0.5, alpha=0.2)

    mean_prac = np.mean(all_prac, axis=0)
    mean_theo = np.mean(all_theo, axis=0)
    mean_cosmo = np.mean(all_cosmo, axis=0)
    mean_color = np.mean(all_color, axis=0)
    std_prac = np.std(all_prac, axis=0)
    std_theo = np.std(all_theo, axis=0)

    ax.plot(common_pos, mean_prac, color='#3498db', lw=3, label='Practical')
    ax.fill_between(common_pos, mean_prac - std_prac, mean_prac + std_prac,
                    color='#3498db', alpha=0.15)
    ax.plot(common_pos, mean_theo, color='#e74c3c', lw=3, label='Theoretical')
    ax.fill_between(common_pos, mean_theo - std_theo, mean_theo + std_theo,
                    color='#e74c3c', alpha=0.15)
    ax.plot(common_pos, mean_cosmo, color='#e67e22', lw=2, ls='--', label='Cosmological')
    ax.plot(common_pos, mean_color, color='#9b59b6', lw=2, ls=':', label='Color stages')

    ax.axvspan(0.75, 1.0, alpha=0.08, color='grey')
    ax.set_title(f"Gruppe {gruppe} ({len(g_texts)} texts)",
                 fontsize=12, fontweight='bold', color=GROUP_COLORS[gruppe])
    ax.set_xlabel('Text position', fontsize=10)
    if g_idx == 0:
        ax.set_ylabel('Cosine similarity to pole', fontsize=10)
    ax.legend(fontsize=8, loc='best')

# Bottom-left: correlation scatter (embedding dist vs expert dist)
ax_scatter = fig.add_subplot(gs[1, 0])
ax_scatter.scatter(emb_flat, expert_flat, alpha=0.4, s=20, color='#34495e')
z = np.polyfit(emb_flat, expert_flat, 1)
p_line = np.poly1d(z)
x_range = np.linspace(min(emb_flat), max(emb_flat), 100)
ax_scatter.plot(x_range, p_line(x_range), color='#e74c3c', lw=2, ls='--')
ax_scatter.set_xlabel('Embedding distance (cosine)', fontsize=10)
ax_scatter.set_ylabel('Expert distance (Jaccard)', fontsize=10)
ax_scatter.set_title(f'Embedding vs Expert Distance\nr={r_emb_expert:.3f}, rho={rho_emb_expert:.3f}',
                     fontsize=12, fontweight='bold')

# Bottom-center: early vs late half — does embedding similarity change?
ax_shift = fig.add_subplot(gs[1, 1])

early_mean_sim = np.mean(early_dist[np.triu_indices(n, k=1)])
late_mean_sim = np.mean(late_dist[np.triu_indices(n, k=1)])

# Per-group analysis
for g in ['I', 'II', 'III']:
    g_idx_list = [i for i, nm in enumerate(text_names) if GROUP_MAP[nm] == g]
    early_within = [early_dist[i, j] for i in g_idx_list for j in g_idx_list if i < j]
    late_within = [late_dist[i, j] for i in g_idx_list for j in g_idx_list if i < j]
    if early_within and late_within:
        ax_shift.scatter([np.mean(early_within)], [np.mean(late_within)],
                         c=GROUP_COLORS[g], s=200, zorder=5,
                         edgecolors='black', linewidths=1.5,
                         label=f'Gruppe {g} (within)')

# Between groups
for g1_idx, g1 in enumerate(['I', 'II', 'III']):
    for g2 in ['I', 'II', 'III']:
        if g1 >= g2:
            continue
        g1_list = [i for i, nm in enumerate(text_names) if GROUP_MAP[nm] == g1]
        g2_list = [i for i, nm in enumerate(text_names) if GROUP_MAP[nm] == g2]
        early_between = [early_dist[i, j] for i in g1_list for j in g2_list]
        late_between = [late_dist[i, j] for i in g1_list for j in g2_list]
        if early_between and late_between:
            ax_shift.scatter([np.mean(early_between)], [np.mean(late_between)],
                             c='grey', s=80, alpha=0.5, marker='x', zorder=4)

max_val = max(ax_shift.get_xlim()[1], ax_shift.get_ylim()[1])
ax_shift.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
ax_shift.set_xlabel('Mean distance (early half)', fontsize=10)
ax_shift.set_ylabel('Mean distance (late half)', fontsize=10)
ax_shift.set_title('Do Texts Converge or Diverge\nin Their Late Halves?',
                   fontsize=12, fontweight='bold')
ax_shift.legend(fontsize=9)
ax_shift.text(0.95, 0.05, 'above diagonal =\nmore different in\nlate half',
              transform=ax_shift.transAxes, fontsize=8, ha='right', va='bottom',
              color='grey')

# Bottom-right: method comparison table as text
ax_table = fig.add_subplot(gs[1, 2])
ax_table.axis('off')

# Compute additional correlations
# Late-half embedding vs expert
late_common = late_dist[np.ix_(common_idx, common_idx)]
late_flat = upper_tri(late_common)
r_late, _ = pearsonr(late_flat, expert_flat)
rho_late, _ = spearmanr(late_flat, expert_flat)

# Early-half embedding vs expert
early_common = early_dist[np.ix_(common_idx, common_idx)]
early_flat = upper_tri(early_common)
r_early, _ = pearsonr(early_flat, expert_flat)
rho_early, _ = spearmanr(early_flat, expert_flat)

table_text = (
    "Method Comparison Against Expert\n"
    "─────────────────────────────────────\n"
    f"Embedding (full text):\n"
    f"  r = {r_emb_expert:.3f}, rho = {rho_emb_expert:.3f}\n\n"
    f"Embedding (early half):\n"
    f"  r = {r_early:.3f}, rho = {rho_early:.3f}\n\n"
    f"Embedding (late half):\n"
    f"  r = {r_late:.3f}, rho = {rho_late:.3f}\n\n"
    f"Proxy pipeline (1489 chars):\n"
    f"  r = 0.844, rho = 0.882\n\n"
    f"Quadratic Delta (300 MFW):\n"
    f"  r = 0.731, rho = 0.763\n\n"
    f"Word-list (practical + theoretical):\n"
    f"  coverage = 4.1% of words\n"
    f"  128 practical + 104 theoretical forms"
)
ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes,
              fontsize=11, fontfamily='monospace', va='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle("Embedding Analysis: Group Trajectories and Method Comparison",
             fontsize=15, fontweight='bold')
plt.savefig(OUT_DIR / 'processus_figWW_embedding_summary.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig WW saved.")


# ══════════════════════════════════════════════════════════════
# Print final summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EMBEDDING ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
Model: paraphrase-multilingual-MiniLM-L12-v2
  (384-dim embeddings, trained on 50+ languages including German)

Reference poles:
  Practical chemistry:    {len(PRACTICAL_ANCHORS)} anchor passages
  Theoretical/transmutation: {len(THEORETICAL_ANCHORS)} anchor passages
  Cosmological preamble:  {len(COSMOLOGICAL_ANCHORS)} anchor passages
  Color stages:           {len(COLOR_STAGE_ANCHORS)} anchor passages

Pole separation (cosine similarity between centroids):
  Practical ↔ Theoretical:  {cosine_sim(prac_centroid, theo_centroid):.3f}
  Practical ↔ Cosmological: {cosine_sim(prac_centroid, cosmo_centroid):.3f}
  Theoretical ↔ Cosmological: {cosine_sim(theo_centroid, cosmo_centroid):.3f}
  Practical ↔ Color stages: {cosine_sim(prac_centroid, color_centroid):.3f}

Distance correlations with expert annotations:
  Embedding (full text):   r = {r_emb_expert:.3f}, rho = {rho_emb_expert:.3f}
  Embedding (early half):  r = {r_early:.3f}, rho = {rho_early:.3f}
  Embedding (late half):   r = {r_late:.3f}, rho = {rho_late:.3f}
  Proxy pipeline:          r = 0.844, rho = 0.882
  Quadratic Delta:         r = 0.731, rho = 0.763

Figures:
  Fig SS: processus_figSS_embedding_trajectories.png
  Fig TT: processus_figTT_wordlist_vs_embedding.png
  Fig UU: processus_figUU_embedding_dendrograms.png
  Fig VV: processus_figVV_ambiguous_words.png
  Fig WW: processus_figWW_embedding_summary.png
""")
