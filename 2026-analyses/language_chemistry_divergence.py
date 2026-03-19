#!/usr/bin/env python3
"""
Language vs Chemistry Divergence Analysis
=========================================
Explores where alchemical recipe texts shift from practical chemistry
to theoretical/philosophical content — especially towards the end,
where claims about the philosopher's stone diverge from executable chemistry.

Produces Figures LL through QQ.
"""

import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

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

# ── Vocabulary categories ──

# PRACTICAL CHEMISTRY: substances, equipment, operations with specific quantities
# These are words that describe what you'd actually DO in a laboratory
CHEM_PRACTICAL = {
    # Equipment
    'retorte', 'retorten', 'kolben', 'alembic', 'alembico', 'alembicum',
    'helm', 'tiegel', 'recipiente', 'recipienten', 'vorlage',
    'ofen', 'capelle', 'capellen', 'mörser',
    'zapfloch', 'zapflöchern', 'gießtiegel', 'schmelztiegel', 'schmeltz-tiegel',
    # Substances (specific, measurable)
    'salpeter', 'vitriol', 'vitrioli', 'antimon', 'antimonium',
    'phlegma', 'phlegmata', 'lauge', 'laugen', 'asche', 'aschen',
    'regenwaßer', 'regenwasser', 'kalk',
    # Operations (concrete lab actions)
    'destilliren', 'destilliren', 'destilliert', 'destillation', 'destillationes',
    'destillir', 'destillire', 'destillando', 'destilliret', 'destilliert',
    'distilliren', 'distillirt', 'distillando', 'distillation',
    'filtriren', 'filtrir', 'filtrire', 'filtra', 'filtrationem', 'filtrieren',
    'filtriert', 'filtration',
    'calciniren', 'calcinir', 'calcinire', 'calciniert', 'calcinierte',
    'calcinierten', 'calcinata', 'calcinirten',
    'sublimiren', 'sublimirt', 'sublima', 'sublimiere', 'sublimiren',
    'sublimiert', 'sublimierte', 'sublimirte',
    'solviren', 'solvire', 'solviert', 'solvieren', 'solve', 'solution',
    'coaguliren', 'coagulire', 'coagula', 'coaguliren', 'coaguliert',
    'evaporiren', 'evaporir', 'evaporire', 'evaporiren',
    'rectificiren', 'rectificirt', 'rectificiret',
    'sieden', 'siedet', 'einsieden', 'eingesotten',
    'laugen', 'auslaugen', 'ausgelauget',
    'schmelzen', 'schmeltzen', 'geschmolzen', 'schmeltzet',
    'glüen', 'erglüen', 'erglüe', 'ausgeglüet', 'glüend', 'glüenden',
    'trocknen', 'getrocknet',
    # Measurements
    'pfund', 'lb', 'loth', 'gran', 'maas', 'untz', 'untzen',
    'zentner',
    # Practical descriptors
    'verlutirt', 'verlutiert', 'verlutiren', 'lutirt', 'lutierte', 'lutier',
    'lutiret', 'lutiert',
    'herüber', 'herübergehen', 'herübergegangen',
    'anschießen', 'anschiesen', 'angeschossen',
    'abziehen', 'abdestillirt', 'abtreuflen',
}

# THEORETICAL/PHILOSOPHICAL: cosmological claims, aspirational language,
# references to the philosopher's stone tradition
CHEM_THEORETICAL = {
    # Alchemical philosophy / tradition
    'philosophorum', 'philosophisch', 'philosophische', 'philosophischen',
    'philosophischer', 'philosophico', 'philosophicum',
    'tinctur', 'tinctura', 'tincturam', 'tincturae', 'tingirt', 'tingiren',
    'tingirende', 'tingiret',
    'lapis', 'lapidem',
    'quintum', 'esse',
    'arcanum', 'arcani', 'secretum', 'secreti',
    # Cosmological / nature philosophy
    'himmlisch', 'himmlische', 'himmlischen', 'himmlischer', 'himlischen',
    'himlische', 'himlischer',
    'gestirn', 'gestirne',
    'einflüsse', 'einflüße', 'influenzen',
    'elementen', 'elemento',
    'centrum', 'centro',
    'fundamentum', 'fundament',
    'schöpfer', 'schöpfers',
    'prima', 'materia',
    # Religious / devotional
    'amen', 'gloria', 'laus', 'deo', 'soli',
    # Grand claims about transmutation
    'transmutation', 'verwandeln', 'verwandelt',
    'multiplication', 'multipliciren', 'multipliciret', 'multiplicatio',
    'multipliziren', 'augmentiren', 'augmentatio',
    'projection', 'projectionis',
    'fermentatio', 'fermentation', 'fermentiren', 'fermentierten', 'fermentirt',
    # Claims of universal power
    'universale', 'universalem', 'universalis', 'universal',
    'universalwassers',
    'kranckheiten', 'krankheiten',
    'gesundheit',
    'reichtumb', 'reichtumbs', 'reichthumb',
    # Aspirational / superlative
    'allerhöchste', 'allerhöchsten', 'alleredelste', 'allerfeinste',
    'allerköstlichste', 'allerreinste', 'allerherrlichste',
    'unaussprechlich', 'unaussprechliches',
    'unendlich', 'infinitum',
    'königlich', 'königliche', 'königlichen',
    # Alchemical stage names (philosophical, not practical observations)
    'putrefaction', 'putrefactio', 'putrefactionen',
    'nigredo',
    'albedo',
    'citrinitas',
    'rubedo',
    # Metaphorical language
    'wiedergeburth', 'wiedergeburt',
    'verjüngen',
    'auferstehen',
    'lebendiges',
}

# PROCEDURAL MARKERS: action verbs that indicate recipe steps
# These are neutral — they appear throughout but their density signals
# "recipe mode" vs "discourse mode"
PROCEDURAL = {
    'nimm', 'nimb', 'nehmet', 'nehme', 'nehmen', 'genommen',
    'thu', 'thut', 'thue', 'thuet', 'gethan',
    'setz', 'setzt', 'setze', 'setzet', 'gesetzt',
    'gib', 'giebt', 'gieb', 'gebt', 'gibet', 'geben', 'gegeben',
    'laß', 'laßt', 'laßet', 'lasset', 'lasst', 'lassen',
    'mach', 'macht', 'mache', 'machet',
    'geuß', 'gießet', 'gießen', 'gegossen', 'gegoßen',
    'recipe', '[recipe]',
    'procedir', 'procede', 'procedire', 'procediren', 'procedirt',
}

# COLOR-STAGE terms (the philosophical stages observed in the athanor)
# These are interesting because they sit between practical observation
# and philosophical tradition
COLOR_STAGES = {
    'schwartze', 'schwärtze', 'schwartz',  # nigredo
    'weisse', 'weise', 'weiss', 'weiß',    # albedo (in color context)
    'gelbe', 'gelb',                         # citrinitas
    'rothe', 'roth', 'röthe', 'rubinroth', 'blutroth', 'rubinfarbenes',  # rubedo
    'grüne', 'grün',                         # viriditas
    'farben', 'farbe',                       # color changes generally
}


def load_texts():
    """Load all texts, return dict of {name: word_list}."""
    texts = {}
    for fp in sorted(TXT_DIR.glob("*.txt")):
        fname = fp.stem
        m = re.search(r'(E\d+[a-z]?)', fname)
        if not m:
            continue
        ename = m.group(1)
        if ename not in GROUP_MAP:
            continue
        raw = fp.read_text(encoding='utf-8', errors='replace').lower()
        words = re.findall(r'[a-zäöüß\-]+', raw)
        texts[ename] = words
    return texts


def classify_word(w):
    """Classify a single word. Returns category string or None."""
    if w in CHEM_PRACTICAL:
        return 'practical'
    if w in CHEM_THEORETICAL:
        return 'theoretical'
    if w in PROCEDURAL:
        return 'procedural'
    if w in COLOR_STAGES:
        return 'color_stage'
    return None


def windowed_analysis(words, n_windows=20):
    """Compute category densities across text position using sliding windows."""
    n = len(words)
    if n < 50:
        return None

    window_size = max(50, n // n_windows)
    step = max(1, (n - window_size) // (n_windows - 1)) if n_windows > 1 else n

    results = []
    for i in range(0, n - window_size + 1, step):
        chunk = words[i:i + window_size]
        counts = Counter(classify_word(w) for w in chunk)
        total = len(chunk)
        pos = (i + window_size / 2) / n  # midpoint position [0, 1]
        results.append({
            'pos': pos,
            'practical': counts.get('practical', 0) / total,
            'theoretical': counts.get('theoretical', 0) / total,
            'procedural': counts.get('procedural', 0) / total,
            'color_stage': counts.get('color_stage', 0) / total,
            'practical_n': counts.get('practical', 0),
            'theoretical_n': counts.get('theoretical', 0),
            'procedural_n': counts.get('procedural', 0),
            'color_stage_n': counts.get('color_stage', 0),
        })
    return results


def segment_analysis(words, n_segments=5):
    """Divide text into equal segments and count categories."""
    n = len(words)
    seg_size = n // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_size
        end = (i + 1) * seg_size if i < n_segments - 1 else n
        chunk = words[start:end]
        counts = Counter(classify_word(w) for w in chunk)
        total = len(chunk)
        segments.append({
            'practical': counts.get('practical', 0) / total,
            'theoretical': counts.get('theoretical', 0) / total,
            'procedural': counts.get('procedural', 0) / total,
            'color_stage': counts.get('color_stage', 0) / total,
            'practical_n': counts.get('practical', 0),
            'theoretical_n': counts.get('theoretical', 0),
            'total': total,
        })
    return segments


def find_transition_point(words, window_pct=0.10):
    """Find where theoretical density *sustainably* exceeds practical density.

    Many texts have a cosmological/philosophical preamble before the recipe
    proper begins, so we look for the LAST sustained crossover (after position
    0.30) where theoretical overtakes practical and stays dominant.
    If no late crossover exists, we check if the text ends theoretical
    (last 20%) even without a clear crossover.
    """
    n = len(words)
    w = max(30, int(n * window_pct))
    step = max(1, w // 4)

    # Compute practical - theoretical balance at each position
    positions = []
    balance = []  # positive = more practical, negative = more theoretical
    for i in range(0, n - w, step):
        chunk = words[i:i + w]
        counts = Counter(classify_word(wd) for wd in chunk)
        prac = counts.get('practical', 0)
        theo = counts.get('theoretical', 0)
        pos = (i + w / 2) / n
        positions.append(pos)
        balance.append(prac - theo)

    if not positions:
        return 1.0

    # Find the last crossover from positive to negative after pos 0.30
    # (skip the opening preamble)
    last_crossover = None
    for idx in range(len(balance) - 1):
        if positions[idx] < 0.30:
            continue
        if balance[idx] >= 0 and balance[idx + 1] < 0:
            last_crossover = positions[idx]

    if last_crossover is not None:
        return last_crossover

    # Check if the final section is dominantly theoretical even without
    # a clear crossover
    late_start = int(n * 0.80)
    late = words[late_start:]
    counts = Counter(classify_word(w) for w in late)
    if counts.get('theoretical', 0) > counts.get('practical', 0) * 1.5:
        # Find where the shift begins in the last half
        for idx in range(len(balance) - 1, -1, -1):
            if positions[idx] < 0.50:
                break
            if balance[idx] >= 0:
                return positions[idx]

    return 1.0  # never transitions


def get_late_vocabulary(words, late_start=0.75):
    """Extract classified vocabulary from the last portion of text."""
    n = len(words)
    start = int(n * late_start)
    late_words = words[start:]
    practical_words = [w for w in late_words if classify_word(w) == 'practical']
    theoretical_words = [w for w in late_words if classify_word(w) == 'theoretical']
    color_words = [w for w in late_words if classify_word(w) == 'color_stage']
    return practical_words, theoretical_words, color_words


# ══════════════════════════════════════════════════════════════
print("Loading texts...")
texts = load_texts()
print(f"Loaded {len(texts)} texts")

# Sort texts for consistent display
text_names = sorted(texts.keys(), key=lambda x: (GROUP_MAP[x], x))

# ══════════════════════════════════════════════════════════════
# Basic statistics
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VOCABULARY CLASSIFICATION OVERVIEW")
print("=" * 70)

for nm in text_names:
    words = texts[nm]
    counts = Counter(classify_word(w) for w in words)
    total = len(words)
    prac = counts.get('practical', 0)
    theo = counts.get('theoretical', 0)
    proc = counts.get('procedural', 0)
    color = counts.get('color_stage', 0)
    ratio = prac / theo if theo > 0 else float('inf')
    tp = find_transition_point(words)
    print(f"  {nm:5s} ({GROUP_MAP[nm]:>3s}): {total:5d} words | "
          f"practical={prac:3d} ({100*prac/total:.1f}%) "
          f"theoretical={theo:3d} ({100*theo/total:.1f}%) "
          f"procedural={proc:3d} "
          f"color={color:2d} | "
          f"prac/theo={ratio:5.2f} | transition@{tp:.0%}")


# ══════════════════════════════════════════════════════════════
# FIGURE LL: Per-text language trajectory (practical vs theoretical
#            across text position)
# ══════════════════════════════════════════════════════════════
print("\nGenerating Figure LL: Language trajectories...")

fig, axes = plt.subplots(4, 5, figsize=(24, 18), sharey=True)
axes_flat = axes.flatten()

# Hide unused axes
for ax in axes_flat[len(text_names):]:
    ax.set_visible(False)

for idx, nm in enumerate(text_names):
    ax = axes_flat[idx]
    words = texts[nm]
    wa = windowed_analysis(words, n_windows=25)
    if wa is None:
        ax.text(0.5, 0.5, 'Too short', ha='center', va='center', transform=ax.transAxes)
        continue

    positions = [r['pos'] for r in wa]
    prac_density = [r['practical'] * 100 for r in wa]
    theo_density = [r['theoretical'] * 100 for r in wa]
    color_density = [r['color_stage'] * 100 for r in wa]
    proc_density = [r['procedural'] * 100 for r in wa]

    ax.fill_between(positions, prac_density, alpha=0.3, color='#3498db')
    ax.fill_between(positions, theo_density, alpha=0.3, color='#e74c3c')
    ax.plot(positions, prac_density, color='#3498db', lw=2, label='Practical chemistry')
    ax.plot(positions, theo_density, color='#e74c3c', lw=2, label='Theoretical/philosophical')
    ax.plot(positions, color_density, color='#9b59b6', lw=1.5, ls='--', label='Color stages')

    # Mark transition point
    tp = find_transition_point(words)
    if tp < 1.0:
        ax.axvline(tp, color='#e67e22', ls=':', lw=2, alpha=0.8)
        ax.text(tp, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5,
                f'{tp:.0%}', color='#e67e22', fontsize=8, ha='center', va='top',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                facecolor='white', alpha=0.8))

    # Mark the last 25% region
    ax.axvspan(0.75, 1.0, alpha=0.08, color='grey')

    g = GROUP_MAP[nm]
    ax.set_title(f"{nm} (Gruppe {g})", fontsize=11, fontweight='bold',
                 color=GROUP_COLORS[g])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Text position', fontsize=8)
    if idx % 5 == 0:
        ax.set_ylabel('Term density (%)', fontsize=9)
    ax.tick_params(labelsize=8)

# Shared legend
handles = [
    Line2D([0], [0], color='#3498db', lw=2, label='Practical chemistry'),
    Line2D([0], [0], color='#e74c3c', lw=2, label='Theoretical/philosophical'),
    Line2D([0], [0], color='#9b59b6', lw=1.5, ls='--', label='Color stages'),
    Line2D([0], [0], color='#e67e22', lw=2, ls=':', label='Transition point'),
    Patch(facecolor='grey', alpha=0.15, label='Last 25% of text'),
]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=11,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Language Trajectories: Practical Chemistry vs Theoretical Content\n"
             "across Text Position (sliding window, each panel = one recipe)",
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUT_DIR / 'processus_figLL_language_trajectories.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig LL saved: language trajectories")


# ══════════════════════════════════════════════════════════════
# FIGURE MM: Aggregate heatmap — category density by quintile
# ══════════════════════════════════════════════════════════════
print("Generating Figure MM: Category density heatmap...")

N_SEG = 5
seg_labels = ['0-20%\n(opening)', '20-40%\n(early body)', '40-60%\n(mid body)',
              '60-80%\n(late body)', '80-100%\n(closing)']

# Build matrices
prac_matrix = np.zeros((len(text_names), N_SEG))
theo_matrix = np.zeros((len(text_names), N_SEG))
ratio_matrix = np.zeros((len(text_names), N_SEG))

for i, nm in enumerate(text_names):
    segs = segment_analysis(texts[nm], N_SEG)
    for j, s in enumerate(segs):
        prac_matrix[i, j] = s['practical'] * 100
        theo_matrix[i, j] = s['theoretical'] * 100
        if s['practical'] + s['theoretical'] > 0:
            ratio_matrix[i, j] = (s['theoretical'] - s['practical']) / (s['theoretical'] + s['practical'])
        else:
            ratio_matrix[i, j] = 0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))

# Practical density
im1 = ax1.imshow(prac_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
ax1.set_yticks(range(len(text_names)))
ax1.set_yticklabels([f"{nm} ({GROUP_MAP[nm]})" for nm in text_names], fontsize=9)
for lbl in ax1.get_yticklabels():
    nm = lbl.get_text().split(' ')[0]
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax1.set_xticks(range(N_SEG))
ax1.set_xticklabels(seg_labels, fontsize=8)
ax1.set_title('Practical Chemistry\n(% of words)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='%')

# Theoretical density
im2 = ax2.imshow(theo_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
ax2.set_yticks(range(len(text_names)))
ax2.set_yticklabels([f"{nm} ({GROUP_MAP[nm]})" for nm in text_names], fontsize=9)
for lbl in ax2.get_yticklabels():
    nm = lbl.get_text().split(' ')[0]
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax2.set_xticks(range(N_SEG))
ax2.set_xticklabels(seg_labels, fontsize=8)
ax2.set_title('Theoretical/Philosophical\n(% of words)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='%')

# Balance ratio (positive = more theoretical, negative = more practical)
im3 = ax3.imshow(ratio_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                 vmin=-1, vmax=1)
ax3.set_yticks(range(len(text_names)))
ax3.set_yticklabels([f"{nm} ({GROUP_MAP[nm]})" for nm in text_names], fontsize=9)
for lbl in ax3.get_yticklabels():
    nm = lbl.get_text().split(' ')[0]
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax3.set_xticks(range(N_SEG))
ax3.set_xticklabels(seg_labels, fontsize=8)
ax3.set_title('Balance: Theoretical vs Practical\n(red = more theoretical)', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='theo - prac')

fig.suptitle("Category Density by Text Position (quintiles)\n"
             "How practical vs theoretical language distributes across each recipe",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / 'processus_figMM_category_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig MM saved: category density heatmap")


# ══════════════════════════════════════════════════════════════
# FIGURE NN: Aggregate curves — mean trajectory by Gruppe
# ══════════════════════════════════════════════════════════════
print("Generating Figure NN: Group-average trajectories...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

for g_idx, (gruppe, ax) in enumerate(zip(['I', 'II', 'III'], axes)):
    g_texts = [nm for nm in text_names if GROUP_MAP[nm] == gruppe]

    # Collect all windowed analyses, interpolate to common grid
    common_pos = np.linspace(0.05, 0.95, 50)
    all_prac = []
    all_theo = []
    all_color = []
    all_proc = []

    for nm in g_texts:
        wa = windowed_analysis(texts[nm], n_windows=30)
        if wa is None:
            continue
        pos = np.array([r['pos'] for r in wa])
        prac = np.array([r['practical'] * 100 for r in wa])
        theo = np.array([r['theoretical'] * 100 for r in wa])
        color = np.array([r['color_stage'] * 100 for r in wa])
        proc = np.array([r['procedural'] * 100 for r in wa])

        # Interpolate
        all_prac.append(np.interp(common_pos, pos, prac))
        all_theo.append(np.interp(common_pos, pos, theo))
        all_color.append(np.interp(common_pos, pos, color))
        all_proc.append(np.interp(common_pos, pos, proc))

        # Plot individual traces (thin, transparent)
        ax.plot(pos, prac, color='#3498db', lw=0.5, alpha=0.25)
        ax.plot(pos, theo, color='#e74c3c', lw=0.5, alpha=0.25)

    # Mean curves
    if all_prac:
        mean_prac = np.mean(all_prac, axis=0)
        mean_theo = np.mean(all_theo, axis=0)
        mean_color = np.mean(all_color, axis=0)
        std_prac = np.std(all_prac, axis=0)
        std_theo = np.std(all_theo, axis=0)

        ax.plot(common_pos, mean_prac, color='#3498db', lw=3, label='Practical chemistry')
        ax.fill_between(common_pos, mean_prac - std_prac, mean_prac + std_prac,
                        color='#3498db', alpha=0.15)
        ax.plot(common_pos, mean_theo, color='#e74c3c', lw=3, label='Theoretical/philosophical')
        ax.fill_between(common_pos, mean_theo - std_theo, mean_theo + std_theo,
                        color='#e74c3c', alpha=0.15)
        ax.plot(common_pos, mean_color, color='#9b59b6', lw=2, ls='--',
                label='Color stages')

        # Find where mean crosses
        cross_idx = np.where(np.diff(np.sign(mean_theo - mean_prac)))[0]
        for ci in cross_idx:
            ax.axvline(common_pos[ci], color='#e67e22', ls=':', lw=2, alpha=0.8)

    ax.axvspan(0.75, 1.0, alpha=0.08, color='grey')
    ax.set_title(f"Gruppe {gruppe} ({len(g_texts)} texts)",
                 fontsize=13, fontweight='bold', color=GROUP_COLORS[gruppe])
    ax.set_xlabel('Text position', fontsize=11)
    ax.set_xlim(0, 1)
    if g_idx == 0:
        ax.set_ylabel('Term density (%)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')

fig.suptitle("Group-Averaged Language Trajectories\n"
             "(thin lines = individual texts, thick = group mean, shading = 1 std dev)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_DIR / 'processus_figNN_group_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig NN saved: group-average trajectories")


# ══════════════════════════════════════════════════════════════
# FIGURE OO: Transition point comparison + late-section analysis
# ══════════════════════════════════════════════════════════════
print("Generating Figure OO: Transition points and late-section analysis...")

fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Panel 1: Transition points by text
ax1 = fig.add_subplot(gs[0, 0])
transitions = []
for nm in text_names:
    tp = find_transition_point(texts[nm])
    transitions.append((nm, tp))

y_pos = range(len(transitions))
colors = [GROUP_COLORS[GROUP_MAP[nm]] for nm, _ in transitions]
bars = ax1.barh(y_pos, [tp for _, tp in transitions], color=colors, alpha=0.7,
                edgecolor='white', linewidth=0.5)

# Mark texts that never transition
for i, (nm, tp) in enumerate(transitions):
    if tp >= 1.0:
        ax1.text(0.95, i, 'never', fontsize=8, ha='right', va='center',
                 color='grey', fontstyle='italic')

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{nm} ({GROUP_MAP[nm]})" for nm, _ in transitions], fontsize=9)
for lbl in ax1.get_yticklabels():
    nm = lbl.get_text().split(' ')[0]
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax1.set_xlabel('Position where theoretical > practical', fontsize=10)
ax1.set_title('Transition Point\n(where theory overtakes practice)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1.05)
ax1.axvline(0.75, color='grey', ls='--', alpha=0.5)
ax1.text(0.76, len(transitions) - 0.5, 'last 25%', fontsize=8, color='grey')

# Panel 2: Practical/theoretical ratio in last 25% vs first 75%
ax2 = fig.add_subplot(gs[0, 1])

early_ratios = []
late_ratios = []
for nm in text_names:
    words = texts[nm]
    n = len(words)
    split = int(n * 0.75)
    early = words[:split]
    late = words[split:]

    e_counts = Counter(classify_word(w) for w in early)
    l_counts = Counter(classify_word(w) for w in late)

    e_prac = e_counts.get('practical', 0)
    e_theo = e_counts.get('theoretical', 0)
    l_prac = l_counts.get('practical', 0)
    l_theo = l_counts.get('theoretical', 0)

    e_ratio = e_theo / e_prac if e_prac > 0 else 10
    l_ratio = l_theo / l_prac if l_prac > 0 else 10
    early_ratios.append(e_ratio)
    late_ratios.append(l_ratio)

x = np.arange(len(text_names))
w = 0.35
ax2.bar(x - w/2, early_ratios, w, color='#3498db', alpha=0.7, label='First 75%')
ax2.bar(x + w/2, late_ratios, w, color='#e74c3c', alpha=0.7, label='Last 25%')
ax2.axhline(1.0, color='black', ls='--', lw=1, alpha=0.4)
ax2.text(-0.5, 1.05, 'theoretical = practical', fontsize=8, color='grey')
ax2.set_xticks(x)
ax2.set_xticklabels(text_names, fontsize=8, rotation=45, ha='right')
for i, lbl in enumerate(ax2.get_xticklabels()):
    lbl.set_color(GROUP_COLORS[GROUP_MAP[text_names[i]]])
ax2.set_ylabel('Theoretical / Practical ratio', fontsize=10)
ax2.set_title('Theory-to-Practice Ratio\n(first 75% vs last 25%)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(0, min(12, max(max(early_ratios), max(late_ratios)) * 1.1))

# Panel 3: What's IN the last 25%? Top theoretical terms
ax3 = fig.add_subplot(gs[1, 0])

late_theo_all = Counter()
late_prac_all = Counter()
for nm in text_names:
    prac_w, theo_w, color_w = get_late_vocabulary(texts[nm], 0.75)
    late_theo_all.update(theo_w)
    late_prac_all.update(prac_w)

# Top theoretical terms in late sections
top_theo = late_theo_all.most_common(20)
labels = [w for w, _ in top_theo]
counts = [c for _, c in top_theo]
bars = ax3.barh(range(len(labels)), counts, color='#e74c3c', alpha=0.7)
ax3.set_yticks(range(len(labels)))
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel('Occurrences across all texts (last 25%)', fontsize=10)
ax3.set_title('Most Common Theoretical Terms\nin Late Recipe Sections',
              fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# Panel 4: What's IN the last 25%? Top practical terms
ax4 = fig.add_subplot(gs[1, 1])

top_prac = late_prac_all.most_common(20)
labels = [w for w, _ in top_prac]
counts = [c for _, c in top_prac]
bars = ax4.barh(range(len(labels)), counts, color='#3498db', alpha=0.7)
ax4.set_yticks(range(len(labels)))
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel('Occurrences across all texts (last 25%)', fontsize=10)
ax4.set_title('Most Common Practical Terms\nin Late Recipe Sections',
              fontsize=12, fontweight='bold')
ax4.invert_yaxis()

fig.suptitle("Where Does Theory Take Over from Practice?",
             fontsize=15, fontweight='bold')
plt.savefig(OUT_DIR / 'processus_figOO_transition_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig OO saved: transition analysis")


# ══════════════════════════════════════════════════════════════
# FIGURE PP: Color stage distribution — when do the alchemical
#            stages (nigredo/albedo/citrinitas/rubedo) appear?
# ══════════════════════════════════════════════════════════════
print("Generating Figure PP: Color stage distribution...")

fig, (ax_main, ax_detail) = plt.subplots(2, 1, figsize=(18, 12),
                                          gridspec_kw={'height_ratios': [2, 1]})

# Main panel: position of each color-stage word occurrence
color_map_specific = {
    'schwartze': 'black', 'schwärtze': 'black', 'schwartz': 'black',
    'weisse': '#aaaaaa', 'weise': '#aaaaaa', 'weiss': '#aaaaaa', 'weiß': '#aaaaaa',
    'gelbe': '#DAA520', 'gelb': '#DAA520',
    'grüne': '#2ecc71', 'grün': '#2ecc71',
    'rothe': '#e74c3c', 'roth': '#e74c3c', 'röthe': '#e74c3c',
    'rubinroth': '#c0392b', 'blutroth': '#c0392b', 'rubinfarbenes': '#c0392b',
    'farben': '#9b59b6', 'farbe': '#9b59b6',
}
color_stage_labels = {
    'black': 'Nigredo (black)',
    '#aaaaaa': 'Albedo (white)',
    '#DAA520': 'Citrinitas (yellow)',
    '#2ecc71': 'Viriditas (green)',
    '#e74c3c': 'Rubedo (red)',
    '#c0392b': 'Rubedo (deep red)',
    '#9b59b6': 'Color changes',
}

y_offset = 0
y_ticks = []
y_labels = []
for nm in text_names:
    words = texts[nm]
    n = len(words)
    y_ticks.append(y_offset)
    y_labels.append(nm)

    for i, w in enumerate(words):
        if w in color_map_specific:
            pos = i / n
            c = color_map_specific[w]
            ax_main.scatter(pos, y_offset, c=c, s=30, alpha=0.7, zorder=5,
                            edgecolors='black', linewidths=0.3)
    y_offset += 1

ax_main.set_yticks(y_ticks)
ax_main.set_yticklabels([f"{nm} ({GROUP_MAP[nm]})" for nm in text_names], fontsize=9)
for lbl in ax_main.get_yticklabels():
    nm = lbl.get_text().split(' ')[0]
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm]])
ax_main.set_xlabel('Text position', fontsize=11)
ax_main.set_xlim(-0.02, 1.02)
ax_main.set_ylim(-0.5, len(text_names) - 0.5)
ax_main.invert_yaxis()
ax_main.axvspan(0.75, 1.0, alpha=0.08, color='grey')

# Legend for colors
legend_handles = []
seen = set()
for c, lbl in color_stage_labels.items():
    if c not in seen:
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=c, markersize=8,
                                     markeredgecolor='black', markeredgewidth=0.5,
                                     label=lbl))
        seen.add(c)
ax_main.legend(handles=legend_handles, loc='upper left', fontsize=9, ncol=4,
               bbox_to_anchor=(0, 1.12))
ax_main.set_title('Position of Alchemical Color-Stage Terms in Each Recipe',
                  fontsize=13, fontweight='bold', pad=35)

# Detail panel: histogram of color-stage positions across all texts
all_positions = {'nigredo': [], 'albedo': [], 'citrinitas': [], 'rubedo': [],
                 'viriditas': [], 'farben': []}
stage_to_key = {
    'black': 'nigredo', '#aaaaaa': 'albedo', '#DAA520': 'citrinitas',
    '#2ecc71': 'viriditas', '#e74c3c': 'rubedo', '#c0392b': 'rubedo',
    '#9b59b6': 'farben',
}
for nm in text_names:
    words = texts[nm]
    n = len(words)
    for i, w in enumerate(words):
        if w in color_map_specific:
            c = color_map_specific[w]
            key = stage_to_key[c]
            all_positions[key].append(i / n)

bins = np.linspace(0, 1, 21)
stage_colors = {'nigredo': 'black', 'albedo': '#aaaaaa', 'citrinitas': '#DAA520',
                'viriditas': '#2ecc71', 'rubedo': '#e74c3c', 'farben': '#9b59b6'}
bottom = np.zeros(len(bins) - 1)
for stage in ['nigredo', 'albedo', 'viriditas', 'citrinitas', 'rubedo', 'farben']:
    if all_positions[stage]:
        hist, _ = np.histogram(all_positions[stage], bins=bins)
        ax_detail.bar(bins[:-1], hist, width=0.05, bottom=bottom,
                      color=stage_colors[stage], alpha=0.7,
                      label=stage.capitalize(), align='edge')
        bottom += hist

ax_detail.set_xlabel('Text position', fontsize=11)
ax_detail.set_ylabel('Count', fontsize=11)
ax_detail.set_title('Stacked Histogram: Where Color-Stage Terms Concentrate',
                    fontsize=12, fontweight='bold')
ax_detail.legend(fontsize=9, ncol=6, loc='upper left')
ax_detail.axvspan(0.75, 1.0, alpha=0.08, color='grey')

plt.tight_layout()
plt.savefig(OUT_DIR / 'processus_figPP_color_stages.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig PP saved: color stage distribution")


# ══════════════════════════════════════════════════════════════
# FIGURE QQ: The "chemical plausibility" decline — a composite
#            view of how executable the recipe remains
# ══════════════════════════════════════════════════════════════
print("Generating Figure QQ: Chemical plausibility decline...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel 1: Stacked area — aggregate across all texts
ax1 = axes[0, 0]
N_POS = 40
common_pos = np.linspace(0.05, 0.95, N_POS)
agg_prac = np.zeros(N_POS)
agg_theo = np.zeros(N_POS)
agg_proc = np.zeros(N_POS)
agg_color = np.zeros(N_POS)
n_texts = 0

for nm in text_names:
    wa = windowed_analysis(texts[nm], n_windows=40)
    if wa is None:
        continue
    pos = np.array([r['pos'] for r in wa])
    prac = np.array([r['practical'] * 100 for r in wa])
    theo = np.array([r['theoretical'] * 100 for r in wa])
    proc = np.array([r['procedural'] * 100 for r in wa])
    color = np.array([r['color_stage'] * 100 for r in wa])

    agg_prac += np.interp(common_pos, pos, prac)
    agg_theo += np.interp(common_pos, pos, theo)
    agg_proc += np.interp(common_pos, pos, proc)
    agg_color += np.interp(common_pos, pos, color)
    n_texts += 1

agg_prac /= n_texts
agg_theo /= n_texts
agg_proc /= n_texts
agg_color /= n_texts

ax1.stackplot(common_pos, agg_prac, agg_proc, agg_color, agg_theo,
              labels=['Practical chemistry', 'Procedural markers', 'Color stages',
                      'Theoretical/philosophical'],
              colors=['#3498db', '#95a5a6', '#9b59b6', '#e74c3c'],
              alpha=0.7)
ax1.set_xlabel('Text position', fontsize=11)
ax1.set_ylabel('Mean term density (%)', fontsize=11)
ax1.set_title('Aggregate Composition of Recipe Language\nacross All 17 Texts',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(0, 1)
ax1.axvspan(0.75, 1.0, alpha=0.08, color='grey')

# Panel 2: "Chemical plausibility index" — practical / (practical + theoretical)
ax2 = axes[0, 1]

for nm in text_names:
    wa = windowed_analysis(texts[nm], n_windows=25)
    if wa is None:
        continue
    pos = np.array([r['pos'] for r in wa])
    prac = np.array([r['practical'] for r in wa])
    theo = np.array([r['theoretical'] for r in wa])
    denom = prac + theo
    cpi = np.where(denom > 0, prac / denom, 0.5)

    g = GROUP_MAP[nm]
    ax2.plot(pos, cpi, color=GROUP_COLORS[g], lw=1, alpha=0.4)

# Group means
for gruppe in ['I', 'II', 'III']:
    g_cpis = []
    for nm in text_names:
        if GROUP_MAP[nm] != gruppe:
            continue
        wa = windowed_analysis(texts[nm], n_windows=25)
        if wa is None:
            continue
        pos = np.array([r['pos'] for r in wa])
        prac = np.array([r['practical'] for r in wa])
        theo = np.array([r['theoretical'] for r in wa])
        denom = prac + theo
        cpi = np.where(denom > 0, prac / denom, 0.5)
        g_cpis.append(np.interp(common_pos, pos, cpi))
    if g_cpis:
        mean_cpi = np.mean(g_cpis, axis=0)
        ax2.plot(common_pos, mean_cpi, color=GROUP_COLORS[gruppe], lw=3,
                 label=f'Gruppe {gruppe} mean')

ax2.axhline(0.5, color='black', ls='--', lw=1, alpha=0.4)
ax2.text(0.02, 0.52, 'balanced', fontsize=8, color='grey')
ax2.text(0.02, 0.85, 'more practical', fontsize=8, color='#3498db')
ax2.text(0.02, 0.15, 'more theoretical', fontsize=8, color='#e74c3c')
ax2.set_xlabel('Text position', fontsize=11)
ax2.set_ylabel('Chemical Plausibility Index', fontsize=11)
ax2.set_title('"Chemical Plausibility Index"\npractical / (practical + theoretical)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1.05)
ax2.axvspan(0.75, 1.0, alpha=0.08, color='grey')

# Panel 3: Per-text late-section vocabulary profile (stacked bars)
ax3 = axes[1, 0]

categories_late = []
for nm in text_names:
    prac_w, theo_w, color_w = get_late_vocabulary(texts[nm], 0.75)
    n = len(texts[nm])
    late_n = n - int(n * 0.75)
    categories_late.append({
        'practical': len(prac_w) / late_n * 100,
        'theoretical': len(theo_w) / late_n * 100,
        'color': len(color_w) / late_n * 100,
    })

x = np.arange(len(text_names))
p_vals = [c['practical'] for c in categories_late]
t_vals = [c['theoretical'] for c in categories_late]
c_vals = [c['color'] for c in categories_late]

ax3.bar(x, p_vals, label='Practical', color='#3498db', alpha=0.7)
ax3.bar(x, c_vals, bottom=p_vals, label='Color stages', color='#9b59b6', alpha=0.7)
ax3.bar(x, t_vals, bottom=[p + c for p, c in zip(p_vals, c_vals)],
        label='Theoretical', color='#e74c3c', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(text_names, fontsize=8, rotation=45, ha='right')
for i, lbl in enumerate(ax3.get_xticklabels()):
    lbl.set_color(GROUP_COLORS[GROUP_MAP[text_names[i]]])
ax3.set_ylabel('% of words in last 25%', fontsize=10)
ax3.set_title('What the Last 25% Contains\n(by category)',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)

# Panel 4: The "multiplication claims" — how many texts mention
#           ever-increasing ratios (1:10, 1:100, 1:1000...) and where
ax4 = axes[1, 1]

mult_pattern = re.compile(r'(\d+)\s*(?:theil|theil|teil|part)', re.IGNORECASE)
ratio_claims = []
for nm in text_names:
    words = texts[nm]
    full_text = ' '.join(words)
    n = len(words)

    # Find multiplication/projection claims
    mult_positions = []
    for i, w in enumerate(words):
        if w in ('multiplication', 'multiplicatio', 'multipliciren',
                 'multipliciret', 'multipliziren', 'augmentatio',
                 'fermentatio', 'fermentation', 'projection',
                 'tingirt', 'tingiren', 'tingiret'):
            mult_positions.append(i / n)

    ratio_claims.append({
        'name': nm,
        'positions': mult_positions,
        'count': len(mult_positions),
    })

# Plot as timeline
y_offset = 0
for rc in ratio_claims:
    nm = rc['name']
    g = GROUP_MAP[nm]
    for pos in rc['positions']:
        ax4.scatter(pos, y_offset, c=GROUP_COLORS[g], s=60, alpha=0.8,
                    edgecolors='black', linewidths=0.5, zorder=5)
    if not rc['positions']:
        ax4.text(0.5, y_offset, '(none found)', fontsize=7, ha='center',
                 va='center', color='grey', fontstyle='italic')
    y_offset += 1

ax4.set_yticks(range(len(text_names)))
ax4.set_yticklabels([f"{nm} ({GROUP_MAP[nm]})" for nm in text_names], fontsize=9)
for lbl in ax4.get_yticklabels():
    nm_part = lbl.get_text().split(' ')[0]
    lbl.set_color(GROUP_COLORS[GROUP_MAP[nm_part]])
ax4.set_xlabel('Text position', fontsize=11)
ax4.set_title('Multiplication & Projection Claims\n(where transmutation promises appear)',
              fontsize=12, fontweight='bold')
ax4.set_xlim(-0.02, 1.02)
ax4.invert_yaxis()
ax4.axvspan(0.75, 1.0, alpha=0.08, color='grey')

fig.suptitle("The Decline of Chemical Plausibility Across Recipe Position",
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_DIR / 'processus_figQQ_chemical_plausibility.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Fig QQ saved: chemical plausibility decline")


# ══════════════════════════════════════════════════════════════
# FIGURE RR: Late-section textual convergence — do the endings
#            become more similar to each other than the beginnings?
# ══════════════════════════════════════════════════════════════
print("Generating Figure RR: Late-section convergence...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Compare Jaccard similarity of vocabulary in early vs late sections
# across all text pairs
segments_to_compare = [
    ('First 25%', 0, 0.25),
    ('Middle 50%', 0.25, 0.75),
    ('Last 25%', 0.75, 1.0),
]

for seg_idx, (seg_label, seg_start, seg_end) in enumerate(segments_to_compare):
    ax = axes[seg_idx]

    # Build vocabulary sets for this segment
    vocab_sets = {}
    for nm in text_names:
        words = texts[nm]
        n = len(words)
        start = int(n * seg_start)
        end = int(n * seg_end)
        segment_words = words[start:end]
        # Use content words (not top-50 most common across all texts)
        vocab_sets[nm] = set(segment_words)

    # Compute Jaccard similarity matrix
    n_texts = len(text_names)
    sim_matrix = np.zeros((n_texts, n_texts))
    for i in range(n_texts):
        for j in range(n_texts):
            a = vocab_sets[text_names[i]]
            b = vocab_sets[text_names[j]]
            if len(a | b) > 0:
                sim_matrix[i, j] = len(a & b) / len(a | b)
            else:
                sim_matrix[i, j] = 0

    im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0, vmax=0.6)
    ax.set_xticks(range(n_texts))
    ax.set_xticklabels(text_names, fontsize=7, rotation=90)
    ax.set_yticks(range(n_texts))
    ax.set_yticklabels(text_names, fontsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])
    for lbl in ax.get_yticklabels():
        lbl.set_color(GROUP_COLORS[GROUP_MAP[lbl.get_text()]])

    # Mean similarity (excluding diagonal)
    mask = ~np.eye(n_texts, dtype=bool)
    mean_sim = sim_matrix[mask].mean()
    ax.set_title(f'{seg_label}\n(mean Jaccard = {mean_sim:.3f})',
                 fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Do Recipe Endings Converge? Vocabulary Similarity by Section\n"
             "(higher values = more shared vocabulary between text pairs)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_DIR / 'processus_figRR_section_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig RR saved: late-section convergence")


# ══════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: Language vs Chemistry Divergence")
print("=" * 70)

# Aggregate stats
all_transitions = [find_transition_point(texts[nm]) for nm in text_names]
mean_transition = np.mean([t for t in all_transitions if t < 1.0])
n_transitioning = sum(1 for t in all_transitions if t < 1.0)

# Late section ratios
late_ratios_clean = []
for nm in text_names:
    prac_w, theo_w, _ = get_late_vocabulary(texts[nm], 0.75)
    if len(prac_w) > 0:
        late_ratios_clean.append(len(theo_w) / len(prac_w))
    else:
        late_ratios_clean.append(float('inf'))

print(f"\nTransition point (where theoretical overtakes practical):")
print(f"  Mean: {mean_transition:.0%} (across {n_transitioning}/{len(text_names)} texts that transition)")
print(f"  Texts that never transition: {len(text_names) - n_transitioning}")
for nm in text_names:
    tp = find_transition_point(texts[nm])
    if tp < 1.0:
        print(f"    {nm} ({GROUP_MAP[nm]}): {tp:.0%}")

print(f"\nTheory/Practice ratio in last 25%:")
for i, nm in enumerate(text_names):
    print(f"  {nm} ({GROUP_MAP[nm]}): {late_ratios_clean[i]:.2f}")

# Group averages
for g in ['I', 'II', 'III']:
    g_ratios = [late_ratios_clean[i] for i, nm in enumerate(text_names) if GROUP_MAP[nm] == g and late_ratios_clean[i] < float('inf')]
    if g_ratios:
        print(f"  Gruppe {g} mean: {np.mean(g_ratios):.2f}")

print(f"\nFigures saved:")
print(f"  Fig LL: processus_figLL_language_trajectories.png")
print(f"  Fig MM: processus_figMM_category_heatmap.png")
print(f"  Fig NN: processus_figNN_group_trajectories.png")
print(f"  Fig OO: processus_figOO_transition_analysis.png")
print(f"  Fig PP: processus_figPP_color_stages.png")
print(f"  Fig QQ: processus_figQQ_chemical_plausibility.png")
print(f"  Fig RR: processus_figRR_section_convergence.png")
