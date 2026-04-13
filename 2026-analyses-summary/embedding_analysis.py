#!/usr/bin/env python3

import re
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm as np_norm
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to download punkt
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# === CONFIG ===
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
XML_PATH = Path("../sammlung_aller_texte.xml")
TXT_DIR = Path("../2026-analyses/processus_prev_work/processus_universalis-main/ProcessusUniversalis_relevant-files-for-2025/txt-files-lowercase_processus")

model = SentenceTransformer(MODEL_NAME)

# === HELPERS ===
def clean_text(text):
    text = re.sub(r'\[\w+:.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_from_xml(root):
    texts = {}
    for div in root.findall(".//div"):
        div_type = div.get("type", "")
        m = re.search(r'a(\d+)', div_type)
        if not m: continue
        key = 'a' + m.group(1)
        content_parts = []
        for elem in div.iter():
            if elem.text:
                txt = elem.text.strip()
                if txt:
                    if elem.tag == 'head' and not txt.endswith(('.', '!', '?')): txt += '.'
                    content_parts.append(txt)
            if elem.tail:
                tail = elem.tail.strip()
                if tail: content_parts.append(tail)
        if key in A_TO_E:
            texts[A_TO_E[key]] = clean_text(" ".join(content_parts))
    return texts

def split_sentences(text):
    text = re.sub(r'\[CZ:\s*(.*?)\]', r'\1', text)
    sentences = sent_tokenize(text, language='german')
    return [s.strip() for s in sentences if len(s.split()) > 3]

def chunk_embed_official(text, cs=80, ov=40):
    words = text.split()
    chunks = [' '.join(words[i:i+cs]) for i in range(0, max(1, len(words)-cs+1), cs-ov)]
    if not chunks: chunks = [text]
    return chunks

def cosine_sim(a, b):
    return np.dot(a, b) / (np_norm(a) * np_norm(b) + 1e-10)

def upper_tri(mat):
    return mat[np.triu_indices(len(mat), k=1)]

# === DISTANCE METHODS ===
def get_mean_pooling_dist(doc_embs, names):
    n = len(names); dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_sim(doc_embs[names[i]], doc_embs[names[j]])
            dist[i, j] = dist[j, i] = max(0, 1 - sim)
    return dist

def get_filtered_alignment_dist(unit_embs_dict, names, top_p=0.3):
    n = len(names); dist = np.zeros((n, n)); key_embs = {}
    for nm in names:
        embs = unit_embs_dict[nm]
        sim_matrix = cosine_similarity(embs, embs)
        scores = np.mean(sim_matrix, axis=1)
        n_keep = max(2, int(len(embs) * top_p))
        top_indices = np.argsort(scores)[-n_keep:]
        key_embs[nm] = embs[top_indices]
    for i in range(n):
        for j in range(i+1, n):
            ea, eb = key_embs[names[i]], key_embs[names[j]]
            sim_matrix = cosine_similarity(ea, eb)
            max_a, max_b = np.mean(np.max(sim_matrix, axis=1)), np.mean(np.max(sim_matrix, axis=0))
            dist[i, j] = dist[j, i] = max(0, 1 - (max_a + max_b) / 2)
    return dist

def get_chamfer_dist(unit_embs_dict, names):
    n = len(names); dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            ea, eb = unit_embs_dict[names[i]], unit_embs_dict[names[j]]
            sim_matrix = cosine_similarity(ea, eb)
            max_a, max_b = np.mean(np.max(sim_matrix, axis=1)), np.mean(np.max(sim_matrix, axis=0))
            dist[i, j] = dist[j, i] = max(0, 1 - (max_a + max_b) / 2)
    return dist

def get_top_k_dist(unit_embs_dict, names, k=10):
    n = len(names); dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            ea, eb = unit_embs_dict[names[i]], unit_embs_dict[names[j]]
            sim_matrix = cosine_similarity(ea, eb)
            flat_sims = sim_matrix.flatten()
            actual_k = min(k, len(flat_sims))
            top_k_sims = np.sort(flat_sims)[-actual_k:]
            dist[i, j] = dist[j, i] = max(0, 1 - np.mean(top_k_sims))
    return dist

def get_emd_dist(unit_embs_dict, names):
    n = len(names); dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            ea, eb = unit_embs_dict[names[i]], unit_embs_dict[names[j]]
            sim_mat = cosine_similarity(ea, eb)
            cost_matrix = np.clip(1 - sim_mat, 0, None)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            dist[i, j] = dist[j, i] = cost_matrix[row_ind, col_ind].sum() / len(row_ind)
    return dist

def evaluate_nn(dist_embedding, dist_expert, common_names, top_k=1):
    n = len(common_names)
    hits = 0
    emb_m, exp_m = dist_embedding.copy(), dist_expert.copy()
    np.fill_diagonal(emb_m, np.inf); np.fill_diagonal(exp_m, np.inf)
    for i in range(n):
        expert_neighbors = np.argsort(exp_m[i])[:top_k]
        if np.argmin(emb_m[i]) in expert_neighbors: hits += 1
    return hits, n

def print_spectrum(unit_embs, units_dict, names, n_per_tier=4):
    all_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sim_matrix = cosine_similarity(unit_embs[a], unit_embs[b])
            for ia in range(sim_matrix.shape[0]):
                for ib in range(sim_matrix.shape[1]):
                    all_pairs.append((sim_matrix[ia, ib], a, b, units_dict[a][ia], units_dict[b][ib]))
    all_pairs.sort(reverse=True, key=lambda x: x[0])
    tiers = [("EXCELLENT (0.95-1.00)", 0.95, 1.01), ("STRONG (0.85-0.95)", 0.85, 0.95), 
             ("MODERATE (0.70-0.85)", 0.70, 0.85), ("SLIGHT (0.50-0.70)", 0.50, 0.70)]
    print("\n" + "="*80 + "\nQUALITATIVE UNIT-MATCH SPECTRUM\n" + "="*80)
    seen = set()
    for label, low, high in tiers:
        print(f"\n--- {label} ---")
        tier_pairs = [p for p in all_pairs if low <= p[0] < high]
        if not tier_pairs: continue
        indices = np.linspace(0, len(tier_pairs)-1, n_per_tier, dtype=int)
        for idx in indices:
            sim, a, b, sa, sb = tier_pairs[idx]
            if sa[:40] in seen: continue
            seen.add(sa[:40])
            print(f"[{sim:.3f}] {a} ↔ {b}\n  A: {sa[:120]}...\n  B: {sb[:120]}...")

import sys

# Define a class to write to both console and file
class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

# Redirect stdout
sys.stdout = Tee("embedding_analysis_output.txt")
tree = ET.parse(XML_PATH); root = tree.getroot()
A_TO_E = {'a1': 'E16','a2': 'E37','a3': 'E38','a4': 'E44','a5': 'E17','a6': 'E19','a7': 'E39','a8': 'E34',
          'a9': 'E2','a12': 'E45','a13': 'E42','a15': 'E32b','a16': 'E27','a21': 'E3','a22': 'E35','a25': 'E22','a26': 'E11'}
anno_features = {}
for div in root.findall('.//div'):
    m = re.search(r'a(\d+)', div.get('type', ''))
    if not m: continue
    ename = A_TO_E.get('a' + m.group(1))
    if not ename: continue
    features = set()
    for keys_el in div.findall('.//keys'):
        vals = keys_el.get('n', '')
        ktype = keys_el.get('type', '')
        if vals and 'FEHLT' not in vals:
            for v in vals.split(';'):
                v = v.strip()
                if v: features.add(f"{ktype}::{v}")
    if features: anno_features[ename] = features

baseline_texts = {}
for f in sorted(TXT_DIR.glob("*.txt")):
    m = re.search(r'(E\d+[a-z]?)', f.stem)
    if m:
        ename = m.group(1)
        if ename in A_TO_E.values():
            baseline_texts[ename] = f.read_text(encoding='utf-8', errors='replace').strip()

raw_units = extract_from_xml(root)
common = sorted([nm for nm in raw_units.keys() if nm in anno_features and nm in baseline_texts])
nc = len(common)
expert_dist = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        a, b = anno_features[common[i]], anno_features[common[j]]
        expert_dist[i, j] = expert_dist[j, i] = 1 - len(a & b) / len(a | b)

print(f"\n" + "#"*80 + f"\n### COMPARATIVE REPORT\n" + "#"*80)
print("Methodological Overview:")
print("- Mean-Pool: Reduces documents to a single centroid vector by averaging all sentence embeddings.")
print("- Filtered-Align: Selects the most representative (central) sentences and computes distance via Chamfer-style alignment.")
print("- Chamfer: Computes the average of maximum sentence-to-sentence similarities for all sentence pairs.")
print("- Top-K (10): Averages the 10 strongest sentence-to-sentence pairings to isolate core shared procedural logic.")
print("- EMD/OT: Uses bipartite matching (linear sum assignment) to find the globally optimal alignment cost.")

# RUN BASELINE
doc_embs_b = {nm: np.mean(model.encode(chunk_embed_official(baseline_texts[nm])), axis=0) for nm in common}
dist_b = get_mean_pooling_dist(doc_embs_b, common)
r_b, rho_b = pearsonr(upper_tri(dist_b), upper_tri(expert_dist))[0], spearmanr(upper_tri(dist_b), upper_tri(expert_dist))[0]
h1_b, tot = evaluate_nn(dist_b, expert_dist, common, 1)
h3_b, _ = evaluate_nn(dist_b, expert_dist, common, 3)
print(f"\nBASELINE (Mean-Pool) | Pearson: {r_b:.3f} | Spearman: {rho_b:.3f} | NN-1: {h1_b}/{tot} | NN-3: {h3_b}/{tot}")

# RUN IMPROVED
doc_embs_i, unit_embs_i, units_dict_i = {}, {}, {}
for nm in common:
    units = split_sentences(raw_units[nm])
    if not units: units = [raw_units[nm]]
    unit_embs_i[nm] = model.encode(units)
    units_dict_i[nm] = units

print(f"\n  {'Method':<14} | Pearson | Spearman | NN Top-1  | NN Top-3")
print(f"  {'-'*65}")
methods = {
    'Mean-Pool': get_mean_pooling_dist({nm: np.mean(e, axis=0) for nm, e in unit_embs_i.items()}, common),
    'Filtered-Align': get_filtered_alignment_dist(unit_embs_i, common, top_p=0.3),
    'Chamfer': get_chamfer_dist(unit_embs_i, common), 
    'Top-K (10)': get_top_k_dist(unit_embs_i, common, k=10),
    'EMD/OT': get_emd_dist(unit_embs_i, common)
}
for m_name, dist_all in methods.items():
    r, _ = pearsonr(upper_tri(dist_all), upper_tri(expert_dist))
    rho, _ = spearmanr(upper_tri(dist_all), upper_tri(expert_dist))
    h1, tot = evaluate_nn(dist_all, expert_dist, common, 1)
    h3, _ = evaluate_nn(dist_all, expert_dist, common, 3)
    print(f"  {m_name:<14} | {r:.3f}   | {rho:.3f}    | {h1:>2}/{tot}     | {h3:>2}/{tot}")

print_spectrum(unit_embs_i, units_dict_i, common)
