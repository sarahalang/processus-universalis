import json
import os
import re
import csv
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../data/atomic_extraction_results.json")
XML_PATH = os.path.join(SCRIPT_DIR, "../../sammlung_aller_texte.xml")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Thresholds to test for sensitivity analysis
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

def load_expert_ground_truth_from_xml():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    
    doc_features = {}
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
        features = set()
        for keys_el in div.findall('.//keys'):
            vals = keys_el.get('n', '')
            ktype = keys_el.get('type', '')
            if vals and 'FEHLT' not in vals:
                for v in vals.split(';'):
                    v = v.strip()
                    if v: features.add(f"{ktype}::{v}")
        if features:
            doc_features[tid] = features

    tids = sorted(list(doc_features.keys()))
    n = len(tids)
    sim_matrix = np.zeros((n, n))
    
    def jaccard(s1, s2):
        if not s1 and not s2: return 0.0
        return len(s1 & s2) / len(s1 | s2)

    for i in range(n):
        for j in range(i+1, n):
            s = jaccard(doc_features[tids[i]], doc_features[tids[j]])
            sim_matrix[i, j] = sim_matrix[j, i] = s
    np.fill_diagonal(sim_matrix, 1.0)
    
    return tids, sim_matrix

def jaccard_sim(set1, set2):
    if not set1 and not set2: return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def evaluate_nn(derived_sim, expert_sim, k=1):
    """
    Measures how often the closest neighbor in the derived matrix 
    is among the top-k neighbors in the expert matrix.
    """
    n = derived_sim.shape[0]
    hits = 0
    
    # higher is closer
    d_sim = derived_sim.copy()
    e_sim = expert_sim.copy()
    np.fill_diagonal(d_sim, -1)
    np.fill_diagonal(e_sim, -1)
    
    for i in range(n):
        llm_neighbor = np.argmax(d_sim[i])
        # Indices of top-k expert similarities
        expert_neighbors = np.argsort(e_sim[i])[-k:]
        if llm_neighbor in expert_neighbors:
            hits += 1
    return hits

def run_evaluation():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading expert ground truth from {XML_PATH}...")
    expert_ids, expert_sim = load_expert_ground_truth_from_xml()
    
    available_tids = {entry['text_id'] for entry in data}
    common_tids = sorted(list(available_tids.intersection(set(expert_ids))))
    
    if not common_tids:
        print("No common text IDs found.")
        return

    print(f"Evaluating {len(common_tids)} texts.")

    text_to_idx = {tid: i for i, tid in enumerate(expert_ids)}
    indices = [text_to_idx[tid] for tid in common_tids]
    expert_sim_filtered = expert_sim[np.ix_(indices, indices)]
    expert_upper = expert_sim_filtered[np.triu_indices(len(common_tids), k=1)]

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    modes = {
        "Intent Only": lambda step: step.get('normalized_intent', ''),
        "Raw Source Only": lambda step: step.get('raw_source', ''),
        "Intent + Context": lambda step: f"{step.get('normalized_intent', '')} {step.get('context_and_theory', '')}"
    }

    all_results = []

    for mode_name, text_func in modes.items():
        print(f"\n[Mode: {mode_name}]")
        
        all_units = []
        for entry in data:
            if entry['text_id'] not in common_tids: continue
            for step in entry['extracted_units']:
                all_units.append({
                    'text_id': entry['text_id'],
                    'text': text_func(step)
                })
        
        if not all_units: continue
        embeddings = model.encode([u['text'] for u in all_units], show_progress_bar=True)

        for t in THRESHOLDS:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=t,
                metric='cosine',
                linkage='average'
            ).fit(embeddings)
            labels = clustering.labels_
            n_clusters = len(set(labels))

            doc_clusters = {tid: set() for tid in common_tids}
            for i, label in enumerate(labels):
                doc_clusters[all_units[i]['text_id']].add(label)

            n = len(common_tids)
            derived_sim = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                     derived_sim[i, j] = derived_sim[j, i] = jaccard_sim(
                         doc_clusters[common_tids[i]], 
                         doc_clusters[common_tids[j]]
                     )
            np.fill_diagonal(derived_sim, 1.0)

            derived_upper = derived_sim[np.triu_indices(n, k=1)]
            rho, _ = spearmanr(derived_upper, expert_upper)
            
            # NN Evaluation
            nn1 = evaluate_nn(derived_sim, expert_sim_filtered, k=1)
            nn3 = evaluate_nn(derived_sim, expert_sim_filtered, k=3)
            
            print(f"  Threshold {t:.1f} -> Clusters: {n_clusters:<4} | rho: {rho:.4f} | NN-1: {nn1}/{n}")
            all_results.append({
                "Mode": mode_name,
                "Threshold": t,
                "Clusters": n_clusters,
                "Spearman rho": rho,
                "NN-1": f"{nn1}/{n}",
                "NN-3": f"{nn3}/{n}"
            })

    print("\n" + "#"*70)
    print("EXTRACTION PERFORMANCE LEADERBOARD (Sweep Analysis)")
    print("#"*70)
    df = pd.DataFrame(all_results).sort_values("Spearman rho", ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_evaluation()
