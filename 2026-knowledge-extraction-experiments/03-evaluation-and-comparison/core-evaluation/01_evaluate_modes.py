import json
import os
import re
import csv
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import collections
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../data/atomic_extraction_results.json")
XML_PATH = os.path.join(SCRIPT_DIR, "../../sammlung_aller_texte.xml")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Thresholds to test
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
    u = len(set1.union(set2))
    return len(set1.intersection(set2)) / u if u > 0 else 0.0

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
        "Hybrid": lambda step: f"{step.get('normalized_intent', '')} {step.get('context_and_theory', '')}"
    }

    all_results = []

    for mode_name, text_func in modes.items():
        print(f"\n[Mode: {mode_name}]")
        all_units = []
        for entry in data:
            if entry['text_id'] not in common_tids: continue
            for step in entry['extracted_units']:
                all_units.append({'tid': entry['text_id'], 'text': text_func(step)})
        
        if not all_units: continue
        embeddings = model.encode([u['text'] for u in all_units], show_progress_bar=True)

        for t in THRESHOLDS:
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric='cosine', linkage='average').fit(embeddings)
            labels = clustering.labels_
            n_raw_clusters = len(set(labels))

            # Filter logic: Identify clusters with count < 3
            counts = collections.Counter(labels)
            valid_cids = {cid for cid, count in counts.items() if count >= 3}
            n_filtered_clusters = len(valid_cids)

            # Build similarity matrices
            n_docs = len(common_tids)
            sim_raw = np.zeros((n_docs, n_docs))
            sim_filtered = np.zeros((n_docs, n_docs))
            
            doc_sets_raw = {tid: set() for tid in common_tids}
            doc_sets_filtered = {tid: set() for tid in common_tids}

            for i, label in enumerate(labels):
                doc_sets_raw[all_units[i]['tid']].add(label)
                if label in valid_cids:
                    doc_sets_filtered[all_units[i]['tid']].add(label)

            for i in range(n_docs):
                for j in range(i+1, n_docs):
                    sim_raw[i,j] = sim_raw[j,i] = jaccard_sim(doc_sets_raw[common_tids[i]], doc_sets_raw[common_tids[j]])
                    sim_filtered[i,j] = sim_filtered[j,i] = jaccard_sim(doc_sets_filtered[common_tids[i]], doc_sets_filtered[common_tids[j]])
            
            np.fill_diagonal(sim_raw, 1.0)
            np.fill_diagonal(sim_filtered, 1.0)

            rho_raw, _ = spearmanr(sim_raw[np.triu_indices(n_docs, k=1)], expert_upper)
            rho_filt, _ = spearmanr(sim_filtered[np.triu_indices(n_docs, k=1)], expert_upper)
            
            print(f"  T {t:.1f} | Raw: {rho_raw:.3f} ({n_raw_clusters}c) | Filtered: {rho_filt:.3f} ({n_filtered_clusters}c)")
            
            all_results.append({
                "Mode": mode_name,
                "Threshold": t,
                "Clusters (Filt)": n_filtered_clusters,
                "Spearman rho (Raw)": rho_raw,
                "Spearman rho (Filt)": rho_filt
            })

    print("\n" + "#"*75)
    print("COMPARISON: RAW VS SPARSE-FILTERED (MIN 3)")
    print("#"*75)
    df = pd.DataFrame(all_results).sort_values("Spearman rho (Filt)", ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_evaluation()
