import json
import os
import re
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../data/atomic_extraction_results.json")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def pilot_clustering():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Flatten all atomic steps and collect intents
    all_steps = []
    for entry in data:
        for step in entry['atomic_steps']:
            all_steps.append({
                'text_id': entry['text_id'],
                'segment_id': entry['segment_id'],
                'category': step.get('step_category', 'unknown'),
                'intent': step.get('normalized_intent', ''),
                'raw': step.get('raw_source', '')
            })

    intents = [s['intent'] for s in all_steps]
    
    if len(intents) < 2:
        print("Not enough intents to cluster.")
        return

    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(intents)} intents...")
    X = model.encode(intents)

    # 3. Semantic Clustering (Agglomerative)
    # We use Cosine Distance for semantic space
    n_clusters = min(8, len(intents) // 2 + 1)
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.5, # Adjust for cluster tightness
        metric='cosine', 
        linkage='average'
    ).fit(X)
    labels = clustering.labels_

    # 4. Organize results
    bclusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(all_steps[i])

    # 5. Output Findings
    print("\n" + "="*50)
    print("SEMANTIC PILOT CLUSTERING (Sentence Embeddings)")
    print("="*50)

    for cluster_id in sorted(clusters.keys()):
        steps = clusters[cluster_id]
        print(f"\n[Cluster {cluster_id}]")
        # Identify the "Centroid" intent or most representative one
        print("-" * 30)
        for s in steps:
            cat_mark = "[P]" if s['category'] == 'procedural' else "[D]"
            print(f"  {cat_mark} {s['intent'][:80]}...")
            print(f"      [{s['text_id']}-Seg{s['segment_id']}] Source: {s['raw'][:60]}...")

if __name__ == "__main__":
    pilot_clustering()
