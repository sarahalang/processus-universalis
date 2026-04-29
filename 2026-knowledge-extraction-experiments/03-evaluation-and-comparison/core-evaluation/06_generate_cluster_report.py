import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import collections

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../../data/atomic_extraction_results.json")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "../../data/cluster_report.html")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def generate_report():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
    modes = {
        "Intent": lambda step: step.get('normalized_intent', ''),
        "Raw": lambda step: step.get('raw_source', ''),
        "Hybrid": lambda step: f"{step.get('normalized_intent', '')} {step.get('context_and_theory', '')}"
    }

    # Structure: report_data[mode][threshold][cluster_id] = [list of steps]
    report_data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for mode_name, text_func in modes.items():
        print(f"Processing mode: {mode_name}...")
        units = []
        for entry in data:
            for step in entry['extracted_units']:
                units.append({
                    'text_id': entry['text_id'],
                    'type': step.get('unit_type', 'unknown'),
                    'intent': step.get('normalized_intent', ''),
                    'context': step.get('context_and_theory', ''),
                    'raw': step.get('raw_source', ''),
                    'text': text_func(step)
                })
        
        embeddings = model.encode([u['text'] for u in units])
        
        for t in thresholds:
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=t, metric='cosine', linkage='average'
            ).fit(embeddings)
            
            # Calculate cluster purity for this threshold
            temp_df = pd.DataFrame({'label': clustering.labels_, 'type': [u['type'] for u in units]})
            composition = temp_df.groupby('label')['type'].value_counts().unstack(fill_value=0)
            if 'procedural' not in composition.columns: composition['procedural'] = 0
            if 'descriptive' not in composition.columns: composition['descriptive'] = 0

            for i, label in enumerate(clustering.labels_):
                proc_c = composition.loc[label, 'procedural']
                desc_c = composition.loc[label, 'descriptive']
                purity_str = f"({int(proc_c)}P / {int(desc_c)}D)"
                
                # We need to store a COPY of the unit because labels change per threshold
                unit_copy = units[i].copy()
                unit_copy['purity_info'] = purity_str
                report_data[mode_name][t][int(label)].append(unit_copy)

    # Build HTML with nested details
    html_body = "<h1>Atomic Cluster Hierarchy (All Configs)</h1>"
    for mode, configs in report_data.items():
        html_body += f"<details open><summary><h2>Mode: {mode}</h2></summary>"
        for t, clusters in configs.items():
            html_body += f"<details><summary><b>Threshold {t}</b> ({len(clusters)} clusters)</summary>"
            # Sort clusters by size
            for cid, steps in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
                purity = steps[0]['purity_info']
                html_body += f"<details><summary>Cluster {cid} ({len(steps)} steps) <small style='color:#666'>{purity}</small></summary><ul>"
                for s in steps:
                    html_body += f"<li>[{s['text_id']}] <i>{s['type']}</i>: {s['intent']} <br><small><b>Context:</b> {s['context']}</small><br><small>Source: {s['raw']}</small></li>"
                html_body += "</ul></details>"
            html_body += "</details>"
        html_body += "</details>"
            
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(f"<html><head><style>details{{margin:10px; padding:10px; border:1px solid #ccc; border-radius:5px;}} summary{{cursor:pointer; padding:5px;}} ul{{list-style-type:none;}} li{{margin-bottom:10px; padding-bottom:5px; border-bottom:1px solid #eee;}}</style></head><body>{html_body}</body></html>")
    print(f"Interactive hierarchical report with purity saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_report()
