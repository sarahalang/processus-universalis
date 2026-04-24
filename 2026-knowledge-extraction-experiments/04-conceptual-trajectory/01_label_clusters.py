import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
import collections

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../data/atomic_extraction_results.json")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "../data/labeled_segments.json")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.5

# Setup OpenAI client for labeling
def load_env():
    script_env = os.path.join(SCRIPT_DIR, "../../.env")
    if os.path.exists(script_env):
        with open(script_env, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

load_env()
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

def get_cluster_label(cluster_steps):
    """Ask LLM to summarize a cluster of steps into a single concept label."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Concatenate a sample of intents/context to get the "essence"
    samples = []
    for s in cluster_steps[:10]:
        samples.append(f"- {s['intent']} | {s['context']}")
    sample_text = "\n".join(samples)
    
    prompt = f"""
    You are an alchemical scholar. Label the following procedural steps with a 2-4 word concept name.
    Do not explain, do not add prefixes, just return the name.
    
    Examples: "Distilling the Menstruum", "Calcining the Earth", "Purifying the Salt".
    
    Steps:
    {sample_text}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.2
        )
        content = response.choices[0].message.content
        return content.strip().replace('"', '') if content else "Unnamed Cluster"
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Unnamed Cluster"

def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Prepare units for the best mode (Intent + Context)
    units = []
    for entry in data:
        for step in entry['extracted_units']:
            units.append({
                'text_id': entry['text_id'],
                'segment_id': entry['segment_id'],
                'type': step.get('unit_type', 'unknown'),
                'intent': step.get('normalized_intent', ''),
                'context': step.get('context_and_theory', ''),
                'raw': step.get('raw_source', ''),
                'text': f"{step.get('normalized_intent', '')} {step.get('context_and_theory', '')}"
            })
    
    # 2. Cluster
    embeddings = model.encode([u['text'] for u in units])
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=THRESHOLD, metric='cosine', linkage='average'
    ).fit(embeddings)
    
    # 3. Label Clusters
    cluster_groups = collections.defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        cluster_groups[int(label)].append(units[i])
        
    print(f"Generating labels for {len(cluster_groups)} clusters...")
    
    # Initialize client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Batch the labeling: create a mapping of all clusters
    all_cluster_descriptions = ""
    for cid, steps in cluster_groups.items():
        sample_text = "\n".join([f"- {s['intent']} | {s['context']}" for s in steps[:5]])
        all_cluster_descriptions += f"\n\nCluster {cid} Steps:\n{sample_text}\n"

    prompt = f"""
    You are an alchemical scholar. For each cluster of steps provided below, generate a 2-4 word concept name.
    
    Format your response as a JSON object:
    {{
      "Cluster 0": "Example Label",
      "Cluster 1": "Another Label"
    }}

    Data to analyze:
    {all_cluster_descriptions}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        labels_json = json.loads(response.choices[0].message.content)
        cluster_labels = {int(k.split()[1]): v for k, v in labels_json.items()}
    except Exception as e:
        print(f"Batch labeling failed: {e}. Falling back to individual labeling.")
        # Fallback to individual if batch fails
        cluster_labels = {}
        for cid, steps in cluster_groups.items():
            cluster_labels[cid] = "Unnamed Cluster"
        
    # 4. Assign labels back to data
    for i, label in enumerate(clustering.labels_):
        units[i]['cluster_label'] = cluster_labels[int(label)]
        
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(units, f, indent=2, ensure_ascii=False)
        
    print(f"Labeled data saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
