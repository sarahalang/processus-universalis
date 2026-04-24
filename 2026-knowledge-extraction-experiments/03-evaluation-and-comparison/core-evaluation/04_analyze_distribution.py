import json
import os
import pandas as pd
import collections

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../data/labeled_segments.json")

def analyze_distribution():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Please run label_clusters.py first.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 1. Document Inventory (Concepts per text)
    doc_inventories = df.groupby('text_id')['cluster_label'].nunique()
    avg_concepts = doc_inventories.mean()
    
    print("\n" + "="*60)
    print("ANALYSIS: CONCEPTUAL DENSITY PER DOCUMENT")
    print("="*60)
    print(doc_inventories.sort_values(ascending=False).to_string())
    print("-" * 60)
    print(f"Average Unique Concepts per Document: {avg_concepts:.2f}")

    # 2. Global Distribution
    concept_doc_counts = df.groupby('cluster_label')['text_id'].nunique()
    total_docs = df['text_id'].nunique()
    concept_total_counts = df.groupby('cluster_label').size()
    
    concept_stats = pd.DataFrame({
        'Doc Frequency': concept_doc_counts,
        'Doc %': (concept_doc_counts / total_docs * 100).round(1),
        'Total Statements': concept_total_counts
    }).sort_values('Doc Frequency', ascending=False)
    
    print("\n" + "="*60)
    print(f"ANALYSIS: GLOBAL CONCEPT DISTRIBUTION ({total_docs} docs total)")
    print("="*60)
    print(concept_stats.head(20).to_string())
    
    print("\n" + "="*60)
    print("ANALYSIS: DISTRIBUTION OF CONCEPTUAL RARITY")
    print("="*60)
    rarity_dist = concept_stats['Doc Frequency'].value_counts().sort_index()
    for doc_count, cluster_count in rarity_dist.items():
        print(f"Found in {doc_count:2} documents: {cluster_count:3} concepts")

    # Save to CSV
    output_csv = os.path.join(SCRIPT_DIR, "../data/concept_distribution_stats.csv")
    concept_stats.to_csv(output_csv)
    print(f"\nFull distribution stats saved to {output_csv}")

if __name__ == "__main__":
    analyze_distribution()
