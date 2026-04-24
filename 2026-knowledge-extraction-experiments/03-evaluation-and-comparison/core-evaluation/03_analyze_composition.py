import json
import os
import pandas as pd
import collections

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../../data/labeled_segments.json")

def analyze_composition():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Please run label_clusters.py first.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Group by cluster and count types
    composition = df.groupby('cluster_label')['type'].value_counts().unstack(fill_value=0)
    
    # Ensure columns exist even if one type is missing from the entire run
    if 'procedural' not in composition.columns: composition['procedural'] = 0
    if 'descriptive' not in composition.columns: composition['descriptive'] = 0
    
    # Add metrics
    composition['Total'] = composition.sum(axis=1)
    composition['Proc%'] = (composition['procedural'] / composition['Total'] * 100).round(1)
    composition['Desc%'] = (composition['descriptive'] / composition['Total'] * 100).round(1)
    
    # Define "Purity"
    def get_category(row):
        if row['Proc%'] == 100: return 'Pure Procedural'
        if row['Desc%'] == 100: return 'Pure Descriptive'
        if row['Proc%'] >= 75: return 'Mostly Procedural'
        if row['Desc%'] >= 75: return 'Mostly Descriptive'
        return 'Balanced Mixed'

    composition['Category'] = composition.apply(get_category, axis=1)
    
    # Summary Statistics
    summary = composition['Category'].value_counts()
    
    print("\n" + "="*60)
    print("CLUSTER COMPOSITION ANALYSIS: LABORATORY VS. THEORY")
    print("="*60)
    print(summary.to_string())
    print("-" * 60)
    print(f"Total Clusters Analyzed: {len(composition)}")
    print(f"Total Atomic Units:      {len(df)}")
    
    # Show Most Interesting Mixed Clusters
    mixed = composition[composition['Category'] == 'Balanced Mixed'].sort_values('Total', ascending=False)
    if not mixed.empty:
        print("\nTOP HYBRID CLUSTERS (Strong link between Theory and Practice):")
        print(mixed[['procedural', 'descriptive', 'Total']].head(15))

    # Save detailed CSV for further plotting
    output_csv = os.path.join(SCRIPT_DIR, "../../data/cluster_composition.csv")
    composition.to_csv(output_csv)
    print(f"\nDetailed composition report saved to {output_csv}")

if __name__ == "__main__":
    analyze_composition()
