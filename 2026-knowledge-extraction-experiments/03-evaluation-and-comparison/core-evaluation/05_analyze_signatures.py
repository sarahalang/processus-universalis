import json
import os
import pandas as pd
import numpy as np
import re
from scipy.stats import pointbiserialr

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../../data/labeled_segments.json")

# Mapping data from previous analysis
A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11'
}
E_TO_GROUP = {
    'E2': 'I', 'E3': 'I', 'E11': 'I', 'E22': 'I', 'E35': 'I',
    'E16': 'II', 'E17': 'II', 'E19': 'II', 'E27': 'II', 'E32b': 'II',
    'E34': 'III', 'E37': 'III', 'E38': 'III', 'E39': 'III',
    'E42': 'III', 'E44': 'III', 'E45': 'III',
}

def get_group(tid):
    m = re.search(r'a(\d+)', tid)
    if m:
        a_key = 'a' + m.group(1)
        ename = A_TO_E.get(a_key)
        return E_TO_GROUP.get(ename)
    return None

def analyze_signatures():
    if not os.path.exists(INPUT_JSON):
        print("Error: labeled_segments.json not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    matrix = df.groupby(['text_id', 'cluster_label']).size().unstack(fill_value=0)
    
    valid_tids = [tid for tid in matrix.index if get_group(tid) is not None]
    X = matrix.loc[valid_tids]
    y = np.array([get_group(tid) for tid in valid_tids])
    
    print("\n" + "="*80)
    print("ALCHEMICAL GROUP SIGNATURES (Diagnostic Concepts)")
    print("="*80)
    
    # Analyze each group
    for group in ['I', 'II', 'III']:
        y_binary = (y == group).astype(int)
        
        signatures = []
        for cluster in X.columns:
            # Point-Biserial correlation between cluster count and group membership
            corr, _ = pointbiserialr(y_binary, X[cluster])
            signatures.append({'cluster': cluster, 'corr': corr})
        
        sig_df = pd.DataFrame(signatures).sort_values('corr', ascending=False)
        print(f"\nSIGNATURES FOR GROUP {group}:")
        # Show top 5 positive (diagnostic) and bottom 5 negative (absent)
        print(sig_df.head(20).to_string(index=False))
        #print("...")
        #print(sig_df.tail(7).to_string(index=False))

if __name__ == "__main__":
    analyze_signatures()
