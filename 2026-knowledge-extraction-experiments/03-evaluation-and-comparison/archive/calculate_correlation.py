import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import xml.etree.ElementTree as ET

# Configuration (using absolute paths)
BASE_DIR = '/home/pet/processus-universalis'
MATRIX_PATH = f'{BASE_DIR}/2026-analyses/data/processus_matrix.csv'
XML_PATH = f'{BASE_DIR}/2026-analyses/data/processus-sammlung_aller_texte.xml'

# A_TO_E mapping (as inferred from capstone_analysis context)
A_TO_E = {f'a{i}': f'E{i}' for i in range(1, 100)}

# 1. Reconstruct Expert Features
tree = ET.parse(XML_PATH)
root = tree.getroot()
anno_features = {}

for div in root.findall('div'):
    dtype = div.get('type', '')
    m_a = re.search(r'a(\d+)', dtype)
    if m_a:
        a_key = 'a' + m_a.group(1)
        ename = A_TO_E.get(a_key)
        features = set()
        for keys_el in div.findall('.//keys'):
            kvals = keys_el.get('n', '')
            ktype = keys_el.get('type', '')
            if kvals and 'FEHLT' not in kvals:
                for val in kvals.split(';'):
                    if val.strip():
                        features.add(f"{ktype}::{val.strip()}")
        if features:
            anno_features[ename] = features

# Align indices
anno_keys = set(anno_features.keys())
matrix_keys = set(pd.read_csv(MATRIX_PATH, index_col='e_name').index)
common = sorted(list(anno_keys.intersection(matrix_keys)))
nc = len(common)
expert_dist = np.zeros((nc, nc))
for i in range(nc):
    for j in range(i+1, nc):
        a = anno_features[common[i]]
        b = anno_features[common[j]]
        if len(a | b) > 0:
            expert_dist[i, j] = expert_dist[j, i] = 1 - len(a & b) / len(a | b)

# 2. Calculate New Distance Matrix
df = pd.read_csv(MATRIX_PATH, index_col='e_name')
# Filter out non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.loc[common] # Align
new_dist_array = pdist(df_numeric.values, metric='jaccard')
new_dist_matrix = squareform(new_dist_array)

# 3. Correlation
def upper_tri(mat):
    return mat[np.triu_indices(len(mat), k=1)]

corr_pearson, _ = pearsonr(upper_tri(new_dist_matrix), upper_tri(expert_dist))
corr_spearman, _ = spearmanr(upper_tri(new_dist_matrix), upper_tri(expert_dist))

print(f"Correlation with Expert Matrix:")
print(f"Pearson r: {corr_pearson:.3f}")
print(f"Spearman rho: {corr_spearman:.3f}")
