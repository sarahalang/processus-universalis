import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import re

# ----------------------------
# CONFIG
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON = os.path.join(
    SCRIPT_DIR,
    "../data/atomic_extraction_results.json"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "../data/splitstree_outputs"
)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# A → E MAPPING (IMPORTANT ADDITION)
# ----------------------------
A_TO_E = {
    'a1': 'E16',
    'a2': 'E37',
    'a3': 'E38',
    'a4': 'E44',
    'a5': 'E17',
    'a6': 'E19',
    'a7': 'E39',
    'a8': 'E34',
    'a9': 'E2',
    'a12': 'E45',
    'a13': 'E42',
    'a15': 'E32b',
    'a16': 'E27',
    'a21': 'E3',
    'a22': 'E35',
    'a25': 'E22',
    'a26': 'E11',
}


# ----------------------------
# LABEL ENRICHMENT (NEW CORE FUNCTION)
# ----------------------------
def enrich_id(raw_id: str) -> str:
    """
    Converts:
        g1a11 -> g1_a11_E16
        a3 -> a3_E38
    """
    raw_id = str(raw_id)

    # extract group (g1, g2, ...)
    g_match = re.match(r'(g\d+)', raw_id)
    group = g_match.group(1) if g_match else ""

    # extract aXX
    m = re.search(r'(a\d+)', raw_id)
    a_id = m.group(1) if m else raw_id

    e_id = A_TO_E.get(a_id, "")

    parts = [p for p in [group, a_id, e_id] if p]
    return "_".join(parts)


# ----------------------------
# HELPERS
# ----------------------------
def canonical_id(text_id):
    import re
    m = re.search(r'(a\d+)', str(text_id))
    return m.group(1) if m else None


def build_units(data, mode="Hybrid"):
    def text_func(step):
        if mode == "Intent Only":
            return step.get("normalized_intent", "")
        elif mode == "Raw":
            return step.get("raw_source", "")
        else:
            return f"{step.get('normalized_intent','')} {step.get('context_and_theory','')}"

    units = []
    for entry in data:
        cid = str(entry["text_id"])

        # IMPORTANT CHANGE: enrich here
        cid_enriched = enrich_id(cid)

        if not cid:
            continue

        for step in entry["extracted_units"]:
            units.append({
                "tid": cid_enriched,
                "text": text_func(step)
            })

    return units


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------
# STEP 1: CLUSTERING
# ----------------------------
def cluster_units(units):
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode([u["text"] for u in units], show_progress_bar=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=THRESHOLD,
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(embeddings)

    for i, lab in enumerate(labels):
        units[i]["cluster"] = int(lab)

    return units


# ----------------------------
# STEP 2A: CLUSTER → BINARY MATRIX
# ----------------------------
def export_cluster_matrix(units):
    docs = sorted(set(u["tid"] for u in units))
    clusters = sorted(set(u["cluster"] for u in units))

    matrix = pd.DataFrame(0, index=docs, columns=[f"C{c}" for c in clusters])

    for u in units:
        matrix.loc[u["tid"], f"C{u['cluster']}"] = 1

    return matrix


def write_nexus_binary(df, path):
    ntax, nchar = df.shape

    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n")
        f.write("Begin data;\n")
        f.write(f" Dimensions ntax={ntax} nchar={nchar};\n")
        f.write(" Format datatype=standard symbols=\"01\" missing=?;\n")
        f.write("Matrix\n")

        for taxon, row in df.iterrows():
            f.write(f"{taxon}\t{''.join(map(str, row.values))}\n")

        f.write(";\nEnd;\n")


# ----------------------------
# STEP 2B: DISTANCE MATRIX
# ----------------------------
def export_distance_matrix(units):
    docs = sorted(set(u["tid"] for u in units))

    doc_sets = defaultdict(set)
    for u in units:
        doc_sets[u["tid"]].add(u["cluster"])

    n = len(docs)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist[i, j] = 1 - jaccard(doc_sets[docs[i]], doc_sets[docs[j]])

    return docs, dist


def write_nexus_distance(docs, dist, path):
    n = len(docs)

    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n")
        f.write("Begin distances;\n")
        f.write(f" Dimensions ntax={n};\n")
        f.write(" Format triangle=both;\n")
        f.write(" Matrix\n")

        for i in range(n):
            row = " ".join(f"{dist[i][j]:.4f}" for j in range(n))
            f.write(f"{docs[i]} {row}\n")

        f.write(";\nEnd;\n")


def write_phylip(docs, dist, path):
    n = len(docs)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = " ".join(f"{dist[i][j]:.4f}" for j in range(n))
            f.write(f"{docs[i]} {row}\n")


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Building units...")
    units = build_units(data, mode="Hybrid")

    print("Clustering...")
    units = cluster_units(units)

    print("Exporting cluster binary matrix...")
    cluster_matrix = export_cluster_matrix(units)

    bin_path = os.path.join(OUTPUT_DIR, "cluster_characters.nex")
    write_nexus_binary(cluster_matrix, bin_path)

    print("Exporting distance matrix...")
    docs, dist = export_distance_matrix(units)

    dist_nex = os.path.join(OUTPUT_DIR, "distance_matrix.nex")
    dist_phy = os.path.join(OUTPUT_DIR, "distance_matrix.phy")

    write_nexus_distance(docs, dist, dist_nex)
    write_phylip(docs, dist, dist_phy)

    print("\nDONE")
    print("Saved:")
    print(" -", bin_path)
    print(" -", dist_nex)
    print(" -", dist_phy)


if __name__ == "__main__":
    main()
