import json
import os
import csv
import re

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "../data/extracted_opac_knowledge.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../data/automated_procedural_matrix.csv")

CATEGORIES = [
    "Theoretische Erläuterung",
    "Probenahme / Materialgewinnung",
    "Mazeration / Extraktion",
    "Filtration",
    "Sieden / Konzentration",
    "Destillation / Rektifikation",
    "Sublimation",
    "Calcination / Rösten",
    "Mischen / Vorbereitung",
    "Lagerung / Reifung",
    "Multiplikation / Projektion"
]

def normalize_vorgang(vorgang):
    if not vorgang or vorgang == 'Unknown':
        return None
    v = str(vorgang).lower()
    if re.search(r'theoretisch|erläuterung|vorbetrachtung', v):
        return 'Theoretische Erläuterung'
    if re.search(r'destillation|destillieren|rectifikation|rektifikation', v):
        return 'Destillation / Rektifikation'
    if re.search(r'sublimation|sublimieren', v):
        return 'Sublimation'
    if re.search(r'probenahme|ausgraben|gewinnung|auswahl|vorbereitung der erde', v):
        return 'Probenahme / Materialgewinnung'
    if re.search(r'mazeration|auslaugen|lauge|extraktion|extraktion von salz', v):
        return 'Mazeration / Extraktion'
    if re.search(r'calcination|calcinieren|brennen|rösten', v):
        return 'Calcination / Rösten'
    if re.search(r'filtration|filtrieren|seihen|klären', v):
        return 'Filtration'
    if re.search(r'lagerung|aufbewahrung|aufheben|liegen lassen|warmhalten', v):
        return 'Lagerung / Reifung'
    if re.search(r'multiplikation|projektion|tinctur', v):
        return 'Multiplikation / Projektion'
    if re.search(r'einsieden|kochen|abdampfen|evaporat|verdampfung|coagulation|koagulation', v):
        return 'Sieden / Konzentration'
    if re.search(r'mischen|vermengen|zusetzen|einfüllen', v):
        return 'Mischen / Vorbereitung'
    return None

def generate_matrix():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 1. Aggregate presence of categories per document
    # doc_matrix = { "text_id": set(present_categories) }
    doc_matrix = {}
    
    for res in results:
        tid = res.get('text_id')
        if not tid: continue
        
        if tid not in doc_matrix:
            doc_matrix[tid] = set()
            
        for step in res.get('schritte', []):
            vorgang = step.get('haupt_vorgang')
            if isinstance(vorgang, list):
                for v in vorgang:
                    norm = normalize_vorgang(v)
                    if norm: doc_matrix[tid].add(norm)
            else:
                norm = normalize_vorgang(vorgang)
                if norm: doc_matrix[tid].add(norm)

    # 2. Write to CSV
    text_ids = sorted(doc_matrix.keys())
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        header = ["text_id"] + CATEGORIES + ["total_steps", "completeness_score"]
        writer.writerow(header)
        
        for tid in text_ids:
            present = doc_matrix[tid]
            row = [tid]
            count = 0
            for cat in CATEGORIES:
                val = 1 if cat in present else 0
                row.append(val)
                count += val
            
            score = round(count / len(CATEGORIES), 3)
            row.extend([count, score])
            writer.writerow(row)

    print(f"Automated procedural matrix generated at: {OUTPUT_FILE}")
    print(f"Processed {len(text_ids)} documents across {len(CATEGORIES)} categories.")

if __name__ == "__main__":
    generate_matrix()
