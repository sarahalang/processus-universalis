import json
import os
import re
from collections import Counter

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "../data/extracted_opac_knowledge.json")

def normalize_vorgang(vorgang):
    if not vorgang or vorgang == 'Unknown':
        return 'Unknown'
    
    v = str(vorgang).lower()
    
    # Define normalization rules
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
    
    return vorgang 

def analyze_results():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {DATA_FILE} is not a valid JSON file.")
            return

    num_segments = len(results)
    print(f"--- General Statistics ---")
    print(f"Total Segments Processed: {num_segments}")

    all_vorgange_raw = []
    all_vorgange_norm = []
    all_substanzen = []
    all_details = []
    all_kontext = []
    
    total_steps = 0
    steps_with_details = 0
    steps_with_context = 0
    steps_with_substanz = 0

    for res in results:
        schritte = res.get('schritte', [])
        total_steps += len(schritte)
        
        for step in schritte:
            # 1. Haupt-Vorgang
            vorgang = step.get('haupt_vorgang', 'Unknown')
            if isinstance(vorgang, list):
                vorgang = ", ".join(filter(None, [str(v) for v in vorgang]))
            vorgang = (vorgang or 'Unknown').strip()
            all_vorgange_raw.append(vorgang)
            all_vorgange_norm.append(normalize_vorgang(vorgang))
            
            # 2. Ziel-Substanz
            substanz = step.get('ziel_substanz', 'Unknown')
            if isinstance(substanz, list):
                substanz = ", ".join(filter(None, [str(s) for s in substanz]))
            substanz = (substanz or 'Unknown').strip()
            if substanz and substanz != 'Unknown':
                steps_with_substanz += 1
                all_substanzen.append(substanz)
            
            # 3. Vollzug & Details (Coverage: contains at least one non-empty string)
            details = step.get('vollzug_details', [])
            if isinstance(details, list):
                clean_details = [str(d).strip() for d in details if str(d).strip()]
                if clean_details:
                    steps_with_details += 1
                    all_details.extend(clean_details)
            elif isinstance(details, str) and details.strip():
                steps_with_details += 1
                all_details.append(details.strip())
            
            # 4. Kontext (Coverage: contains at least one non-empty string)
            kontext = step.get('kontext', [])
            if isinstance(kontext, list):
                clean_kontext = [str(k).strip() for k in kontext if str(k).strip()]
                if clean_kontext:
                    steps_with_context += 1
                    all_kontext.extend(clean_kontext)
            elif isinstance(kontext, str) and kontext.strip():
                steps_with_context += 1
                all_kontext.append(kontext.strip())

    avg_steps = total_steps / num_segments if num_segments > 0 else 0
    
    print(f"Total Extraction Steps: {total_steps}")
    print(f"Average Steps per Segment: {avg_steps:.2f}")
    
    print(f"\n--- Key Analysis: NORMALIZED HAUPT-VORGANG (Top 10) ---")
    for v, count in Counter(all_vorgange_norm).most_common(10):
        print(f"  {count:3d} | {v}")

    print(f"\n--- Key Analysis: ZIEL-SUBSTANZ (Top 10) ---")
    for s, count in Counter(all_substanzen).most_common(10):
        print(f"  {count:3d} | {s}")

    print(f"\n--- Key Analysis: VOLLZUG & DETAILS (Top 10 individual actions) ---")
    for d, count in Counter(all_details).most_common(10):
        print(f"  {count:3d} | {d}")

    print(f"\n--- Key Analysis: KONTEXT (Top 10) ---")
    for k, count in Counter(all_kontext).most_common(10):
        print(f"  {count:3d} | {k}")

    print(f"\n--- Extraction Coverage Details ---")
    print(f"1. Haupt-Vorgang (Operation): {total_steps}/{total_steps} (100%)")
    print(f"2. Ziel-Substanz (Substance): {steps_with_substanz}/{total_steps} ({steps_with_substanz/total_steps:.1%})")
    print(f"3. Vollzug & Details (Procedures): {steps_with_details}/{total_steps} ({steps_with_details/total_steps:.1%})")
    print(f"4. Kontext (Environment): {steps_with_context}/{total_steps} ({steps_with_context/total_steps:.1%})")

if __name__ == "__main__":
    analyze_results()
