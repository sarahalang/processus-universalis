import json
import xml.etree.ElementTree as ET

def align_knowledge_to_expert():
    # Load extracted JSON knowledge
    with open("2026-analyses-summary/extracted_opac_knowledge.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)
    
    # Load XML
    tree = ET.parse("sammlung_aller_texte.xml")
    root = tree.getroot()
    
    print(f"{'Text/Seg':<10} | {'Extracted Haupt-Vorgang':<30} | {'Expert Category (Match)'}")
    print("-" * 100)
    
    alignment_count = 0
    total_steps = 0
    
    for entry in extracted:
        tid = entry['text_id']
        div = root.find(f".//div[@type='{tid}']")
        if div is None: continue
        
        # Expert keys for this document (type: keywords)
        expert_keys = {k.get('type'): k.get('n', '').lower() for k in div.findall('keys')}
        
        for step in entry.get('schritte', []):
            total_steps += 1
            op = step.get('haupt_vorgang', '').lower()
            
            # Simple heuristic:
            # Does the expert have a category that contains words from the Haupt-Vorgang?
            match = "None"
            for ktype, kvals in expert_keys.items():
                # Split op into words and check if any match keywords
                op_words = [w for w in re.split(r'[^a-zäöüß]+', op) if len(w) > 3]
                if any(w in kvals for w in op_words):
                    match = ktype
                    alignment_count += 1
                    break
            
            print(f"{tid}/{entry['seg_id']:<5} | {op[:28]:<30} | {match}")

    print(f"\nAlignment Ratio: {alignment_count}/{total_steps} ({alignment_count/total_steps*100:.1f}%)")

import re
if __name__ == "__main__":
    align_knowledge_to_expert()
