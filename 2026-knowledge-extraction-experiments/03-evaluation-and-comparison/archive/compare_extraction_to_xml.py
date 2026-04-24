import json
import xml.etree.ElementTree as ET

def verify_extraction():
    # Load extracted JSON
    with open("2026-analyses-summary/extracted_opac_knowledge.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)
    
    # Load XML
    tree = ET.parse("sammlung_aller_texte.xml")
    root = tree.getroot()
    
    print(f"{'Text/Seg':<10} | {'Expert Keys Found in XML':<40}")
    print("-" * 80)
    
    for entry in extracted:
        tid = entry['text_id']
        seg_id = entry['seg_id']
        
        # Find div in XML
        div = root.find(f".//div[@type='{tid}']")
        if div is not None:
            # Find keys in this div (simplified check)
            keys = [k.get('type') for k in div.findall('keys') if k.get('type')]
            print(f"{tid}/{seg_id:<5} | {', '.join(set(keys[:3]))}...")
        
        print(f"       -> Extracted Haupt-Vorgang: {[s.get('haupt_vorgang') for s in entry['schritte']]}")

if __name__ == "__main__":
    verify_extraction()
