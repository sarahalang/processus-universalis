import json
import xml.etree.ElementTree as ET
import re

def get_segment_text(div, seg_id):
    # This matches the segmentation logic in unsupervised_segmentation.py
    # Re-simulating segment extraction to get the corresponding XML snippet
    all_text = " ".join([t.strip() for t in div.itertext() if t.strip()])
    # Very simplified split for comparison:
    # We take segments by looking for sentence breaks as the original script did
    sentence_breaks = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', all_text)
    # The original script does a complex boundary logic, 
    # for comparison we just need to know which keys belong to which sentence
    return sentence_breaks

def verify_extraction_accurate():
    with open("2026-analyses-summary/extracted_opac_knowledge.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)
    
    tree = ET.parse("sammlung_aller_texte.xml")
    root = tree.getroot()
    
    print(f"{'Text/Seg':<10} | {'Main Operation (Extracted)':<30} | {'Keys Found in Segment (XML)'}")
    print("-" * 110)
    
    for entry in extracted:
        tid = entry['text_id']
        seg_id = entry['seg_id']
        seg_text = entry.get('full_text', '') # This would require loading the full text map
        
        # Find the div
        div = root.find(f".//div[@type='{tid}']")
        if div is None: continue
        
        # Find keys that actually fall within the range of this segment
        # In the XML, keys are nodes. We look for keys that appear near the segment text.
        # This is hard because of re-segmentation. Let's look for keyword presence.
        
        extracted_ops = [s.get('haupt_vorgang') for s in entry['schritte']]
        
        # Simple heuristic: which expert keys are mentioned in this segment text?
        found_keys = []
        for key_node in div.findall('keys'):
            n_attr = key_node.get('n', '').lower()
            if any(w in seg_text.lower() for w in re.findall(r'\b[a-zäöüß]{4,}\b', n_attr)):
                found_keys.append(key_node.get('type'))
        
        print(f"{tid}/{seg_id:<5} | {str(extracted_ops[:1]):<30} | {', '.join(set(found_keys))}")

# We need the segments with text to run this comparison properly
def run_comparison():
    import csv
    with open("2026-analyses-summary/unsupervised_segments_full.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        segments = { (row['text_id'], row['seg_id']): row['full_text'] for row in reader }

    with open("2026-analyses-summary/extracted_opac_knowledge.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)

    tree = ET.parse("sammlung_aller_texte.xml")
    root = tree.getroot()

    print(f"{'Text/Seg':<10} | {'Main Operation':<30} | {'Expert Keys'}")
    print("-" * 100)

    for entry in extracted:
        key = (entry['text_id'], str(entry['seg_id']))
        text = segments.get(key, "")
        div = root.find(f".//div[@type='{entry['text_id']}']")
        
        found_keys = []
        if div is not None:
            for key_node in div.findall('keys'):
                n_attr = key_node.get('n', '').lower()
                # Check if this key's vocabulary exists in the segment text
                keywords = re.findall(r'\b[a-zäöüß]{4,}\b', n_attr)
                if any(k in text.lower() for k in keywords):
                    found_keys.append(key_node.get('type'))
        
        ops = [s.get('haupt_vorgang') for s in entry['schritte']]
        print(f"{key[0]}/{key[1]:<5} | {str(ops[:1]):<30} | {', '.join(set(found_keys))}")

if __name__ == "__main__":
    run_comparison()
