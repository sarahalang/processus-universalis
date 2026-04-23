import xml.etree.ElementTree as ET
import csv
import os
import re

def validate_correlation(xml_path, csv_path):
    # 1. Parse XML to get Ground Truth positions
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        if os.path.exists("../" + xml_path):
            tree = ET.parse("../" + xml_path)
        else:
            raise
    
    root = tree.getroot()
    
    # Store GT keys per div: {tid: [(char_offset, key_type), ...]}
    gt_map = {}
    div_texts = {}
    
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
        full_text = ""
        keys_at_pos = []
        
        # Handle initial text
        if div.text:
            full_text += div.text
            
        for child in div:
            # Mark the position of the key
            if child.tag == 'keys':
                k_type = child.get('type')
                keys_at_pos.append((len(full_text), k_type))
            
            # Add child text and tail (recursive approach for mixed content)
            def get_full_inner(node):
                inner = ""
                if node.text: inner += node.text
                for sub in node:
                    inner += get_full_inner(sub)
                    if sub.tail: inner += sub.tail
                return inner
            
            if child.tag != 'keys':
                full_text += get_full_inner(child)
            
            if child.tail:
                full_text += child.tail
        
        # Clean up whitespace for matching
        clean_full_text = re.sub(r'\s+', ' ', full_text).strip()
        div_texts[tid] = clean_full_text
        
        # Map raw offsets to clean offsets
        # This is non-trivial, so we will use a simpler containment check
        gt_map[tid] = [(k[1], k[0]) for k in keys_at_pos]

    # 2. Load Unsupervised Segments
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        us_segments = list(reader)

    # 3. Correlate using key phrases from the original text
    # Since we can't easily map exact character offsets after re.sub, 
    # we check if the segment text CONTAINS the text surrounding the expert key.
    
    results = []
    
    # We need to re-parse to get the text *after* each key to identify it
    key_phrases = {} # {tid: [(key_type, unique_snippet), ...]}
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
        key_phrases[tid] = []
        for i, child in enumerate(div):
            if child.tag == 'keys':
                k_type = child.get('type')
                if not k_type: continue # Skip empty key types
                
                # Find the next text node or tail text
                snippet = ""
                # Try tail of current node
                if child.tail and child.tail.strip():
                    snippet = re.sub(r'\s+', ' ', child.tail).strip()[:50]
                
                # If tail was empty, look ahead in siblings
                if not snippet:
                    for sibling in list(div)[i+1:]:
                        if sibling.text and sibling.text.strip():
                            snippet = re.sub(r'\s+', ' ', sibling.text).strip()[:50]
                            break
                        if sibling.tail and sibling.tail.strip():
                            snippet = re.sub(r'\s+', ' ', sibling.tail).strip()[:50]
                            break
                
                if snippet:
                    key_phrases[tid].append((k_type, snippet))

    for seg in us_segments:
        tid = seg['text_id']
        seg_text = seg['full_text'].strip()
        
        keys_found = []
        if tid in key_phrases:
            for k_type, snippet in key_phrases[tid]:
                # Use a bit more fuzzy check for snippet in segment
                if snippet and snippet in seg_text:
                    keys_found.append(k_type)
        
        unique_keys = sorted(list(set(keys_found)))
        results.append({
            'tid': tid,
            'key_count': len(unique_keys),
            'keys': unique_keys
        })

    # 4. Calculate Metrics
    total = len(results)
    perfect = len([r for r in results if r['key_count'] == 1])
    mixed = len([r for r in results if r['key_count'] > 1])
    empty = len([r for r in results if r['key_count'] == 0])
    
    # "Empty" here often means the segment is part of a longer key's text 
    # and doesn't contain a *start* of a new key. 
    # "Mixed" is the real metric for "Correctness" (purity).
    
    print("=== SEGMENTATION VALIDATION REPORT ===")
    print(f"Total Segments Analyzed: {total}")
    print("-" * 40)
    print(f"Pure Segments (Contains 1 Key start):   {perfect} ({perfect/total*100:.1f}%)")
    print(f"Mixed Segments (Contains >1 Key start): {mixed} ({mixed/total*100:.1f}%)")
    print(f"Continued Segments (0 Key starts):      {empty} ({empty/total*100:.1f}%)")
    print("-" * 40)
    print("Note: 'Continued' segments are parts of a long step. 'Mixed' segments are potential over-groupings.")
    
    if mixed > 0:
        print("\nTop 'Mixed' Segments (Where the machine failed to split):")
        mixed_examples = sorted([r for r in results if r['key_count'] > 1], key=lambda x: x['key_count'], reverse=True)
        for ex in mixed_examples[:5]:
            print(f"[{ex['tid']}] Contains {ex['key_count']} keys: {', '.join(ex['keys'][:3])}...")

if __name__ == "__main__":
    validate_correlation("sammlung_aller_texte.xml", "2026-analyses-summary/unsupervised_segments_full.csv")
