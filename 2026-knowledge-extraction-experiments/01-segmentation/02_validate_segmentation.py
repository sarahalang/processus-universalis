import xml.etree.ElementTree as ET
import csv
import os
import re
import pandas as pd
import collections

def validate_segmentation(xml_path, csv_path):
    # 1. Load segments
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        segments = list(reader)

    df = pd.DataFrame(segments)
    df['word_count'] = df['word_count'].astype(int)

    # 2. General Stats
    doc_counts = df.groupby('text_id').size()
    avg_segments_per_doc = doc_counts.mean()
    max_segments_doc = doc_counts.idxmax()
    min_segments_doc = doc_counts.idxmin()
    avg_word_len = df['word_count'].mean()

    print("=== SEGMENTATION VALIDATION REPORT ===")
    print(f"Total Segments: {len(df)}")
    print(f"Average Words per Segment: {avg_word_len:.1f}")
    print(f"Average Segments per Document: {avg_segments_per_doc:.1f}")
    print(f"Document with Most Segments: {max_segments_doc} ({doc_counts[max_segments_doc]} segments)")
    print(f"Document with Fewest Segments: {min_segments_doc} ({doc_counts[min_segments_doc]} segments)")
    print("-" * 40)

    # 3. Validation against XML keys (using existing logic)
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        if os.path.exists("../../" + xml_path):
            tree = ET.parse("../../" + xml_path)
        else:
            raise

    root = tree.getroot()
    key_phrases = {}
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
        key_phrases[tid] = []
        for i, child in enumerate(div):
            if child.tag == 'keys':
                k_type = child.get('type')
                if not k_type: continue
                snippet = ""
                if child.tail and child.tail.strip():
                    snippet = re.sub(r'\s+', ' ', child.tail).strip()[:50]
                if not snippet:
                    for sibling in list(div)[i+1:]:
                        if sibling.text and sibling.text.strip():
                            snippet = re.sub(r'\s+', ' ', sibling.text).strip()[:50]
                            break
                if snippet:
                    key_phrases[tid].append((k_type, snippet))

    results = []
    for seg in segments:
        tid = seg['text_id']
        seg_text = seg['full_text'].strip()
        keys_found = []
        if tid in key_phrases:
            for k_type, snippet in key_phrases[tid]:
                if snippet and snippet in seg_text:
                    keys_found.append(k_type)

        unique_keys = sorted(list(set(keys_found)))
        results.append({'tid': tid, 'key_count': len(unique_keys), 'keys': unique_keys})

    # 4. Final Validation Output
    total = len(results)
    perfect = len([r for r in results if r['key_count'] == 1])
    mixed = len([r for r in results if r['key_count'] > 1])
    empty = len([r for r in results if r['key_count'] == 0])

    print(f"Pure Segments (1 key):   {perfect} ({perfect/total*100:.1f}%)")
    print(f"Mixed Segments (>1 key): {mixed} ({mixed/total*100:.1f}%)")
    print(f"Continued (0 keys):      {empty} ({empty/total*100:.1f}%)")

if __name__ == "__main__":
    validate_segmentation("../../sammlung_aller_texte.xml", "../data/texttiling_segments.csv")
