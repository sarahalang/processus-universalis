import csv
import json
import re

def compare_extraction_to_headings():
    # Load the expert segments (headings as truth)
    expert_segments = []
    with open("2026-analyses-summary/processus_segments_by_head.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        expert_segments = list(reader)

    # Load extracted JSON knowledge
    with open("2026-analyses-summary/extracted_opac_knowledge.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)

    print(f"{'Text ID':<8} | {'Heading (Expert)':<30} | {'Main Operation (Extracted)'}")
    print("-" * 110)

    # We match by text_id and a simplified content-similarity (first 50 chars)
    for exp in expert_segments:
        tid = exp['text_id']
        head = exp['heading']
        expert_content = exp['segment_text'][:50] # Snippet for matching

        # Find matching segments in the extracted JSON
        matches = [e for e in extracted if e['text_id'] == tid]
        
        # Look for the extracted segment whose text most likely matches this expert heading
        found_ops = []
        for m in matches:
            # We assume a heuristic: if segment text snippet matches expert text snippet
            # (Note: In a production system, we'd map segment IDs explicitly)
            found_ops = [s.get('haupt_vorgang') for s in m.get('schritte', [])]
        
        print(f"{tid:<8} | {head[:30]:<30} | {str(found_ops[:2])}")

if __name__ == "__main__":
    compare_extraction_to_headings()
