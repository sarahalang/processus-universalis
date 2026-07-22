import xml.etree.ElementTree as ET
import csv
import re
import os

def extract_structural_segments(xml_path):
    """
    Segments the XML based on natural structural boundaries (tags like <head> and <keys>).
    This treats every tag shift as a potential paragraph/procedural boundary.
    """
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        if os.path.exists("../" + xml_path):
            tree = ET.parse("../" + xml_path)
        else:
            raise
    
    root = tree.getroot()
    results = []
    total_segments = 0
    
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
        div_n = div.get('n', 'unknown')
        
        # A 'natural' segment is text found at the start of a div or after any child tag
        # The XML structure uses these tags to mark shifts in topic/step, 
        # which align with the original manuscript indentation (Level 1).
        
        segments = []
        
        # 1. Check for text at the very beginning of the div
        if div.text and div.text.strip():
            segments.append(div.text.strip())
            
        # 2. Check for text following each child tag (tail text)
        for child in div:
            if child.tail and child.tail.strip():
                segments.append(child.tail.strip())
        
        for i, s in enumerate(segments):
            # Clean up whitespace and line breaks
            clean_text = re.sub(r'\s+', ' ', s).strip()
            w_count = len(clean_text.split())
            
            # Filter out very short segments (e.g. single numbers or accidental artifacts)
            if w_count > 5:
                results.append({
                    'text_id': tid,
                    'div_name': div_n,
                    'segment_id': i,
                    'word_count': w_count,
                    'full_text': clean_text
                })
                total_segments += 1
                
    return results, total_segments

if __name__ == "__main__":
    xml_input = "sammlung_aller_texte.xml"
    output_csv = "2026-knowledge-extraction-experiments/data/structural_segments.csv"
    
    print(f"Starting structural (paragraph-based) segmentation from {xml_input}...")
    try:
        data, count = extract_structural_segments(xml_input)
        
        headers = ['text_id', 'div_name', 'segment_id', 'word_count', 'full_text']
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            
        print(f"Success! Extracted {count} natural structural segments.")
        print(f"Results saved to {output_csv}")
        
        # Summary stats
        counts = [d['word_count'] for d in data]
        print(f"Min Words: {min(counts)}")
        print(f"Max Words: {max(counts)}")
        print(f"Mean Words: {sum(counts)/len(counts):.1f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
