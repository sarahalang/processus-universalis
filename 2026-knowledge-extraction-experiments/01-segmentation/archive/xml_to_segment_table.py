import xml.etree.ElementTree as ET
import csv
import os

def parse_processus_xml_by_heading(xml_path):
    """
    Parses the 'sammlung_aller_texte.xml' and creates a list of dicts where
    each segment is defined as the content between <head> tags.
    Aggregates all <keys> found within that span.
    """
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        # Fallback if running from a different directory level
        if os.path.exists("../" + xml_path):
            tree = ET.parse("../" + xml_path)
        else:
            raise

    root = tree.getroot()
    # Get all unique keyword types for columns
    keyword_types = [kw.get('type') for kw in root.findall('.//keywords/keyword') if kw.get('type')]
    
    rows = []

    for div in root.findall('.//div'):
        text_id = div.get('type', 'unknown')
        div_n = div.get('n', 'unknown')
        
        current_segment = None
        
        def finalize_segment(seg):
            if not seg: return None
            # Flatten keys by joining unique non-empty values
            flat_keys = {}
            for k, vals in seg['keys'].items():
                clean_vals = [v for v in vals if v and v.strip() and v.strip() != "FEHLT;"]
                unique_vals = []
                for v in clean_vals:
                    if v not in unique_vals:
                        unique_vals.append(v)
                flat_keys[k] = " ".join(unique_vals) if unique_vals else "N/A"
            
            row = {
                'text_id': seg['text_id'],
                'div_name': seg['div_name'],
                'heading': seg['heading'].strip(),
                'segment_text': seg['content'].strip()
            }
            row.update(flat_keys)
            return row

        initial_text = div.text.strip() if div.text else ""
        current_segment = {
            'text_id': text_id,
            'div_name': div_n,
            'heading': "[Start of Document]",
            'content': initial_text,
            'keys': {k: [] for k in keyword_types}
        }

        for child in div:
            if child.tag == 'head':
                if current_segment and (current_segment['heading'] or current_segment['content']):
                    rows.append(finalize_segment(current_segment))
                
                current_segment = {
                    'text_id': text_id,
                    'div_name': div_n,
                    'heading': child.text.strip() if child.text else "",
                    'content': child.tail.strip() if child.tail else "",
                    'keys': {k: [] for k in keyword_types}
                }
            elif child.tag == 'keys':
                if current_segment:
                    k_type = child.get('type')
                    k_val = child.get('n', '').strip()
                    if k_type in current_segment['keys']:
                        current_segment['keys'][k_type].append(k_val)
                    if child.tail:
                        current_segment['content'] += " " + child.tail.strip()
            else:
                if current_segment:
                    inner = child.text.strip() if child.text else ""
                    tail = child.tail.strip() if child.tail else ""
                    if inner:
                        current_segment['content'] += " " + inner
                    if tail:
                        current_segment['content'] += " " + tail
        
        if current_segment:
            rows.append(finalize_segment(current_segment))

    final_rows = [r for r in rows if r and (r['heading'] or r['segment_text'])]
    return final_rows, keyword_types

if __name__ == "__main__":
    xml_input = "sammlung_aller_texte.xml"
    output_csv = "2026-analyses-summary/processus_segments_by_head.csv"
    
    print(f"Starting extraction by <head> from {xml_input}...")
    try:
        rows, keyword_types = parse_processus_xml_by_heading(xml_input)
        print(f"Extracted {len(rows)} segments.")
        
        if rows:
            headers = ['text_id', 'div_name', 'heading', 'segment_text'] + keyword_types
            with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Table saved to: {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")
