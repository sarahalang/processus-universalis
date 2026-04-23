import xml.etree.ElementTree as ET
import re
import os

def count_whitespace_segments(xml_path):
    if not os.path.exists(xml_path):
        xml_path = "../../" + xml_path
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    divs = root.findall('.//div')
    
    total_count = 0
    for div in divs:
        tid = div.get('type', '???')
        
        # Get raw text with ALL whitespace preserved, no tags
        raw_text = "".join(div.itertext())
        
        # Split on newline followed by 0-4 spaces and a non-space character (the I1 pattern)
        raw_segments = re.split(r'\n\s{0,4}(?=[^\s])', raw_text)
        
        # Filter for meaningful blocks (more than 10 words)
        valid_segments = [s for s in raw_segments if len(re.sub(r'\s+', ' ', s).strip().split()) > 10]
        
        count = len(valid_segments)
        total_count += count
        print(f"ID {tid:<6} | Segments: {count}")

    print(f"TOTAL | {total_count}")

if __name__ == "__main__":
    count_whitespace_segments('sammlung_aller_texte.xml')
