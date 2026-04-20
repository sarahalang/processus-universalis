import xml.etree.ElementTree as ET
import re
import numpy as np
import csv
import os
from collections import Counter

# --- TEXTTILING LOGIC ---
def get_words(text):
    return [w.lower() for w in re.findall(r'\b[a-zäöüß]{3,}\b', text.lower())]

def text_tiling(text, win_size=50):
    words = get_words(text)
    if len(words) < win_size * 2: 
        return [text]
    
    similarities = []
    for i in range(len(words) - 2 * win_size + 1):
        win1 = Counter(words[i:i+win_size])
        win2 = Counter(words[i+win_size:i+2*win_size])
        
        all_words = set(win1.keys()) | set(win2.keys())
        dot = sum(win1[w] * win2[w] for w in all_words)
        norm1 = np.sqrt(sum(win1[w]**2 for w in win1))
        norm2 = np.sqrt(sum(win2[w]**2 for w in win2))
        similarities.append(dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0)
    
    # Identify boundaries using a valley threshold
    if not similarities: return [text]
    threshold = np.mean(similarities) - 0.5 * np.std(similarities)
    boundaries = [i + win_size for i, s in enumerate(similarities) if s < threshold]
    
    refined = [0]
    for b in boundaries:
        if b - refined[-1] > win_size:
            refined.append(b)
    refined.append(len(words))
    
    return [" ".join(words[refined[i]:refined[i+1]]) for i in range(len(refined)-1)]

# --- MAIN EXTRACTION ---
def run_segmentation_pipeline(xml_path, output_csv):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    results = []
    print(f"Segmenting {xml_path} using TextTiling...")
    
    for div in root.findall('.//div'):
        text_id = div.get('type', 'unknown')
        div_n = div.get('n', 'unknown')
        
        # Keep raw text (stripping tags but maintaining flow)
        full_text = "".join(div.itertext())
        
        segments = text_tiling(full_text)
        
        for i, s in enumerate(segments):
            results.append({
                'text_id': text_id,
                'div_name': div_n,
                'segment_id': i,
                'word_count': len(s.split()),
                'full_text': s
            })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['text_id', 'div_name', 'segment_id', 'word_count', 'full_text'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved {len(results)} segments to {output_csv}")

if __name__ == "__main__":
    run_segmentation_pipeline("../../sammlung_aller_texte.xml", "data/texttiling_segments.csv")
