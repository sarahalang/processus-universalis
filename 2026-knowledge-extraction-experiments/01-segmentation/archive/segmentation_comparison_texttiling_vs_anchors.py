import xml.etree.ElementTree as ET
import re
import numpy as np
from collections import Counter

# --- UTILS ---
def get_words(text):
    return [w.lower() for w in re.findall(r'\b[a-zäöüß]{3,}\b', text.lower())]

def load_data(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {div.get('type'): "".join(div.itertext()) for div in root.findall('.//div')}

# --- 1. TEXTTILING (Global Vocabulary Shift) ---
def text_tiling(text, win_size=50):
    words = get_words(text)
    if len(words) < win_size * 2: return [text]
    
    similarities = []
    for i in range(len(words) - 2 * win_size + 1):
        win1 = Counter(words[i:i+win_size])
        win2 = Counter(words[i+win_size:i+2*win_size])
        
        all_words = set(win1.keys()) | set(win2.keys())
        dot = sum(win1[w] * win2[w] for w in all_words)
        norm1 = np.sqrt(sum(win1[w]**2 for w in win1))
        norm2 = np.sqrt(sum(win2[w]**2 for w in win2))
        similarities.append(dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0)
    
    threshold = np.mean(similarities) - 0.5 * np.std(similarities)
    boundaries = [i + win_size for i, s in enumerate(similarities) if s < threshold]
    
    refined = [0]
    for b in boundaries:
        if b - refined[-1] > win_size:
            refined.append(b)
    refined.append(len(words))
    
    return [" ".join(words[refined[i]:refined[i+1]]) for i in range(len(refined)-1)]

# --- 2. AUTO-ANCHORS (Technical Term Shift) ---
def get_auto_anchors(all_texts, top_n=50):
    all_words = []
    for text in all_texts:
        all_words.extend(get_words(text))
    
    counts = Counter(all_words)
    # Heuristic: top words that are neither extremely common nor rare
    return [w for w, c in counts.most_common(200) if c > 5 and len(w) > 4][:top_n]

def anchor_segmentation(text, anchors, win_size=50):
    words = get_words(text)
    if len(words) < win_size * 2: return [text]
    
    vectors = []
    for i in range(0, len(words) - win_size + 1, win_size // 2):
        win = words[i:i+win_size]
        vec = np.array([1 if a in win else 0 for a in anchors])
        vectors.append(vec)
        
    boundaries = [0]
    for i in range(len(vectors)-1):
        v1, v2 = vectors[i], vectors[i+1]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        sim = np.dot(v1, v2) / (norm + 1e-9)
        if sim < 0.2: 
            boundaries.append(i * (win_size // 2))
    
    boundaries.append(len(words))
    return [" ".join(words[boundaries[i]:boundaries[i+1]]) for i in range(len(boundaries)-1)]

# --- MAIN ---
if __name__ == "__main__":
    xml_path = "../../sammlung_aller_texte.xml"
    texts = load_data(xml_path)
    anchors = get_auto_anchors(texts.values())
    
    print(f"{'ID':<6} | {'TextTiling':<12} | {'AutoAnchors':<12}")
    print("-" * 40)
    
    stats = {'TextTiling': [], 'AutoAnchors': []}
    
    for tid, text in texts.items():
        tt_segs = text_tiling(text)
        aa_segs = anchor_segmentation(text, anchors)
        print(f"{tid:<6} | {len(tt_segs):<12} | {len(aa_segs):<12}")
        
        stats['TextTiling'].extend([len(s.split()) for s in tt_segs])
        stats['AutoAnchors'].extend([len(s.split()) for s in aa_segs])
    
    print("\n--- Statistics (Word Count per Segment) ---")
    for method, lengths in stats.items():
        print(f"{method:<12} | Mean: {np.mean(lengths):.1f} | Median: {np.median(lengths):.1f} | Std: {np.std(lengths):.1f}")

    # Quality Check: Distribution Sanity
    print("\n--- Quality Check (Distribution of Segment Lengths) ---")
    for method, lengths in stats.items():
        small = sum(1 for l in lengths if l < 20)
        large = sum(1 for l in lengths if l > 300)
        print(f"{method:<12} | Segments < 20 words: {small} | Segments > 300 words: {large}")

    print("\n--- Detailed Consecutive Samples ---")
    for tid in ['g1a1', 'g2a4', 'g1a15']:
        text = texts.get(tid, '')
        tt_segs = text_tiling(text)
        aa_segs = anchor_segmentation(text, anchors)
        
        print(f"\nDocument: {tid}")
        print("-" * 20)
        print(f"TextTiling (First 3 segments):")
        for i, s in enumerate(tt_segs[:3]):
            print(f"  {i}: [{len(s.split())} words] {s}")
            
        print(f"\nAutoAnchors (First 3 segments):")
        for i, s in enumerate(aa_segs[:3]):
            print(f"  {i}: [{len(s.split())} words] {s}")
