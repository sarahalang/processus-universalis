import xml.etree.ElementTree as ET
import csv
import re
import os
from collections import Counter

# --- 1. CONFIGURATION & ANCHORS ---
STOPWORDS = set([
    'und', 'der', 'die', 'das', 'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
    'in', 'auf', 'mit', 'zu', 'von', 'aus', 'nach', 'bei', 'für', 'an', 'über',
    'um', 'vor', 'durch', 'als', 'wie', 'so', 'ist', 'sind', 'war', 'werden',
    'wird', 'wurde', 'kann', 'muss', 'soll', 'nicht', 'nur', 'doch', 'aber',
    'auch', 'noch', 'schon', 'jetzt', 'dann', 'da', 'hier', 'dort', 'wenn',
    'daß', 'dass', 'denn', 'man', 'sich', 'uns', 'wir', 'ihr', 'sie', 'es',
    'ich', 'du', 'mein', 'dein', 'sein', 'ihr', 'ihre', 'dies', 'dieser',
    'dieses', 'diesen', 'diesem', 'alle', 'allem', 'allen', 'aller', 'alles',
    'welche', 'welches'
])

def get_technical_anchors(xml_path, top_n=150):
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        if os.path.exists("../" + xml_path):
            tree = ET.parse("../" + xml_path)
        else:
            raise
    
    root = tree.getroot()
    
    # 1. Extract words from expert keywords
    expert_words = set()
    for kw in root.findall('.//keywords/keyword'):
        n_attr = kw.get('n', '')
        # Split by semicolon, then by space, filter words
        for phrase in n_attr.split(';'):
            for word in re.findall(r'\b[a-zäöüß]{4,}\b', phrase.lower()):
                if word not in STOPWORDS:
                    expert_words.add(word)
    
    # 2. Extract common long words from text
    all_text = " ".join([t.strip() for t in root.itertext() if t.strip()]).lower()
    words = re.findall(r'\b[a-zäöüß]{5,}\b', all_text)
    filtered = [w for w in words if w not in STOPWORDS]
    counts = Counter(filtered)
    
    # Combine both, prioritizing expert words
    combined = list(expert_words)
    common_words = [w for w, c in counts.most_common(top_n) if w not in expert_words]
    combined.extend(common_words)
    
    return combined[:top_n]

def calculate_similarity(v1, v2):
    set1 = set([i for i, val in enumerate(v1) if val > 0])
    set2 = set([i for i, val in enumerate(v2) if val > 0])
    if not set1 or not set2: 
        # If both are empty, they are "similar" in their lack of info
        if not set1 and not set2: return 1.0
        return 0.0 # One has info, other doesn't
    return len(set1 & set2) / len(set1 | set2)

# --- 2. IMPROVED SENTENCE-AWARE SEGMENTATION ENGINE ---
def segment_document(text, anchors, threshold=0.4, max_words=200):
    """
    Segments text only at sentence boundaries where possible.
    If a sentence itself is too long, it will be split.
    """
    # 1. Improved Sentence Splitting
    clean_text = re.sub(r'\s+', ' ', text)
    # Split on punctuation followed by space and Capital
    sentence_breaks = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', clean_text)
    sentences = [s.strip() for s in sentence_breaks if len(s.strip()) > 5]
    
    if not sentences:
        return [clean_text]

    # 2. Vectorize each sentence
    sent_vectors = []
    for s in sentences:
        s_lower = s.lower()
        vec = [1 if a in s_lower else 0 for a in anchors]
        sent_vectors.append(vec)

    # 3. Find boundaries using lexical shifts
    win_size = 1 # Smaller window for more sensitivity
    boundaries = [0]
    for i in range(win_size, len(sentences) - win_size):
        # Look at the sentence before and after
        past_win = [max(col) for col in zip(*sent_vectors[i-win_size:i])]
        future_win = [max(col) for col in zip(*sent_vectors[i:i+win_size])]
        sim = calculate_similarity(past_win, future_win)
        if sim < threshold:
            boundaries.append(i)
    
    boundaries.append(len(sentences))
    unique_b = sorted(list(set(boundaries)))
    
    # 4. Group sentences and handle outliers
    final_segs = []
    for k in range(len(unique_b)-1):
        seg_sentences = sentences[unique_b[k]:unique_b[k+1]]
        seg_text = " ".join(seg_sentences)
        
        # If the block of sentences is too long, split by sentence count
        if len(seg_text.split()) > max_words:
            if len(seg_sentences) > 1:
                mid = len(seg_sentences) // 2
                final_segs.append(" ".join(seg_sentences[:mid]))
                final_segs.append(" ".join(seg_sentences[mid:]))
            else:
                # This is a single, massive "sentence". We must force-split it.
                words = seg_text.split()
                # Split into chunks of max_words
                for i in range(0, len(words), max_words):
                    final_segs.append(" ".join(words[i:i+max_words]))
        else:
            final_segs.append(seg_text)
            
    # Recursive check: if any segment is still too long, run one more time with higher threshold
    fully_refined = []
    for s in final_segs:
        if len(s.split()) > max_words + 20:
            fully_refined.extend(segment_document(s, anchors, threshold + 0.1, max_words))
        else:
            fully_refined.append(s)
            
    return fully_refined

# --- 3. PROCESSING ---
def run_unsupervised_pipeline(xml_path):
    print("Stage 1: Discovering Anchors...")
    anchors = get_technical_anchors(xml_path)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    results = []
    total_segments = 0
    
    print("Stage 2: Refined Sentence Segmentation (Target 300 words)...")
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
        div_n = div.get('n', 'unknown')
        
        raw_text = " ".join([t.strip() for t in div.itertext() if t.strip()])
        segs = segment_document(raw_text, anchors)
        
        for i, s in enumerate(segs):
            s_lower = s.lower()
            found_anchors = [a for a in anchors if a in s_lower]
            w_count = len(s.split())
            
            if w_count > 10:
                results.append({
                    'text_id': tid,
                    'div_name': div_n,
                    'seg_id': i,
                    'word_count': w_count,
                    'top_anchors': ", ".join(found_anchors[:10]),
                    'full_text': s.strip()
                })
                total_segments += 1
            
    return results, total_segments

if __name__ == "__main__":
    xml_file = "sammlung_aller_texte.xml"
    output_csv = "2026-analyses-summary/unsupervised_segments_full.csv"
    
    try:
        data, count = run_unsupervised_pipeline(xml_file)
        
        headers = ['text_id', 'div_name', 'seg_id', 'word_count', 'top_anchors', 'full_text']
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"\nSuccess! Created {count} refined segments.")
        print(f"Results saved to {output_csv}")
        
        counts = [d['word_count'] for d in data]
        print(f"Max Words: {max(counts)}")
        print(f"Mean Words: {sum(counts)/len(counts):.1f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
