import xml.etree.ElementTree as ET
import re
import numpy as np
import csv
import os
from collections import Counter

# --- UTILS ---
def get_clean_words(tokens):
    """Normalize tokens for math (lowercase, minimum length)."""
    return [t.lower() for t in tokens if re.match(r'[a-zäöüß]{3,}', t.lower())]

def text_tiling_with_sentence_snapping(text, win_size=50):
    # 1. Tokenize keeping original structure (spaces, punct)
    orig_tokens = text.split()
    if len(orig_tokens) < win_size * 2:
        return [text]

    # 2. Identify sentence boundaries (indices of tokens that end sentences)
    # Refined logic: Only snap to tokens ending in . ! or ? if they aren't common
    # abbreviations and the subsequent token begins with an uppercase letter.
    abbreviations = {'lb.', 'thl.', 'pf.', 'u.', 'un.', 'lib.', 'q.', 'nb.', 'p.', 'cap.', 'dr.'}
    sent_boundaries = []
    for i in range(len(orig_tokens) - 1):
        token = orig_tokens[i]
        next_token = orig_tokens[i+1]

        # Check for sentence-final punctuation
        if re.search(r'[.!?]$', token) and token.lower() not in abbreviations:
            # Check if next token starts with an uppercase letter (A-Z or German Umlauts)
            if re.match(r'[A-ZÄÖÜ]', next_token):
                sent_boundaries.append(i)

    if not sent_boundaries:
        sent_boundaries = [len(orig_tokens) - 1]

    # 3. Normalized word stream for math
    math_words = []
    math_to_orig_idx = []
    for i, t in enumerate(orig_tokens):
        if re.match(r'[a-zäöüß]{3,}', t.lower()):
            math_words.append(t.lower())
            math_to_orig_idx.append(i)

    if len(math_words) < win_size * 2:
        return [text]

    # 4. Word-level TextTiling (Math)
    similarities = []
    for i in range(len(math_words) - 2 * win_size + 1):
        win1 = Counter(math_words[i:i+win_size])
        win2 = Counter(math_words[i+win_size:i+2*win_size])

        all_words = set(win1.keys()) | set(win2.keys())
        dot = sum(win1[w] * win2[w] for w in all_words)
        norm1 = np.sqrt(sum(win1[w]**2 for w in win1))
        norm2 = np.sqrt(sum(win2[w]**2 for w in win2))
        similarities.append(dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0)

    if not similarities: return [text]
    threshold = np.mean(similarities) - 0.5 * np.std(similarities)

    # 5. Determine raw boundaries (at math-word level)
    raw_math_boundaries = [i + win_size for i, s in enumerate(similarities) if s < threshold]

    # 6. Map math boundaries to closest validated sentence boundaries
    snapped_boundaries = [0]
    for rb in raw_math_boundaries:
        orig_idx = math_to_orig_idx[rb]
        # Find the closest validated sentence boundary
        closest_sb = min(sent_boundaries, key=lambda x: abs(x - orig_idx))

        if (closest_sb + 1) - snapped_boundaries[-1] > win_size:
            snapped_boundaries.append(closest_sb + 1)

    snapped_boundaries.append(len(orig_tokens))
    final_idx = sorted(list(set(snapped_boundaries)))

    # 7. Extract segments
    segments = []
    for i in range(len(final_idx)-1):
        seg_tokens = orig_tokens[final_idx[i]:final_idx[i+1]]
        segments.append(" ".join(seg_tokens))

    return segments

# --- MAIN EXTRACTION ---
def run_segmentation_pipeline(xml_path, output_csv):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = []
    print(f"Segmenting {xml_path} using TextTiling with Robust Sentence Snapping...")

    for div in root.findall('.//div'):
        text_id = div.get('type', 'unknown')
        div_n = div.get('n', 'unknown')

        full_text_blocks = []
        if div.text: full_text_blocks.append(div.text)
        for child in div:
            if child.tag != 'head':
                if child.text: full_text_blocks.append(child.text)
            if child.tail: full_text_blocks.append(child.tail)

        full_text = " ".join(full_text_blocks)
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        segments = text_tiling_with_sentence_snapping(full_text)

        for i, s in enumerate(segments):
            clean_s = re.sub(r'\s+', ' ', s).strip()
            results.append({
                'text_id': text_id,
                'div_name': div_n,
                'segment_id': i,
                'word_count': len(clean_s.split()),
                'full_text': clean_s
            })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['text_id', 'div_name', 'segment_id', 'word_count', 'full_text'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} segments to {output_csv}")

if __name__ == "__main__":
    run_segmentation_pipeline("../../sammlung_aller_texte.xml", "../data/texttiling_segments_new.csv")
