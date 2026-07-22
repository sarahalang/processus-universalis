import csv
import xml.etree.ElementTree as ET
import re
import os
from collections import defaultdict

# --- 1. SETUP ---
STOPWORDS = set([
    'und', 'der', 'die', 'das', 'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
    'in', 'auf', 'mit', 'zu', 'von', 'aus', 'nach', 'bei', 'für', 'an', 'über',
    'um', 'vor', 'durch', 'als', 'wie', 'so', 'ist', 'sind', 'war', 'werden',
    'wird', 'wurde', 'kann', 'muss', 'soll', 'nicht', 'nur', 'doch', 'aber',
    'auch', 'noch', 'schon', 'jetzt', 'dann', 'da', 'hier', 'dort', 'wenn',
    'daß', 'dass', 'denn', 'man', 'sich', 'uns', 'wir', 'ihr', 'sie', 'es',
    'ich', 'du', 'mein', 'dein', 'sein', 'ihr', 'ihre', 'dies', 'dieser',
    'dieses', 'diesen', 'diesem', 'alle', 'allem', 'allen', 'aller', 'alles',
    'welche', 'welches', 'den', 'dem', 'des', 'am', 'im', 'zum', 'zur', 'vom',
    'bis', 'über', 'unter', 'nach', 'vor', 'hinter', 'neben', 'zwischen',
    'wieder', 'ganz', 'sehr', 'etwa', 'etwan', 'noch', 'schon', 'jetzt', 'dann',
    'daher', 'damit', 'dazu', 'dabei', 'daran', 'darauf', 'darin', 'darunter',
    'als', 'also', 'aber', 'oder', 'sondern', 'doch', 'jedoch', 'allein',
    'daß', 'dass', 'obwohl', 'obgleich', 'wenngleich', 'wenn', 'weil', 'da',
    'indem', 'während', 'solange', 'sobald', 'nachdem', 'bevor', 'ehe',
    'man', 'jemand', 'niemand', 'alle', 'viele', 'manche', 'einige', 'etliche',
    'etwas', 'nichts', 'alles', 'beide', 'zwei', 'drei', 'vier', 'fünf',
    'haben', 'sein', 'werden', 'können', 'müssen', 'sollen', 'wollen', 'dürfen',
    'mögen', 'machen', 'tun', 'gehen', 'kommen', 'lassen', 'geben', 'sehen',
    'sagen', 'wissen', 'nein', 'nicht'
])

def cologne_phonetic(word):
    word = word.lower().strip()
    if not word: return ''
    code = []
    prev_code = ''
    for i, ch in enumerate(word):
        before = word[i-1] if i > 0 else ''
        after = word[i+1] if i < len(word) - 1 else ''
        c = ''
        if ch in 'aeiouäöüjyàáâãåèéêëìíîïòóôõùúûýÿ': c = '0'
        elif ch == 'h': c = ''
        elif ch in 'bp': c = '1'
        elif ch in 'dt': c = '8' if after in 'csz' else '2'
        elif ch in 'fvw': c = '3'
        elif ch in 'gkq': c = '4'
        elif ch == 'c': c = '4' if after in 'ahkoqux' else '8'
        elif ch == 'x': c = '8' if before in 'ckq' else '48'
        elif ch == 'l': c = '5'
        elif ch in 'mn': c = '6'
        elif ch == 'r': c = '7'
        elif ch in 'szßẞ': c = '8'
        if c and c != prev_code:
            code.append(c)
            prev_code = c[-1] if c else ''
        elif c: prev_code = c[-1] if c else ''
    result = ''.join(code)
    if result: result = result[0] + result[1:].replace('0', '')
    return result

def load_expert_definitions(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    definitions = {} # {category: {phonetic_code: original_word}}
    word_to_categories = defaultdict(set)
    
    for kw in root.findall('.//keywords/keyword'):
        ctype = kw.get('type')
        if not ctype: continue
        
        n_attr = kw.get('n', '').lower()
        # Increase min length to 4
        words = set(re.findall(r'\b[a-zäöüß]{4,}\b', n_attr))
        words = {w for w in words if w not in STOPWORDS}
        
        if len(words) < 3: continue 
        
        definitions[ctype] = {}
        for w in words:
            code = cologne_phonetic(w)
            if code:
                definitions[ctype][code] = w
                word_to_categories[code].add(ctype)
    
    # Calculate weights based on how many categories a word appears in
    weights = {code: 1.0 / (len(cats)**0.5) for code, cats in word_to_categories.items()}
    
    return definitions, weights

def load_ground_truth_map(xml_path):
    """
    Builds a map of {tid: [(snippet, category), ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    gt_map = defaultdict(list)
    for div in root.findall('.//div'):
        tid = div.get('type', 'unknown')
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
                        if sibling.tail and sibling.tail.strip():
                            snippet = re.sub(r'\s+', ' ', sibling.tail).strip()[:50]
                            break
                if snippet:
                    gt_map[tid].append((snippet, k_type))
    return gt_map

def predict_category(text, definitions, weights):
    # Vectorize text using phonetic codes
    text_words = re.findall(r'\b[a-zäöüß]{4,}\b', text.lower())
    text_mapping = {}
    for w in text_words:
        if w not in STOPWORDS:
            code = cologne_phonetic(w)
            if code: text_mapping[code] = w
            
    text_set = set(text_mapping.keys())
    
    scores = {}
    matches_info = {}
    for ctype, def_map in definitions.items():
        score = 0
        def_codes = set(def_map.keys())
        overlap = text_set & def_codes
        
        matched_words = []
        for code in overlap:
            score += weights[code]
            matched_words.append(text_mapping[code])
            
        # Jaccard-like score with weights
        union_codes = text_set | def_codes
        union_weight = sum(weights[c] for c in union_codes if c in weights)
        
        if union_weight > 0:
            scores[ctype] = score / union_weight
        else:
            scores[ctype] = 0
        matches_info[ctype] = matched_words
        
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:3], matches_info

# --- 3. MAIN ---
if __name__ == "__main__":
    xml_file = "sammlung_aller_texte.xml"
    csv_file = "2026-analyses-summary/unsupervised_segments_full.csv"
    
    print("Loading definitions and ground truth (Refined)...")
    definitions, weights = load_expert_definitions(xml_file)
    gt_map = load_ground_truth_map(xml_file)
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        segments = list(reader)
    
    print(f"Analyzing {len(segments)} segments...")
    
    correct_top1 = 0
    correct_top3 = 0
    total_evaluated = 0
    
    results = []
    
    for seg in segments:
        tid = seg['text_id']
        seg_text = seg['full_text']
        
        true_keys = []
        for snippet, k_type in gt_map.get(tid, []):
            if snippet in seg_text:
                true_keys.append(k_type)
        
        if not true_keys: continue
            
        total_evaluated += 1
        predictions, matches_info = predict_category(seg_text, definitions, weights)
        pred_top1 = predictions[0][0] if predictions[0][1] > 0 else None
        pred_top3 = [p[0] for p in predictions if p[1] > 0]
        
        match_top1 = pred_top1 in true_keys
        match_top3 = any(k in true_keys for k in pred_top3)
        
        if match_top1: correct_top1 += 1
        if match_top3: correct_top3 += 1
        
        results.append({
            'tid': tid,
            'true_keys': ", ".join(true_keys),
            'pred_top1': pred_top1,
            'score_top1': predictions[0][1],
            'matches': ", ".join(matches_info.get(pred_top1, [])[:5]) if pred_top1 else "",
            'match': "YES" if match_top1 else "NO"
        })

    print("\n" + "="*60)
    print("INFORMATION EXTRACTION RECONSTRUCTION REPORT (REFINED)")
    print("="*60)
    print(f"Segments with Expert Keys: {total_evaluated}")
    print(f"Top-1 Accuracy: {correct_top1/total_evaluated*100:.1f}%")
    print(f"Top-3 Accuracy: {correct_top3/total_evaluated*100:.1f}%")
    print("-" * 60)
    
    print("\nSample Errors (Where prediction != expert):")
    errors = [r for r in results if r['match'] == "NO"]
    for err in errors[:10]:
        print(f"[{err['tid']}] Expert: {err['true_keys']:<30} | Predicted: {err['pred_top1']} ({err['score_top1']:.2f})")
        print(f"       Matches: {err['matches']}")
