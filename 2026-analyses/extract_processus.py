"""
Extract annotations from Processus Universalis XML into structured formats.
Produces:
  - processus_annotations.csv  (one row per text × category, with individual values)
  - processus_matrix.csv       (presence/absence matrix: texts × categories)
  - processus_data.json        (full structured data)
"""

import xml.etree.ElementTree as ET
import csv
import json
import re
from collections import OrderedDict

XML_PATH = '/Users/slang/claude/processus-sammlung_aller_texte.xml'
tree = ET.parse(XML_PATH)
root = tree.getroot()

# ── Nomenclature mapping ──
# Old A-names (used in XML) → New E-names (current project nomenclature)
A_TO_E = {
    'A1': 'E16', 'A2': 'E37', 'A3': 'E38', 'A4': 'E44', 'A5': 'E17',
    'A6': 'E19', 'A7': 'E39', 'A8': 'E34', 'A9': 'E2', 'A11': 'E43',
    'A12': 'E45', 'A13': 'E42', 'A14': 'E32a', 'A15': 'E32b', 'A16': 'E27',
    'A21': 'E3', 'A22': 'E35', 'A25': 'E22', 'A26': 'E11', 'A28': 'E21',
    'A29': 'E9', 'A31': 'E18',
}
E_TO_A = {v: k for k, v in A_TO_E.items()}

# Old group numbers (in XML) → New group names (current nomenclature)
# Old G1 → Gruppe II, Old G2 → Gruppe III, Old G3 → Gruppe I
OLD_GROUP_TO_NEW = {1: 'II', 2: 'III', 3: 'I'}
NEW_GROUP_TO_OLD = {v: k for k, v in OLD_GROUP_TO_NEW.items()}

# ── 1. Extract master keyword vocabulary ──
master_keywords = OrderedDict()
for kw in root.findall('./keywords/keyword'):
    cat = kw.get('type', '').strip()
    vals = [v.strip() for v in kw.get('n', '').split(';') if v.strip()]
    if cat:
        master_keywords[cat] = vals

CATEGORIES = list(master_keywords.keys())

# ── 2. Extract per-text annotations ──
texts_data = []
for div in root.findall('.//div'):
    text_id = div.get('type', '')        # e.g. "g1a1"
    text_name = div.get('n', '')          # e.g. "A1 Höchster Schatz..."
    date = div.get('when', '')

    # Extract group from type attribute
    group_match = re.match(r'g(\d+)', text_id)
    group = int(group_match.group(1)) if group_match else None

    # Extract text number
    num_match = re.search(r'a(\d+)', text_id)
    text_num = int(num_match.group(1)) if num_match else None

    # Derive old A-name and new E-name
    a_name = f'A{text_num}' if text_num else None
    e_name = A_TO_E.get(a_name) if a_name else None

    # New group nomenclature
    new_group = OLD_GROUP_TO_NEW.get(group) if group else None

    # Collect all <keys> annotations
    annotations = OrderedDict()
    for cat in CATEGORIES:
        annotations[cat] = []

    for keys_el in div.findall('.//keys'):
        cat = keys_el.get('type', '').strip()
        raw = keys_el.get('n', '').strip()
        if not cat:
            continue
        vals = [v.strip() for v in raw.split(';') if v.strip()]
        if cat in annotations:
            annotations[cat] = vals

    # Extract plain text (strip XML tags)
    full_text = ''.join(div.itertext()).strip()
    # Collapse whitespace
    full_text = re.sub(r'\s+', ' ', full_text)

    texts_data.append({
        'text_id': text_id,
        'text_num': text_num,
        'a_name': a_name,
        'e_name': e_name,
        'text_name': text_name,
        'old_group': group,
        'new_group': new_group,
        'date': date if date else None,
        'annotations': annotations,
        'text_length': len(full_text),
        'word_count': len(full_text.split()),
    })

# ── 3. Write detailed CSV (one row per text × category) ──
with open('/Users/slang/claude/processus_annotations.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text_id', 'a_name', 'e_name', 'old_group', 'new_group',
                      'date', 'text_name',
                      'category', 'present', 'value_count', 'values'])
    for t in texts_data:
        for cat in CATEGORIES:
            vals = t['annotations'][cat]
            is_fehlt = vals == ['FEHLT']
            present = 0 if is_fehlt or not vals else 1
            val_count = 0 if is_fehlt else len(vals)
            vals_str = '; '.join(vals) if not is_fehlt else ''
            writer.writerow([
                t['text_id'], t['a_name'], t['e_name'],
                t['old_group'], t['new_group'], t['date'] or '',
                t['text_name'], cat, present, val_count, vals_str
            ])

# ── 4. Write presence/absence matrix CSV ──
with open('/Users/slang/claude/processus_matrix.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text_id', 'a_name', 'e_name', 'old_group', 'new_group',
                      'date', 'text_name', 'word_count'] + CATEGORIES)
    for t in texts_data:
        row = [t['text_id'], t['a_name'], t['e_name'],
               t['old_group'], t['new_group'], t['date'] or '',
               t['text_name'], t['word_count']]
        for cat in CATEGORIES:
            vals = t['annotations'][cat]
            is_fehlt = vals == ['FEHLT']
            row.append(0 if is_fehlt or not vals else 1)
        writer.writerow(row)

# ── 5. Write JSON ──
json_data = {
    'nomenclature': {
        'a_to_e': A_TO_E,
        'e_to_a': E_TO_A,
        'old_group_to_new': {str(k): v for k, v in OLD_GROUP_TO_NEW.items()},
        'new_group_to_old': {k: str(v) for k, v in NEW_GROUP_TO_OLD.items()},
        'note': 'XML uses old A-names and old group numbers (G1/G2/G3). '
                'Current project uses E-names and Roman numeral groups (I/II/III). '
                'Old G1→Gruppe II, Old G2→Gruppe III, Old G3→Gruppe I.',
    },
    'master_vocabulary': master_keywords,
    'categories': CATEGORIES,
    'texts': []
}
for t in texts_data:
    entry = {k: v for k, v in t.items() if k != 'annotations'}
    entry['annotations'] = {}
    for cat in CATEGORIES:
        vals = t['annotations'][cat]
        is_fehlt = vals == ['FEHLT']
        entry['annotations'][cat] = {
            'present': not is_fehlt and len(vals) > 0,
            'values': [] if is_fehlt else vals
        }
    json_data['texts'].append(entry)

with open('/Users/slang/claude/processus_data.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

# ── 6. Write equivalency reference CSV ──
with open('/Users/slang/claude/processus_nomenclature.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['a_name', 'e_name', 'old_group', 'new_group',
                      'xml_id', 'date', 'text_name', 'in_xml'])
    for a_name, e_name in sorted(A_TO_E.items(), key=lambda x: int(re.search(r'\d+', x[0]).group())):
        # Find matching text in corpus (if present)
        match = [t for t in texts_data if t['a_name'] == a_name]
        if match:
            t = match[0]
            writer.writerow([a_name, e_name, t['old_group'], t['new_group'],
                              t['text_id'], t['date'] or '', t['text_name'], 'yes'])
        else:
            writer.writerow([a_name, e_name, '', '', '', '', '', 'no'])

print("Done. Files written:")
print("  processus_annotations.csv   - detailed (one row per text × category)")
print("  processus_matrix.csv        - presence/absence matrix")
print("  processus_data.json         - full structured data")
print("  processus_nomenclature.csv  - A↔E name equivalency reference")
print(f"\n  {len(texts_data)} texts × {len(CATEGORIES)} categories = {len(texts_data)*len(CATEGORIES)} annotation cells")

# Print nomenclature summary
print("\n  Nomenclature mapping (texts in XML corpus marked with *):")
for a_name, e_name in sorted(A_TO_E.items(), key=lambda x: int(re.search(r'\d+', x[0]).group())):
    match = [t for t in texts_data if t['a_name'] == a_name]
    marker = ' *' if match else '  '
    grp = ''
    if match:
        t = match[0]
        grp = f'  G{t["old_group"]}→Gruppe {t["new_group"]}'
    print(f'    {a_name:<5} → {e_name:<5}{marker}{grp}')
