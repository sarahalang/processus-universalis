import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from itertools import combinations

# Mapping from E-codes to Groups
GROUP_MAP = {
    'E2': 'I', 'E3': 'I', 'E11': 'I', 'E22': 'I', 'E35': 'I',
    'E16': 'II', 'E17': 'II', 'E19': 'II', 'E27': 'II', 'E32b': 'II',
    'E34': 'III', 'E37': 'III', 'E38': 'III', 'E39': 'III',
    'E42': 'III', 'E44': 'III', 'E45': 'III',
}

# a-code → E-code mapping
A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11',
}

# Parse XML
tree = ET.parse("../sammlung_aller_texte.xml")
root = tree.getroot()

# Store: list of (manuscript, group, heading)
all_headings = []

for div in root.findall(".//div"):
    div_type = div.get("type", "")
    if "a" not in div_type:
        continue

    a_code = f"a{div_type.split('a')[-1]}"
    e_code = A_TO_E.get(a_code)
    if not e_code:
        continue

    group = GROUP_MAP.get(e_code)
    if not group:
        continue

    # Find ALL <head> elements in this div
    head_elements = div.findall(".//head")
    if not head_elements:
        # If no <head>, use placeholder
        all_headings.append({
            "manuscript": div_type,
            "group": group,
            "heading": "[keine Überschrift]"
        })
        continue

    # Extract text from each <head>
    for head_elem in head_elements:
        heading_text = head_elem.text.strip() if head_elem.text else ""
        # Handle empty or "FEHLT" values
        if not heading_text or heading_text == "FEHLT":
            heading_text = "[keine Überschrift]"
        all_headings.append({
            "manuscript": div_type,
            "group": group,
            "heading": heading_text
        })

# --- Output: ALL headings, one per line (all texts, all heads) ---
print("\n" + "="*80)
print("ALL HEADINGS (ALL TEXTS, ALL HEADINGS, ONE PER LINE)")
print("="*80)

# Sort by manuscript for consistency
sorted_headings = sorted(all_headings, key=lambda x: x["manuscript"])

for entry in sorted_headings:
    print(f"{entry['manuscript']:>8} | {entry['group']:>2} | {entry['heading']}")

# --- Optional: Summary Statistics ---
print("\n" + "="*80)
print("SUMMARY: Heading Analysis")
print("="*80)

# Group by group
grouped = defaultdict(list)
for entry in all_headings:
    grouped[entry["group"]].append(entry)

for group in sorted(grouped.keys()):
    entries = grouped[group]
    headings = [e["heading"] for e in entries]
    counter = Counter(headings)
    unique_headings = len(set(headings))

    print(f"\nGroup {group}: {len(entries)} headings (from {len(set(e['manuscript'] for e in entries))} texts)")
    print(f"  Unique headings: {unique_headings}")
    print(f"  Most common heading: {counter.most_common(1)[0][0]} ({counter.most_common(1)[0][1]} times)")
    print(f"  Top 5 headings:")
    for h, count in counter.most_common(5):
        print(f"    {h} ({count})")

# --- Optional: Find common headings across groups ---
print("\n" + "="*80)
print("COMMON HEADINGS ACROSS GROUPS")
print("="*80)

all_headings_by_group = {}
for group, entries in grouped.items():
    all_headings_by_group[group] = set(e["heading"] for e in entries)

shared = {}
for g1, g2 in combinations(sorted(all_headings_by_group.keys()), 2):
    common = all_headings_by_group[g1] & all_headings_by_group[g2]
    if common:
        shared[f"{g1} ↔ {g2}"] = sorted(common)

for pair, common_headings in shared.items():
    print(f"\n{pair}:")
    for h in common_headings:
        print(f"  - {h}")
