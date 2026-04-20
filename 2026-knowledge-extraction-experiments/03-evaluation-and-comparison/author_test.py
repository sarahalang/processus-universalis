import xml.etree.ElementTree as ET
from collections import defaultdict

GROUP_MAP = {
    'E2': 'I', 'E3': 'I', 'E11': 'I', 'E22': 'I', 'E35': 'I',
    'E16': 'II', 'E17': 'II', 'E19': 'II', 'E27': 'II', 'E32b': 'II',
    'E34': 'III', 'E37': 'III', 'E38': 'III', 'E39': 'III',
    'E42': 'III', 'E44': 'III', 'E45': 'III',
}

A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11',
}

tree = ET.parse("../sammlung_aller_texte.xml")
root = tree.getroot()

# store: group -> list of manuscripts with their attributions
grouped = defaultdict(list)

for div in root.findall(".//div"):
    div_type = div.get("type", "")  # e.g. g1a1

    if "a" not in div_type:
        continue

    # extract a-code
    a_code = f"a{div_type.split('a')[-1]}"

    e_code = A_TO_E.get(a_code)
    if not e_code:
        continue

    group = GROUP_MAP.get(e_code)
    if not group:
        continue

    # extract Zuschreibung
    zuschreibung = None
    for key in div.findall(".//keys"):
        if key.get("type") == "Zuschreibung der Vorschrift":
            zuschreibung = key.get("n")
            break

    # normalize values
    if zuschreibung:
        values = [
            v.strip() for v in zuschreibung.split(";")
            if v.strip() and v.strip() != "FEHLT"
        ]
    else:
        values = []

    # store full manuscript entry
    grouped[group].append({
        "manuscript": div_type,
        "a_code": a_code,
        "e_code": e_code,
        "values": values
    })

# --- output ---
for group in sorted(grouped.keys()):
    print(f"\n=== Group {group} ===")

    for entry in grouped[group]:
        manus = entry["manuscript"]
        values = entry["values"]

        if values:
            joined = "; ".join(values)
        else:
            joined = "[keine Zuschreibung]"

        print(f"{manus}: {joined}")

# --- optional stats ---
print("\n=== Summary ===")
for group, entries in grouped.items():
    total = len(entries)
    with_attr = sum(1 for e in entries if e["values"])
    print(f"{group}: {with_attr}/{total} manuscripts have Zuschreibungen")
