import json
import os
from collections import Counter

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../data/atomic_extraction_results.json")

def analyze_procedural_richness():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_steps = []
    for entry in data:
        for step in entry['atomic_steps']:
            all_steps.append(step)

    # 1. Inventory of Categories
    categories = Counter([s['step_category'] for s in all_steps])
    
    # 2. Inventory of Materials & Apparatus
    materials = []
    apparatus = []
    operations = []
    
    for s in all_steps:
        if s['step_category'] == 'procedural':
            details = s.get('procedural_details', {})
            if details:
                materials.extend(details.get('materials', []))
                apparatus.extend(details.get('apparatus', []))
                operations.append(details.get('operation', ''))

    # 3. Context Analysis (Richness Check)
    contexts = [s.get('context_and_theory', '') for s in all_steps if s.get('context_and_theory')]

    print("\n" + "="*50)
    print("PILOT EXTRACTION QUALITY ANALYSIS")
    print("="*50)
    
    print(f"\n[Basic Stats]")
    print(f"Total Atomic Units: {len(all_steps)}")
    for cat, count in categories.items():
        print(f"  - {cat.capitalize()}: {count}")

    print(f"\n[Top Operations (Verbs)]")
    for op, count in Counter(operations).most_common(10):
        print(f"  - {op}: {count}")

    print(f"\n[Material Inventory (Unique Sample)]")
    unique_materials = sorted(list(set(materials)))
    for mat in unique_materials[:15]:
        print(f"  - {mat}")
    print(f"  ... ({len(unique_materials)} unique substances found)")

    print(f"\n[Apparatus Inventory (Unique Sample)]")
    unique_apparatus = sorted(list(set(apparatus)))
    for app in unique_apparatus[:10]:
        print(f"  - {app}")
    print(f"  ... ({len(unique_apparatus)} unique tools/vessels found)")

    print(f"\n[Qualitative Context Sample]")
    print("-" * 30)
    for ctx in contexts[:5]:
        print(f"  * {ctx}")

    print(f"\n[Dependency Analysis]")
    deps = [s.get('state_dependencies', '') for s in all_steps if s.get('state_dependencies') and s.get('state_dependencies') != "None"]
    print(f"Found {len(deps)} steps with explicit prerequisites.")
    for d in deps[:3]:
        print(f"  - Prereq: {d}")

if __name__ == "__main__":
    analyze_procedural_richness()
