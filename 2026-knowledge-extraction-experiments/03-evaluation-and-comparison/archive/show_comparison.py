import csv

def show_detailed_comparison(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 1. Comparison: Multiplicatio
    print("=== COMPARISON 1: Multiplicatio (Checking for Phrasing vs Keys) ===")
    m_ids = ['g1a1', 'g2a12', 'g1a16']
    for row in data:
        if row['text_id'] in m_ids and "multiplicatio" in row['heading'].lower():
            print(f"[{row['text_id']}] Heading: {row['heading']}")
            print(f"Keys: {row['Multiplikation']}")
            print(f"Text Preview: {row['segment_text'][:400]}...")
            print("-" * 50)
    
    print("\n" + "="*80 + "\n")

    # 2. Comparison: Athanor
    print("=== COMPARISON 2: Athanor (Identical Keys, checking if Text is identical) ===")
    a_ids = ['g2a2', 'g2a8', 'g2a11', 'g2a12']
    for row in data:
        if row['text_id'] in a_ids and "athanor" in row['heading'].lower():
            print(f"[{row['text_id']}] Heading: {row['heading']}")
            print(f"Keys: {row['Beschreibung des Athanors']}")
            print(f"Text Preview: {row['segment_text'][:400]}...")
            print("-" * 50)

if __name__ == "__main__":
    show_detailed_comparison("2026-analyses-summary/processus_segments_by_head.csv")
