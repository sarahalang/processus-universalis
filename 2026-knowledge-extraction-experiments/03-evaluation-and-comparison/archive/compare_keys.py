import csv

def compare_segments(csv_path, heading_to_find, key_col):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        matches = [row for row in reader if heading_to_find.lower() in row['heading'].lower()]
    
    print(f"Comparison for Heading like '{heading_to_find}'")
    print(f"{'Text ID':<10} | {'Key: ' + key_col}")
    print("-" * 100)
    for m in matches:
        print(f"{m['text_id']:<10} | {m.get(key_col, 'N/A')}")

if __name__ == "__main__":
    compare_segments("2026-analyses-summary/processus_segments_by_head.csv", "Multiplicatio", "Multiplikation")
    print("\n")
    compare_segments("2026-analyses-summary/processus_segments_by_head.csv", "Athanor", "Beschreibung des Athanors")
