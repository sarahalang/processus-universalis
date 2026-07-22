import csv

def find_similar_headings(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Group by heading
    by_heading = {}
    for row in data:
        h = row['heading'].strip()
        if len(h) < 3: continue # Skip short headings like "A."
        if h not in by_heading:
            by_heading[h] = []
        by_heading[h].append(row)

    # Print headings that appear in multiple text_ids
    print(f"{'Heading':<40} | {'Text IDs'}")
    print("-" * 60)
    for h, rows in by_heading.items():
        if len(rows) > 1:
            tids = ", ".join(set(r['text_id'] for r in rows))
            if len(set(r['text_id'] for r in rows)) > 1:
                print(f"{h[:40]:<40} | {tids}")

if __name__ == "__main__":
    find_similar_headings("2026-analyses-summary/processus_segments_by_head.csv")
