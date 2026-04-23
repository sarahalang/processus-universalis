import csv
from collections import defaultdict

def analyze_segmentation_quality(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Group segments by text_id
    text_stats = defaultdict(list)
    for row in data:
        text_id = row['text_id']
        content = row['segment_text']
        word_count = len(content.split())
        text_stats[text_id].append(word_count)

    print(f"{'Text ID':<10} | {'Seg Count':<10} | {'Avg Words':<10} | {'Max Words':<10} | {'Min Words':<10}")
    print("-" * 65)
    
    # Sort by number of segments (ascending)
    sorted_stats = sorted(text_stats.items(), key=lambda x: len(x[1]))
    
    for tid, counts in sorted_stats:
        avg_w = sum(counts) / len(counts)
        max_w = max(counts)
        min_w = min(counts)
        seg_count = len(counts)
        print(f"{tid:<10} | {seg_count:<10} | {avg_w:<10.1f} | {max_w:<10} | {min_w:<10}")

    # Summary
    total_texts = len(text_stats)
    # Filter for texts that have very few segments
    # Note: Every text has at least one '[Start of Document]' segment
    one_seg_texts = [tid for tid, counts in text_stats.items() if len(counts) <= 2]
    
    print("\n" + "="*65)
    print(f"Total Texts Analyzed: {total_texts}")
    print(f"Texts with very few segments (<= 2): {len(one_seg_texts)}")
    if one_seg_texts:
        print(f"Outliers: {', '.join(one_seg_texts)}")

if __name__ == "__main__":
    analyze_segmentation_quality("2026-analyses-summary/processus_segments_by_head.csv")
