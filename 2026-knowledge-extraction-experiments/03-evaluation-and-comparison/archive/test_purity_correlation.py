import pandas as pd
import os
from scipy.stats import pearsonr

# Load the composition we just saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "../data/cluster_composition.csv")

if not os.path.exists(csv_path):
    print("Error: cluster_composition.csv not found.")
else:
    df = pd.read_csv(csv_path)

    # Calculate Purity (Absolute dominance of one type)
    # 100 = Pure Procedural or Pure Descriptive
    # 0   = Perfectly Balanced 50/50
    df['Purity'] = (df['Proc%'] - df['Desc%']).abs()

    # Correlate Size vs Purity
    r, p = pearsonr(df['Total'], df['Purity'])

    print("\n" + "="*50)
    print("HYPOTHESIS TEST: DOES SIZE DESTROY PURITY?")
    print("="*50)
    print(f"Pearson Correlation (Size vs. Purity): r = {r:.3f}")
    print(f"P-value: {p:.4e}")
    print("-" * 50)

    # Look at Purity by Size Tiers
    print("\nAVERAGE PURITY BY CLUSTER SIZE:")
    df['SizeTier'] = pd.cut(df['Total'], bins=[0, 2, 5, 10, 100], 
                            labels=['Unique/Duo (1-2)', 'Small (3-5)', 'Medium (6-10)', 'Large (11+)'])
    tier_stats = df.groupby('SizeTier', observed=False).agg({
        'Purity': 'mean',
        'cluster_label': 'count'
    }).rename(columns={'cluster_label': 'Cluster Count'})
    
    print(tier_stats.round(2).to_string())
    
    print("\nCONCLUSION:")
    if r < -0.3:
        print("CONFIRMED: Larger clusters are significantly more 'Hybrid'.")
        print("The 'Shared Tradition' is conceptually mixed, while only rare steps are pure.")
    else:
        print("REJECTED: Purity remains stable regardless of cluster size.")
