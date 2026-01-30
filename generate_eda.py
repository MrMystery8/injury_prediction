
import pandas as pd
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s') # Cleaner output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    args = parser.parse_args()
    
    df = pd.read_parquet(args.panel)
    
    # Filter using Policy A: Drop truly missing
    # label_known == 1 AND ineligible == 0
    training_set = df[(df['label_known'] == 1) & (df['ineligible'] == 0)].copy()
    
    print(f"Total Rows: {len(df)}")
    print(f"Training Candidates (Known Label + Eligible): {len(training_set)} ({len(training_set)/len(df)*100:.1f}%)")
    print("-" * 50)
    
    # 1. Prevalence by Season
    print("\n### Prevalence by Season")
    season_stats = training_set.groupby('season')['target'].agg(['count', 'mean']).reset_index()
    season_stats['mean'] = season_stats['mean'] * 100
    season_stats.columns = ['Season', 'Weeks', 'Injury Rate (%)']
    print(season_stats.to_markdown(index=False, floatfmt=".2f"))
    
    # 2. Prevalence by Position Group
    print("\n### Prevalence by Position")
    # Simplify position
    def simple_pos(p):
        if pd.isna(p): return 'Unknown'
        p = p.lower()
        if 'goalkeeper' in p: return 'GK'
        if 'defender' in p or 'back' in p: return 'DEF'
        if 'midfield' in p: return 'MID'
        if 'attack' in p or 'winger' in p: return 'FWD'
        return 'Other'
        
    training_set['pos_group'] = training_set['position'].apply(simple_pos)
    pos_stats = training_set.groupby('pos_group')['target'].agg(['count', 'mean']).reset_index()
    pos_stats['mean'] = pos_stats['mean'] * 100
    pos_stats.columns = ['Position', 'Weeks', 'Injury Rate (%)']
    print(pos_stats.to_markdown(index=False, floatfmt=".2f"))

    # 3. Prevalence by Workload Decile (Minutes Last 4 Weeks)
    print("\n### Prevalence by Workload (Minutes Last 4 Weeks)")
    # Filter to >0 minutes to see impact of load vs no load? 
    # Or just all? usually 0-load has mixed risk (recovery vs bench).
    # Let's show all, binning 0 as separate? 
    # Deciles on non-zero is better.
    
    non_zero = training_set[training_set['minutes_last_4w'] > 0].copy()
    zeros = training_set[training_set['minutes_last_4w'] == 0]
    
    if not non_zero.empty:
        non_zero['decile'] = pd.qcut(non_zero['minutes_last_4w'], q=10, labels=False, duplicates='drop')
        
        # Aggregation: 
        # min/max from minutes
        # count/mean from target
        stats = non_zero.groupby('decile').agg({
            'minutes_last_4w': ['min', 'max'],
            'target': ['count', 'mean']
        }).reset_index()
        
        # Flatten cols
        stats.columns = ['Decile', 'Min', 'Max', 'Weeks', 'Rate']
        
        stats['Range'] = stats.apply(lambda x: f"{x['Min']:.0f}-{x['Max']:.0f}", axis=1)
        stats['Injury Rate (%)'] = stats['Rate'] * 100
        
        out = stats[['Decile', 'Range', 'Weeks', 'Injury Rate (%)']]
        print(out.to_markdown(index=False, floatfmt=".2f"))
        
    print(f"\nZero Workload (0 min in 4w): {len(zeros)} weeks, Rate: {zeros['target'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()
