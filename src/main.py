import os
import pandas as pd
from src.data_loader import load_all_csvs, standardize_datasets
from src.preprocessing import process_injuries, create_panel_base
from src.entity_resolution import resolve_entities
from src.features import calculate_rolling_features, construct_labels, add_injury_history
from src.modeling import time_split_train_test, train_baselines, plot_feature_importance

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"

def main():
    # 1. Load Data
    print("Loading data...")
    datasets = load_all_csvs(RAW_DIR)
    datasets = standardize_datasets(datasets)
    
    if not datasets:
        print("No datasets found in data/raw. Please download them first.")
        return

    # Identify tables
    # Heuristics based on expected filenames
    injuries_df = None
    appearances_df = None
    games_df = None
    players_df = None
    
    for name, df in datasets.items():
        if 'injur' in name:
            injuries_df = df
        elif 'appearan' in name:
            appearances_df = df
        elif 'game' in name and 'appear' not in name:
            games_df = df
        elif 'player' in name and 'appear' not in name and 'injur' not in name:
            players_df = df

    # 2. Process Injuries
    if injuries_df is not None:
        print("Processing injuries...")
        injuries_df = process_injuries(injuries_df)
        injuries_df.to_csv(os.path.join(INTERIM_DIR, "injuries_clean.csv"), index=False)
    else:
        print("Warning: No injury dataset found.")

    # 3. Entity Resolution
    if injuries_df is not None and players_df is not None:
        print("Running entity resolution...")
        unique_inj_names = injuries_df['player_name'].unique()
        mapping_df = resolve_entities(unique_inj_names, players_df)
        mapping_df.to_csv(os.path.join(INTERIM_DIR, "player_id_map_candidates.csv"), index=False)
        print(f"Entity resolution candidates saved. Check {os.path.join(INTERIM_DIR, 'player_id_map_candidates.csv')}")

    # 4. Create Panel
    if appearances_df is not None and games_df is not None:
        print("Creating panel base...")
        panel = create_panel_base(appearances_df, games_df)
        if panel is not None:
            # 5. Feature Engineering
            print("Calculating features...")
            panel = calculate_rolling_features(panel)
            
            # Load map to add player_id to injuries for labeling
            # (Assuming we have a resolved map or using raw names if perfect match - usually map needed)
            if os.path.exists(os.path.join(INTERIM_DIR, "player_id_map_candidates.csv")):
                # In real flow, we'd use the FINAL vetted map. For now using candidates top 1
                map_df = pd.read_csv(os.path.join(INTERIM_DIR, "player_id_map_candidates.csv"))
                # Filter high confidence?
                map_df = map_df.drop_duplicates(subset=['injury_name'])
                
                # Merge player_id into injuries
                injuries_df = injuries_df.merge(map_df[['injury_name', 'player_id']], 
                                              left_on='player_name', right_on='injury_name', how='left')
            
            # Labeling
            print("Constructing labels...")
            panel = construct_labels(panel, injuries_df)
            
            panel.to_parquet(os.path.join(PROCESSED_DIR, "model_table.parquet"))
            print(f"Model table saved. Shape: {panel.shape}")

            # 6. Modeling
            print("Running baseline models...")
            # Basic check if we have enough data
            if panel.shape[0] > 100:
                ft_cols = ['minutes_1w', 'minutes_4w', 'matches_1w', 'acwr_minutes'] # minimal set
                train, test, split_date = time_split_train_test(panel)
                print(f"Split date: {split_date}, Train size: {train.shape}, Test size: {test.shape}")
                
                metrics, lr, lgbm = train_baselines(train, test, ft_cols)
                print("Results:", metrics)
                
                # Plot
                plot_feature_importance(lgbm, ft_cols, "reports/eda/figures/feature_importance.png")

if __name__ == "__main__":
    main()
