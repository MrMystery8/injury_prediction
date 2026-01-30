
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from src.data.loader import load_backbone, load_labels, detect_tables

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_top5_purity(panel, games, competitions):
    """
    Check A: Verify only Top-5 league games are in the pipeline.
    """
    logger.info("--- Check A: Top-5 Purity ---")
    
    # Get distinct competition IDs in panel
    panel_comp_ids = panel['competition_id'].unique()
    
    # Join to competitions to see names/types
    relevant_comps = competitions[competitions['competition_id'].isin(panel_comp_ids)].copy()
    
    if relevant_comps.empty:
        logger.error("FAIL: No competition info found for panel IDs!")
        return False
        
    logger.info(f"Competitions in Panel ({len(relevant_comps)}):")
    cols = ['competition_id', 'name', 'type', 'country_name', 'competition_code'] 
    # Handle missing cols
    cols = [c for c in cols if c in relevant_comps.columns]
    
    # Check for keywords "Cup", "Qualif", "Youth"
    suspicious = relevant_comps[relevant_comps['name'].str.contains('Cup|Pokal|Quali|Youth|U19|U21', case=False, na=False)]
    
    print(relevant_comps[cols].to_markdown(index=False))
    
    if not suspicious.empty:
        logger.warning(f"POSSIBLE FAIL: Found {len(suspicious)} suspicious competition names:")
        print(suspicious[cols].to_markdown(index=False))
        return False # Soft fail?
        
    logger.info("PASS: All competitions look like leagues.")
    return True

def check_label_validity(panel, injuries):
    """
    Check B: Verify 'target=1' means injury in (t, t+30].
    """
    logger.info("--- Check B: Label Validity (Delta Check) ---")
    
    # Get positive rows
    positives = panel[panel['target'] == 1]
    sample_size = min(500, len(positives))
    sample = positives.sample(sample_size, random_state=42)
    
    # We need to find the injury that CAUSED the label. 
    # Join back to injuries on player_id
    # Condition: injury_start > week_start AND injury_start <= week_start + 30
    
    errors = 0
    deltas = []
    
    for idx, row in sample.iterrows():
        pid = row['player_id']
        t = row['week_start']
        
        # Find matching injury
        player_inj = injuries[injuries['player_id'] == pid]
        
        # Exact logic used in processing
        # match = start > t and start <= t+30
        
        matches = player_inj[
            (player_inj['injury_start'] > t) & 
            (player_inj['injury_start'] <= t + pd.Timedelta(days=30)) &
            (player_inj['days_out'] >= 7)
        ]
        
        if matches.empty:
            logger.error(f"FAIL: Row {row['player_id']}@{t} has label=1 but no matching injury found in raw data!")
            errors += 1
        else:
            # Check delta
            # If multiple, take nearest?
            first_match = matches.iloc[0]
            delta = (first_match['injury_start'] - t).days
            deltas.append(delta)
            
            if not (1 <= delta <= 30):
                 logger.error(f"FAIL: Row {pid}@{t} matched injury with delta {delta} days (Must be 1..30)")
                 errors += 1
                 
    if errors > 0:
        logger.error(f"FAIL: Found {errors} errors in {sample_size} samples.")
        return False
        
    avg_delta = np.mean(deltas)
    logger.info(f"PASS: All {sample_size} positive samples verified. Mean time-to-injury: {avg_delta:.1f} days.")
    return True

def check_ineligible_correctness(panel, injuries):
    """
    Check C: Verify 'ineligible=1' means t within [start, end].
    """
    logger.info("--- Check C: Ineligible Correctness ---")
    
    ineligibles = panel[panel['ineligible'] == 1]
    if ineligibles.empty:
        logger.warning("SKIP: No ineligible rows found.")
        return True
        
    sample_size = min(200, len(ineligibles))
    sample = ineligibles.sample(sample_size, random_state=42)
    
    errors = 0
    
    for idx, row in sample.iterrows():
        pid = row['player_id']
        t = row['week_start']
        
        player_inj = injuries[injuries['player_id'] == pid]
        
        # Match condition: t >= start & t <= end
        matches = player_inj[
            (player_inj['injury_start'] <= t) &
            (player_inj['injury_end'] >= t)
        ]
        
        if matches.empty:
            logger.error(f"FAIL: Row {pid}@{t} marked ineligible but not inside any injury window!")
            # Debug: show nearest injury?
            errors += 1
            
    if errors > 0:
        logger.error(f"FAIL: Found {errors} errors in {sample_size} samples.")
        return False
        
    logger.info(f"PASS: All {sample_size} ineligible samples verified against raw injury windows.")
    return True

def check_rolling_rolling(panel, appearances):
    """
    Check D: Recompute 'minutes_last_1w' for random sample using raw data.
    Strictly Prior: [t-7d, t)
    """
    logger.info("--- Check D: Rolling Feature Re-calculation ---")
    
    # Parse dates if needed
    if not pd.api.types.is_datetime64_any_dtype(appearances['date']): # app usually has 'date' col
        appearances['game_date'] = pd.to_datetime(appearances['date'], errors='coerce')
    else:
        appearances['game_date'] = appearances['date']
        
    sample_size = 100 # Expensive check
    sample = panel.sample(sample_size, random_state=42)
    
    errors = 0
    diffs = []
    
    # Index for speed
    apps_indexed = appearances.set_index('game_date').sort_index()
    
    for idx, row in sample.iterrows():
        pid = row['player_id']
        t = row['week_start']
        val_panel = row['minutes_last_1w']
        
        # Recompute
        # Window: [t - 7days, t) -> exclusive of t
        start_win = t - pd.Timedelta(days=7)
        end_win = t # strictly less than
        
        # Filter
        # Need player filter too. 
        # Optimize: Pre-filter apps to player is slow if done 100 times on full df?
        # Just do it.
        
        p_apps = apps_indexed[apps_indexed['player_id'] == pid]
        
        # Slice time
        # p_apps is indexed by date
        # loc[mask]
        mask = (p_apps.index >= start_win) & (p_apps.index < end_win)
        subset = p_apps[mask]
        
        calc_sum = subset['minutes_played'].sum()
        
        if abs(calc_sum - val_panel) > 1.0: # float tol
            logger.error(f"FAIL: Row {pid}@{t} Panel={val_panel}, Recomputed={calc_sum}")
            logger.error(f"   Window: {start_win} to {end_win} (exclusive)")
            logger.error(f"   Found games: {subset[['minutes_played']].to_dict()}")
            errors += 1
            diffs.append(abs(calc_sum - val_panel))
            
    if errors > 0:
        logger.error(f"FAIL: Found {errors} mismatches. Avg diff: {np.mean(diffs):.2f}")
        return False
        
    logger.info("PASS: Rolling minutes match recomputed values exactly.")
    return True

def check_missing_ids(panel, backbone_players, injuries, profiles):
    """
    Check E: Analyze the 20% missing IDs.
    """
    logger.info("--- Check E: Missing ID Breakdown ---")
    
    bb_ids = set(backbone_players['player_id'].unique())
    # Identify active ids in panel? 
    panel_ids = set(panel['player_id'].unique())
    
    inj_ids = set(injuries['player_id'].unique())
    prof_ids = set(profiles['player_id'].unique()) if profiles is not None else set()
    
    missing_in_inj = panel_ids - inj_ids
    pct_missing = len(missing_in_inj) / len(panel_ids) * 100
    
    logger.info(f"Panel Players: {len(panel_ids)}")
    logger.info(f"Missing from Injuries: {len(missing_in_inj)} ({pct_missing:.2f}%)")
    
    # Are they in profiles?
    if prof_ids:
        missing_in_prof = panel_ids - prof_ids
        in_prof_but_no_inj = (panel_ids & prof_ids) - inj_ids
        
        logger.info(f"Missing from Profiles: {len(missing_in_prof)} ({len(missing_in_prof)/len(panel_ids)*100:.2f}%)")
        logger.info(f"Present in Profiles but No Injury Record: {len(in_prof_but_no_inj)} (Likely just healthy players?)")
        
        # If they are in profiles but not injuries, they are VALID matches who act as negatives.
        # If they are missing from profiles AND injuries, they are truly "Unmapped/Missing".
        
        truly_missing = missing_in_inj & missing_in_prof
        logger.info(f"Truly Missing (No Record in Labels DB): {len(truly_missing)} ({len(truly_missing)/len(panel_ids)*100:.2f}%)")
        
        if len(truly_missing) / len(panel_ids) > 0.30:
            logger.warning("High truly missing rate (>30%).")
            return False
            
    return True

def check_workload_composition(panel, appearances, competitions):
    """
    Check F: Verify 'All Competitions' workload claim. 
    Show minutes share by Competition Type for panel players.
    """
    logger.info("--- Check F: Workload Composition Audit ---")
    
    # Filter apps to panel players and relevant date range
    panel_pids = panel['player_id'].unique()
    min_date = panel['week_start'].min()
    max_date = panel['week_start'].max()
    
    relevant = appearances[
        (appearances['player_id'].isin(panel_pids)) & 
        (appearances['date'] >= min_date) & 
        (appearances['date'] <= max_date)
    ].copy()
    
    # Merge comp info
    if 'competition_id' not in relevant.columns:
        relevant = relevant.merge(competitions[['competition_id', 'competition_code', 'type', 'sub_type', 'name']], on='competition_id', how='left')
    else:
        # If columns missing, re-merge
         relevant = relevant.drop(columns=['type', 'sub_type', 'name'], errors='ignore')
         relevant = relevant.merge(competitions[['competition_id', 'competition_code', 'type', 'sub_type', 'name']], on='competition_id', how='left')

    total_min = relevant['minutes_played'].sum()
    if total_min == 0:
        logger.error("FAIL: Zero minutes found for panel players in date range!")
        return False
        
    # Group by type
    by_type = relevant.groupby('type')['minutes_played'].sum().sort_values(ascending=False)
    
    logger.info(f"Total Minutes Analyzed: {total_min:,.0f}")
    logger.info("Minutes Share by Competition Type:")
    print(by_type.to_markdown())
    
    # Check if 'domestic_league' is < 99% (meaning we have other stuff)
    # Domestic league usually dominant, but we want >0 from others
    
    domestic_share = by_type.get('domestic_league', 0) / total_min
    logger.info(f"Domestic League Share: {domestic_share*100:.2f}%")
    
    if domestic_share > 0.999:
        logger.warning("WARNING: Domestic League share is ~100%. Are Cups/Continental matches being included?")
        # Not instant fail, but verifying user concern
        return False
        
    logger.info("PASS: Workload includes non-league minutes.")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--backbone", default="data/raw/davidcariboo:player-scores")
    parser.add_argument("--labels", default="data/raw/xfkzujqjvx97n:football-datasets")
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading Data...")
    panel = pd.read_parquet(args.panel)
    bb_dfs = load_backbone(Path(args.backbone))
    
    # Load raw labels for checking
    lbl_dfs = load_labels(Path(args.labels))
    injuries = lbl_dfs['injuries']
    profiles = lbl_dfs.get('profiles')
    
    # Clean injuries same way as pipeline for fair comparison?
    # Yes, otherwise raw messy data fails check. 
    # Use the function from processing used in pipeline.
    from src.data.processing import clean_injuries
    injuries_clean = clean_injuries(injuries)
    
    # Run Checks
    results = []
    results.append(check_top5_purity(panel, bb_dfs['games'], bb_dfs['competitions']))
    results.append(check_label_validity(panel, injuries_clean))
    results.append(check_ineligible_correctness(panel, injuries_clean))
    results.append(check_rolling_rolling(panel, bb_dfs['appearances']))
    results.append(check_missing_ids(panel, bb_dfs['players'], injuries_clean, profiles))
    results.append(check_workload_composition(panel, bb_dfs['appearances'], bb_dfs['competitions']))
    
    if all(results):
        logger.info("\nVERIFICATION SUCCESS: All checks passed.")
        exit(0)
    else:
        logger.error("\nVERIFICATION FAILED: Some checks failed.")
        exit(1)

if __name__ == "__main__":
    main()
