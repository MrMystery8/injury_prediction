import pandas as pd
import numpy as np

def process_injuries(injuries_df):
    """
    Parses dates and converts injury duration to days.
    Filters out invalid records (negative days_out).
    """
    # Assuming standard column names after Snake Case conversion
    # Adjust these map lookups based on actual CSV headers if they differ
    
    # Common variations map
    col_map = {
        'player': 'player_name',
        'name': 'player_name',
        'start': 'injury_start_date',
        'end': 'injury_end_date',
        'date': 'injury_start_date', # Fallback
        'duration': 'days_out'
    }
    
    # Rename if existing columns match our map keys but not values
    for k, v in col_map.items():
        if k in injuries_df.columns and v not in injuries_df.columns:
            injuries_df = injuries_df.rename(columns={k: v})

    # Ensure required columns exist
    req_cols = ['injury_start_date', 'injury_end_date']
    for c in req_cols:
        if c not in injuries_df.columns:
            # Try to be smart? or just fail gracefully
            print(f"Warning: Missing required column {c} in injuries df. Available: {injuries_df.columns}")
            return injuries_df

    # Parse dates
    injuries_df['injury_start_date'] = pd.to_datetime(injuries_df['injury_start_date'], errors='coerce')
    injuries_df['injury_end_date'] = pd.to_datetime(injuries_df['injury_end_date'], errors='coerce')

    # Compute days_out if missing
    if 'days_out' not in injuries_df.columns:
        injuries_df['days_out'] = (injuries_df['injury_end_date'] - injuries_df['injury_start_date']).dt.days
    
    # Filter valid
    injuries_df = injuries_df.dropna(subset=['injury_start_date'])
    # Only keep where days_out is positive or NaN (if start exists but end doesn't, maybe ongoing? Spec says unit is player-week, needs end date usually)
    # Spec: "Exclude rows where player is already injured (week falls inside any injury window)." -> We need end date.
    
    # If end date missing, maybe impute or drop? For now, drop rows where we can't calculate a window.
    injuries_df = injuries_df.dropna(subset=['injury_end_date'])
    injuries_df = injuries_df[injuries_df['days_out'] >= 0]
    
    return injuries_df

def create_panel_base(appearances_df, games_df):
    """
    Aggregates match-level data to player-week level.
    """
    # Merge appearances with games to get dates
    # Assuming appearances has game_id, player_id, minutes_played
    # games has game_id, date
    
    if 'game_id' not in appearances_df.columns or 'game_id' not in games_df.columns:
        print("Missing game_id linkage.")
        return None
        
    merged = appearances_df.merge(games_df[['game_id', 'date']], on='game_id', how='left')
    merged['date'] = pd.to_datetime(merged['date'])
    merged['week_start'] = merged['date'].dt.to_period('W').dt.start_time
    
    # Aggregate
    agg_funcs = {
        'minutes_played': 'sum',
        'game_id': 'count'
    }
    
    panel = merged.groupby(['player_id', 'week_start']).agg(agg_funcs).reset_index()
    panel = panel.rename(columns={'minutes_played': 'minutes_week', 'game_id': 'matches_week'})
    
    return panel
