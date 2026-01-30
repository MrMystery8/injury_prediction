
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.data.loader import load_backbone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug():
    # Load backbone
    bb = load_backbone(Path('data/raw/davidcariboo:player-scores'))
    apps = bb['appearances']
    
    pid = 275303
    target_week = pd.Timestamp('2023-01-23')
    
    # Filter to player
    p_apps = apps[apps['player_id'] == pid].copy()
    p_apps['game_date'] = pd.to_datetime(p_apps['date'])
    
    # Check raw games around target
    start_win = target_week - pd.Timedelta(days=7)
    window_games = p_apps[(p_apps['game_date'] >= start_win) & (p_apps['game_date'] < target_week)]
    print(f"--- Raw Games in Window [{start_win}, {target_week}) ---")
    print(window_games[['game_date', 'minutes_played']])
    print(f"Sum: {window_games['minutes_played'].sum()}")
    
    # Replicate Processing Logic
    print("\n--- Processing Logic Replication ---")
    
    # Daily Agg
    daily = p_apps.groupby(['player_id', 'game_date'])[['minutes_played']].sum().reset_index()
    
    # Filter like in processing (min/max date globally)
    # Just skip, assume range is wide enough
    
    # Pivot (just this player for simplicity, but logic holds)
    daily['player_id'] = daily['player_id'].astype(int)
    
    # Simulate matrix
    # Create Matrix with index covering range
    min_date = target_week - pd.Timedelta(days=30)
    max_date = target_week + pd.Timedelta(days=5)
    
    full_idx = pd.date_range(min_date, max_date, freq='D')
    
    matrix = daily.pivot(index='game_date', columns='player_id', values='minutes_played').fillna(0)
    matrix = matrix.reindex(full_idx, fill_value=0)
    
    print("\nMatrix slice around target:")
    # Show 16th to 23rd
    subset = matrix.loc['2023-01-16':'2023-01-23']
    print(subset)
    
    # Rolling
    w = 7
    r = matrix.rolling(w).sum()
    rs = r.shift(1)
    
    print(f"\nRolling({w}).sum() at 22nd (Sun):")
    print(r.loc['2023-01-22'])
    
    print(f"\nShift(1) at 23rd (Mon):")
    print(rs.loc['2023-01-23'])
    
    # Merge to panel row
    # Panel row is (pid, 2023-01-23)
    # Join on week_start maps to matrix index
    val = rs.loc['2023-01-23', pid]
    print(f"\nFinal Value in Panel: {val}")

if __name__ == "__main__":
    debug()
