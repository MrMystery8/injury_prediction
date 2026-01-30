
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_top5(games: pd.DataFrame, appearances: pd.DataFrame, competitions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters games and appearances to Top-5 leagues (first tier only).
    Strategy:
    1. Try to use known competition IDs if available (best).
    2. Fallback: Country + Tier == 1.
    """
    # Known Top 5 codes (Transfermarkt typical)
    # GB1 (Premier League), ES1 (LaLiga), L1 (Bundesliga), IT1 (Serie A), FR1 (Ligue 1)
    # But usually stored as competition_id.
    
    # Let's inspect competitions table columns in usage, but writing robust logic:
    
    target_competitions = []
    
    # Clean check (case insensitive)
    # The codes (GB1, ES1...) are often in competition_id itself in this dataset signature
    target_codes = {'GB1', 'ES1', 'L1', 'IT1', 'FR1'}
    
    # Check competition_id
    mask_id = competitions['competition_id'].astype(str).str.upper().isin(target_codes)
    target_competitions.extend(competitions.loc[mask_id, 'competition_id'].unique())
    
    # Check competition_code just in case
    if 'competition_code' in competitions.columns:
         mask_code = competitions['competition_code'].astype(str).str.upper().isin(target_codes)
         target_competitions.extend(competitions.loc[mask_code, 'competition_id'].unique())
        
    # Fallback/Safety: Country + Tier
    if len(target_competitions) < 5:
        logger.info("Using country/tier fallback for Top-5 filtering...")
        countries = {'england', 'spain', 'germany', 'italy', 'france'}
        
        # Check col names - prioritize 'country_name' or 'country' over 'country_id'
        country_cols = [c for c in competitions.columns if 'country' in c.lower()]
        # Sort to prefer 'name' over 'id'
        country_cols = sorted(country_cols, key=lambda x: 0 if 'name' in x else 1 if 'id' not in x else 2)
        country_col = country_cols[0] if country_cols else None
        
        type_col = next((c for c in competitions.columns if 'type' in c.lower() or 'tier' in c.lower()), None)
        
        if country_col:
            mask_country = competitions[country_col].astype(str).str.lower().isin(countries)
            mask_tier = True
            if type_col:
                # domestic_league / first_tier usually
                mask_tier = competitions[type_col].astype(str).str.lower().str.contains('first_tier|domestic_league')
            
            fallback_ids = competitions.loc[mask_country & mask_tier, 'competition_id'].unique()
            target_competitions.extend(fallback_ids)
            
    target_competitions = list(set(target_competitions))
    logger.info(f"Target Competition IDs: {target_competitions}")
    
    # Apply filter
    games_filtered = games[games['competition_id'].isin(target_competitions)].copy()
    
    # Filter appearances
    app_mask = appearances['game_id'].isin(games_filtered['game_id'])
    appearances_filtered = appearances[app_mask].copy()
    
    logger.info(f"Filtered to Top-5: {len(games_filtered)} games, {len(appearances_filtered)} appearances")
    return games_filtered, appearances_filtered

def clean_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans injury table:
    - Drops missing start dates
    - Fixes missing end/days using arithmetic
    - Deduplicates (max days_out)
    """
    df = injuries.copy()
    init_len = len(df)
    
    # 1. Drop missing start
    df = df.dropna(subset=['injury_start'])
    
    # 2. Fix missing end/days
    # If end missing, end = start + days
    mask_no_end = df['injury_end'].isna() & df['days_out'].notna()
    df.loc[mask_no_end, 'injury_end'] = df.loc[mask_no_end, 'injury_start'] + pd.to_timedelta(df.loc[mask_no_end, 'days_out'], unit='D')
    
    # If days missing, days = end - start
    mask_no_days = df['days_out'].isna() & df['injury_end'].notna()
    df.loc[mask_no_days, 'days_out'] = (df.loc[mask_no_days, 'injury_end'] - df.loc[mask_no_days, 'injury_start']).dt.days
    
    # 3. Sanity: end >= start
    mask_invalid = df['injury_end'] < df['injury_start']
    n_invalid = mask_invalid.sum()
    if n_invalid > 0:
        logger.warning(f"Dropping {n_invalid} injuries with end < start")
        df = df[~mask_invalid]
        
    # 4. Dedup: strictly same player + start_date -> keep max days
    df = df.sort_values('days_out', ascending=False).drop_duplicates(subset=['player_id', 'injury_start'], keep='first')
    
    logger.info(f"Injuries cleaned: {init_len} -> {len(df)}")
    return df

def build_weekly_panel(games: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a skeleton panel of (player_id, week_start).
    Logic: Global In-Season Weeks per Competition.
    """
    # 1. Determine weeks per competition season
    # week_start = Monday
    games['week_start'] = games['game_date'] - pd.to_timedelta(games['game_date'].dt.weekday, unit='D')
    
    panel_rows = []
    
    # Iterate by competition-season to define strict active windows
    if 'season' not in games.columns:
        # Fallback if season col missing, infer from date? 
        # Usually 'season' is typically 2020, 2021 etc. in player-scores
        pass

    # Group players by [competition_id, season] they played in
    # This ensures we don't put a Premier League player in a timestamp where only Bundesliga was active if seasons misalign slightly?
    # Actually simpler: Group appearances by (player_id) -> find their primary competitions? 
    # Or just simpler:
    # 1. Get all unique weeks where ANY game happened in specific competition.
    # 2. Get all players who played in that competition-season.
    # 3. Cross join.
    
    # Let's do: For each (competition_id, season), get date range and player set.
    
    # Merge appearances with games to get season/comp info for players
    app_meta = appearances[['player_id', 'game_id']].merge(
        games[['game_id', 'competition_id', 'season', 'week_start']], 
        on='game_id'
    )
    
    # Unique active (comp, season) tuples
    comp_seasons = app_meta[['competition_id', 'season']].drop_duplicates()
    
    for _, row in comp_seasons.iterrows():
        cid, sea = row['competition_id'], row['season']
        
        # Get global weeks for this comp-season
        subset_games = games[(games['competition_id'] == cid) & (games['season'] == sea)]
        if subset_games.empty: continue
        
        min_date = subset_games['week_start'].min()
        max_date = subset_games['week_start'].max()
        
        # Full range of weeks
        if pd.isna(min_date) or pd.isna(max_date): continue
        all_weeks = pd.date_range(min_date, max_date, freq='W-MON')
        
        # Get players active in this comp-season
        players_in_cs = app_meta[(app_meta['competition_id'] == cid) & (app_meta['season'] == sea)]['player_id'].unique()
        
        # Cartesian product
        # Create temp DF
        weeks_df = pd.DataFrame({'week_start': all_weeks})
        weeks_df['key'] = 1
        players_df = pd.DataFrame({'player_id': players_in_cs})
        players_df['key'] = 1
        
        chunk = pd.merge(players_df, weeks_df, on='key').drop('key', axis=1)
        chunk['competition_id'] = cid
        chunk['season'] = sea
        
        panel_rows.append(chunk)
        
    if not panel_rows:
        return pd.DataFrame()
        
    panel = pd.concat(panel_rows, ignore_index=True)
    
    # Dedup: A player might play in 2 comps in same week (e.g. UCL + PL) -> prevent row duplication
    # We want 1 row per player-week.
    panel = panel.drop_duplicates(subset=['player_id', 'week_start'])
    
    return panel

def add_workload_features(panel: pd.DataFrame, appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strict PRIOR rolling window features.
    Warning: Avoid leakage. Only use matches where game_date < week_start.
    """
    # Pre-merge game_date to appearances
    if 'game_date' not in appearances.columns:
        appearances = appearances.merge(games[['game_id', 'game_date']], on='game_id', how='left')
        
    # To do this efficiently:
    # 1. Sort appearances by player, date
    # 2. Use asof merge or iterate? 
    # Pandas strictly prior rolling is tricky with simple .rolling().
    # Better approach:
    #   Aggregate appearances to daily level -> resample daily -> rolling -> shift(1) -> reindex to weekly panel?
    
    # Let's try the resample approach.
    logger.info("Computing rolling workload features (Daily Resample -> Rolling -> Shift -> Weekly Reindex)")
    
    # Daily aggregation
    daily = appearances.groupby(['player_id', 'game_date'])[['minutes_played']].sum().reset_index()
    daily = daily.set_index('game_date')
    
    features_list = []
    
    # Iterate players (slow but safe for 10-20k players? might be too slow for 200k panel if Python loop)
    # Vectorized way:
    # 1. set index to date
    # 2. groupby player
    # 3. resample 'D'
    # 4. rolling totals
    # 5. shift(1) (to exclude today)
    
    # Careful: 'minutes_played' is only on match days. We need 0s on non-match days.
    # GroupBy Resample handles this.
    
    # Optimization: Filter appearances to only those relevant to panel players/dates
    min_panel_date = panel['week_start'].min() - pd.Timedelta(days=60) # Buffer for 8w
    max_panel_date = panel['week_start'].max()
    
    relevant_apps = daily[(daily.index >= min_panel_date) & (daily.index <= max_panel_date)]
    
    # We need to ensure every player has a continuous daily time series?
    # Actually rolling on a sparse series with time-window is supported in pandas.
    # df.rolling('7D').sum()
    
    player_groups = relevant_apps.groupby('player_id')['minutes_played']
    
    # Compute rolling windows
    # closed='left' means for window [t-w, t), excludes t. This is what we want (strictly prior).
    # But pandas rolling window is usually [t-w, t].
    # If we use closed='left', valid.
    
    # Let's define windows: 7D, 14D, 28D, 56D
    windows = {
        'minutes_last_1w': 7,
        'minutes_last_2w': 14,
        'minutes_last_4w': 28,
        'minutes_last_8w': 56
    }
    
    # Create a dense daily frame? No, use the panel date points + asof merge.
    
    # Rolling sums on the events, then lookup?
    # No, rolling sum needs zeros.
    
    # Let's stick to a simpler safe method:
    # For each panel row (player, week_start), sum minutes in range [week_start - window, week_start).
    # Since panel is large (millions?), apply() is slow.
    # SQl style:
    # Join panel to appearances on player_id AND app.date < panel.date AND app.date >= panel.date - window.
    # This is a range join.
    
    # Conditional Join for 1 window?
    # DuckDB is great for this. Pandas is hard.
    
    # Alternative:
    # 1. Pivot appearances to wide (player x date). Fill 0.
    # 2. Rolling sum.
    # 3. Stack back to long.
    # 4. Merge to panel.
    
    # Let's do the "daily resample + rolling + asof join" method. 
    # It handles the zeros correctly.
    
    # Pivot
    # Only pivot relevant players + date range (buffer for 8w windows)
    needed_players = panel['player_id'].unique()
    daily_filtered = daily[daily['player_id'].isin(needed_players)]
    daily_filtered = daily_filtered[(daily_filtered.index >= min_panel_date) & (daily_filtered.index <= max_panel_date)]
    daily_filtered = daily_filtered.reset_index()
    
    # Create daily grid?
    # Might be too big (5000 players * 365*5 days = 9M rows). Acceptable.
    
    # GroupBy Rolling is safer memory-wise
    daily_sorted = daily_filtered.sort_values(['player_id', 'game_date'])
    daily_indexed = daily_sorted.set_index('game_date')
    
    # We need to fill missing days with 0 for accurate rolling time-window sums?
    # pandas rolling('7D') handles irregular time series correctly (sums values in window).
    # BUT it doesn't "see" zeros for missing days.
    # Example: Match on Day 1 (90m). Day 2..7 no match.
    # On Day 8: rolling('7D') sees only Day 1? Sum = 90. Correct.
    # On Day 10: rolling('7D') sees range [Day 3, Day 10]. Match Day 1 is out. Sum = 0.
    # If Day 1 is the only row, Day 10 doesn't exist in index, so we won't get a row for Day 10.
    
    # So we need to compute the rolling values *at the panel dates*.
    # pd.merge_asof is perfect for "latest value available".
    # BUT we need the rolling sum calculated correctly including the decay to zero.
    
    # Correct algo:
    # 1. For each player, reindex daily series to fill all days (or at least panel weeks).
    # 2. Compute rolling sums.
    # 3. Join to panel on week_start.
    
    # Let's pivot to (date, player) matrix -> resample D -> fillna 0 -> rolling -> stack -> unstack?
    # Memory: 2000 days * 5000 players * 8 bytes = 80MB. float64. Very Cheap! 
    # This is the best way.
    
    matrix = daily_filtered.pivot(index='game_date', columns='player_id', values='minutes_played').fillna(0)
    
    # Reindex to full date range to ensure continuity
    # Force freq='D' to allow integer rolling
    full_idx = pd.date_range(matrix.index.min(), matrix.index.max(), freq='D')
    matrix = matrix.reindex(full_idx, fill_value=0)
    
    # Compute rolling features
    roll_feats = {}
    for name, window in windows.items():
        # closed='left' excludes today (strict prior)
        # rolling() in newer pandas supports closed='left'. 
        # If not, shift(1) after rolling closed='right'.
        
        # Using integer window on 'D' frequency index is exact
        r = matrix.rolling(window).sum().shift(1) # shift 1 day forward
        
        # Keep only rows matching panel dates (Mondays)
        # But panel might have arbitrary dates? No, we enforced W-MON.
        # We can just filter matrix to Mondays?
        
        # Flatten
        # stack() creates (Date, Player) index
        # Only keep Mondays (panel is W-MON) to avoid stacking the entire daily matrix.
        r_monday = r[r.index.weekday == 0]
        r_flat = r_monday.stack().reset_index()
        r_flat.columns = ['week_start', 'player_id', name]
        roll_feats[name] = r_flat
        
    # Merge into panel
    # We must merge carefully.
    result = panel.copy()
    for name, df_feat in roll_feats.items():
        result = result.merge(df_feat, on=['player_id', 'week_start'], how='left')
        result[name] = result[name].fillna(0)
        
    # ACWR
    eps = 1
    result['acwr'] = result['minutes_last_1w'] / (result['minutes_last_4w'] / 4 + eps) # smoothed
    
    return result

def add_labels_and_exclusions(panel: pd.DataFrame, injuries: pd.DataFrame, profiles: pd.DataFrame = None) -> pd.DataFrame:
    """
    Labels: 1 if injury starts in (t, t+30].
    Exclusions: ineligible=1 if t in [start, end].
    Target: Only injuries with days_out >= 7.
    Label Known: 1 if player exists in label universe (profiles or injuries).
    """
    # 0. Establish Label Universe
    label_universe_ids = set(injuries['player_id'].unique())
    if profiles is not None:
        label_universe_ids.update(profiles['player_id'].unique())
        
    panel['label_known'] = panel['player_id'].isin(label_universe_ids).astype(int)
    
    # 1. Identify relevant injuries (days_out >= 7)
    serious_inj = injuries[injuries['days_out'] >= 7].copy()
    
    # 2. Mark Ineligible (Already Injured)
    # Range Join: panel.week_start BETWEEN inj.start AND inj.end
    # We can use a numba/apply check or interval index?
    # IntervalIndex is robust.
    
    # Create exclusion mask
    # For each player, build IntervalIndex of injury periods
    
    # Optimize: Iterate players with injuries
    injured_players = serious_inj['player_id'].unique()
    
    # Pre-sort for speed
    panel = panel.sort_values(['player_id', 'week_start'])
    
    # Vectorized approach hard for range overlaps.
    # GroupBy Apply?
    
    logger.info("Computing labels and exclusions...")
    
    # Build huge list of injury intervals?
    # Or merge_asof logic?
    
    # Let's iterate specialized by player for correctness (safer than complex pandas hacks)
    # Only for players who HAVE injuries.
    
    # This might be slow if 10k players.
    # But vectorized interval search is:
    # for each injury, panel.loc[(player==p) & (date >= start) & (date <= end), 'ineligible'] = 1
    
    # This loop is (NumInjuries) long. ~20k injuries? manageable.
    
    # Optimization: Dictionary of Intervals per player
    # processing...
    
    # Actually, let's look at "Target Generation" first.
    # Target = 1 if injury_start in (week_start, week_start + 30]
    # We can join panel to injuries on player_id.
    # Filter: inj_start > week_start AND inj_start <= week_start + 30.
    
    # This join explodes? (Nulls for non-injured).
    # Left join panel -> injuries (on player).
    # Then filter rows where condition met -> mark label=1 in original.
    
    # Let's do that.
    merged = panel[['player_id', 'week_start']].merge(serious_inj, on='player_id', how='inner')
    
    # TARGET LOGIC
    # inj_start in (t, t+30]
    t = merged['week_start']
    start = merged['injury_start']
    mask_target = (start > t) & (start <= t + pd.Timedelta(days=30))
    positives = merged[mask_target]
    
    # Set target=1 for these (p, t)
    # There could be dupes (2 injuries in 30 days) -> just set to 1
    pos_keys = positives[['player_id', 'week_start']].drop_duplicates()
    pos_keys['target'] = 1
    
    panel = panel.merge(pos_keys, on=['player_id', 'week_start'], how='left')
    panel['target'] = panel['target'].fillna(0).astype(int)
    
    # INELIGIBLE LOGIC
    # t in [start, end]
    # Use all injuries (even minor ones? Plan says >=7 for label, but exclusion should be all? 
    # Plan says "exclude columns with days_out >= 7" usually, but technically minor injury doesn't stop play.
    # Adhere to plan: "If week_start inside [injury_start, injury_end] ... ineligible"
    # Usually implies keeping consistency. Let's use same serious_inj set for now, or all if available?
    # Let's use serious_inj for exclusion to maintain definition of "Healthy".
    
    # t >= start AND t <= end
    end = merged['injury_end']
    mask_inelig = (t >= start) & (t <= end)
    ineligibles = merged[mask_inelig]
    
    inel_keys = ineligibles[['player_id', 'week_start']].drop_duplicates()
    inel_keys['ineligible'] = 1
    
    panel = panel.merge(inel_keys, on=['player_id', 'week_start'], how='left')
    panel['ineligible'] = panel['ineligible'].fillna(0).astype(int)
    
    # Clean up dupe columns if merge created any suffix? no, we restricted cols.
    
    return panel

def add_structural_flags(panel: pd.DataFrame, injuries: pd.DataFrame, appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Structural Fixes:
    1. not_censored: 1 if week_start + 30d <= max_known_injury_date.
    2. active_player (Option A): 1 if week_start in [first_app - 14d, last_app + 14d] for that season.
    """
    logger.info("Adding Structural Flags (Censoring & Activity)...")
    
    # 1. Right Censoring
    max_label_date = injuries['injury_start'].max()
    logger.info(f"Max Injury Date in Labels: {max_label_date}")
    
    # week_start + 30 days must be observed (<= max_date)
    cutoff_check = panel['week_start'] + pd.Timedelta(days=30)
    panel['not_censored'] = (cutoff_check <= max_label_date).astype(int)
    
    # 2. Active Player Policy (Option A)
    # Strategy: Calc (Player, Season) -> [MinDate, MaxDate].
    # Then Join to Panel.
    
    logger.info("Mapping Player-Season Active Windows (active_player option A)...")
    
    # Join apps to games to get season
    # Minimal columns
    apps_mini = appearances[['game_id', 'player_id', 'date']].copy()
    if not pd.api.types.is_datetime64_any_dtype(apps_mini['date']):
        apps_mini['date'] = pd.to_datetime(apps_mini['date'], errors='coerce')
        
    games_mini = games[['game_id', 'season']].copy()
    
    # Merge
    merged = apps_mini.merge(games_mini, on='game_id', how='inner')
    
    # Group by Player, Season -> Min/Max
    ranges = merged.groupby(['player_id', 'season'])['date'].agg(['min', 'max']).reset_index()
    ranges.columns = ['player_id', 'season', 'first_app', 'last_app']
    
    # Buffer +/- 14 days
    ranges['start_window'] = ranges['first_app'] - pd.Timedelta(days=14)
    ranges['end_window'] = ranges['last_app'] + pd.Timedelta(days=14)
    
    # Join to panel
    # Panel has player_id, season
    panel = panel.merge(ranges[['player_id', 'season', 'start_window', 'end_window']], on=['player_id', 'season'], how='left')
    
    # Calc active_player
    # If no apps in season (NaN), active=0
    cond = (panel['week_start'] >= panel['start_window']) & (panel['week_start'] <= panel['end_window'])
    panel['active_player'] = cond.astype(int)
    
    # Clean up aux cols
    panel = panel.drop(columns=['start_window', 'end_window'])
    
    return panel

def add_congestion_features(panel: pd.DataFrame, appearances: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Computes congestion/intensity features (User Step 1):
    - matches_last_7d, matches_last_21d (Counts)
    - minutes_last_10d (Short term load)
    - max_minutes_14d (Peak intensity)
    - matches_last_3d (Back-to-back proxy)
    - away_matches_last_28d, away_minutes_last_28d (travel proxy; requires games)
    - yellow_cards_last_28d, red_cards_last_56d (foul/discipline proxy)
    """
    logger.info("Computing Congestion Features (Matches, Max Minutes)...")
    
    # 1. Pivot Daily
    # Only relevant players
    needed_players = panel['player_id'].unique()
    min_panel_date = panel['week_start'].min() - pd.Timedelta(days=60)
    max_panel_date = panel['week_start'].max()
    
    # Check for date col
    date_col = 'game_date' if 'game_date' in appearances.columns else 'date'

    base_cols = ['player_id', date_col, 'minutes_played']
    for extra in ['game_id', 'player_club_id', 'yellow_cards', 'red_cards']:
        if extra in appearances.columns:
            base_cols.append(extra)

    app_mini = appearances[appearances['player_id'].isin(needed_players)][base_cols].copy()
    
    if date_col != 'game_date':
        app_mini = app_mini.rename(columns={date_col: 'game_date'})

    # Restrict to panel date range (+buffer)
    app_mini = app_mini[(app_mini['game_date'] >= min_panel_date) & (app_mini['game_date'] <= max_panel_date)]
    
    # Add 'match_played' boolean
    app_mini['match_played'] = 1

    # Travel proxy: away/home (requires games join)
    if games is not None and not games.empty and {'game_id', 'home_club_id', 'away_club_id'}.issubset(games.columns) and 'game_id' in app_mini.columns and 'player_club_id' in app_mini.columns:
        app_mini = app_mini.merge(games[['game_id', 'home_club_id', 'away_club_id']], on='game_id', how='left')
        app_mini['is_away'] = (app_mini['player_club_id'] == app_mini['away_club_id']).astype(int)
        app_mini['away_minutes'] = app_mini['minutes_played'] * app_mini['is_away']
    
    # Pivot (Date x Player)
    # Fill 0 for mins, 0 for match_played
    pivoted_mins = app_mini.pivot_table(index='game_date', columns='player_id', values='minutes_played', aggfunc='sum').fillna(0)
    pivoted_cnts = app_mini.pivot_table(index='game_date', columns='player_id', values='match_played', aggfunc='sum').fillna(0)

    pivoted_yellow = None
    pivoted_red = None
    if 'yellow_cards' in app_mini.columns:
        pivoted_yellow = app_mini.pivot_table(index='game_date', columns='player_id', values='yellow_cards', aggfunc='sum').fillna(0)
    if 'red_cards' in app_mini.columns:
        pivoted_red = app_mini.pivot_table(index='game_date', columns='player_id', values='red_cards', aggfunc='sum').fillna(0)

    pivoted_away_cnts = None
    pivoted_away_mins = None
    if 'is_away' in app_mini.columns:
        pivoted_away_cnts = app_mini.pivot_table(index='game_date', columns='player_id', values='is_away', aggfunc='sum').fillna(0)
        pivoted_away_mins = app_mini.pivot_table(index='game_date', columns='player_id', values='away_minutes', aggfunc='sum').fillna(0)
    
    # Reindex to full daily range
    full_idx = pd.date_range(pivoted_mins.index.min(), pivoted_mins.index.max(), freq='D')
    # Resample ensures we handle dates if pivot missed range bounds
    pivoted_mins = pivoted_mins.reindex(full_idx, fill_value=0)
    pivoted_cnts = pivoted_cnts.reindex(full_idx, fill_value=0)
    if pivoted_yellow is not None:
        pivoted_yellow = pivoted_yellow.reindex(full_idx, fill_value=0)
    if pivoted_red is not None:
        pivoted_red = pivoted_red.reindex(full_idx, fill_value=0)
    if pivoted_away_cnts is not None:
        pivoted_away_cnts = pivoted_away_cnts.reindex(full_idx, fill_value=0)
    if pivoted_away_mins is not None:
        pivoted_away_mins = pivoted_away_mins.reindex(full_idx, fill_value=0)
    
    # Features Config
    # Name -> (Data, Window, Agg)
    feats_cfg = {
        'matches_last_3d': (pivoted_cnts, 3, 'sum'),   # B2B proxy
        'matches_last_7d': (pivoted_cnts, 7, 'sum'),
        'matches_last_14d': (pivoted_cnts, 14, 'sum'),
        'matches_last_28d': (pivoted_cnts, 28, 'sum'), # Matches last 4w
        'minutes_last_10d': (pivoted_mins, 10, 'sum'),
        'max_minutes_14d': (pivoted_mins, 14, 'max'),
    }

    if pivoted_away_cnts is not None and pivoted_away_mins is not None:
        feats_cfg['away_matches_last_28d'] = (pivoted_away_cnts, 28, 'sum')
        feats_cfg['away_minutes_last_28d'] = (pivoted_away_mins, 28, 'sum')

    if pivoted_yellow is not None:
        feats_cfg['yellow_cards_last_28d'] = (pivoted_yellow, 28, 'sum')
    if pivoted_red is not None:
        feats_cfg['red_cards_last_56d'] = (pivoted_red, 56, 'sum')
    
    result = panel.copy()
    
    for name, (matrix, window, agg) in feats_cfg.items():
        # Rolling
        r = getattr(matrix.rolling(window), agg)().shift(1) # Strict prior
        
        # Stack & Merge
        # Only keep Mondays (panel is W-MON) to avoid stacking the entire daily matrix.
        r_monday = r[r.index.weekday == 0]
        flat = r_monday.stack().reset_index()
        flat.columns = ['week_start', 'player_id', name]
        
        # Merge
        result = result.merge(flat, on=['player_id', 'week_start'], how='left')
        result[name] = result[name].fillna(0)

    # Derived: away share (avoid div-by-zero)
    if 'away_matches_last_28d' in result.columns and 'matches_last_28d' in result.columns:
        eps = 1e-6
        result['away_match_share_28d'] = result['away_matches_last_28d'] / (result['matches_last_28d'] + eps)
        
    return result

def add_context_features(panel: pd.DataFrame, transfers: pd.DataFrame, market_values: pd.DataFrame) -> pd.DataFrame:
    """
    Adds As-Of Context features:
    - days_since_transfer
    - market_value_log
    - market_value_trend (current / 6m_ago)
    """
    logger.info("Adding Context Features (Transfers & Market Value)...")
    
    # Sort panel for merge_asof (Pandas 3.x requires global sort on the `on` key first)
    result = panel.sort_values(['week_start', 'player_id']).copy()
    
    # 1. Transfers
    if transfers is not None and not transfers.empty:
        transfers_sorted = transfers.dropna(subset=['transfer_date']).copy()
        # Remove placeholder dates (seen as 1900-07-01 in raw)
        transfers_sorted = transfers_sorted[transfers_sorted['transfer_date'] >= pd.Timestamp('1950-01-01')]
        transfers_sorted = transfers_sorted.sort_values(['transfer_date', 'player_id'])
        
        # Merge Asof to get *latest* transfer date strictly before or on week_start
        # direction='backward' matches <= week_start. strict prior? 
        # If transfer happened today, does it affect risk? Yes.
        
        # Rename date for merge
        t_merge = transfers_sorted[['player_id', 'transfer_date']].copy()
        t_merge = t_merge.rename(columns={'transfer_date': 'latest_transfer_date'})
        
        result = pd.merge_asof(
            result,
            t_merge.sort_values(['latest_transfer_date', 'player_id']),
            left_on='week_start',
            right_on='latest_transfer_date',
            by='player_id',
            direction='backward',
        )
        
        # Calc days since
        if 'latest_transfer_date' in result.columns:
            result['days_since_transfer'] = (result['week_start'] - result['latest_transfer_date']).dt.days
            result = result.drop(columns=['latest_transfer_date'])
        else:
            result['days_since_transfer'] = 9999
            
    else:
        result['days_since_transfer'] = 9999
        
    # Fill NA (no transfers found) -> Large number?
    result['days_since_transfer'] = result['days_since_transfer'].fillna(3650) # 10 years cap
    
    # 2. Market Values
    if market_values is not None and not market_values.empty:
        mv_sorted = market_values.dropna(subset=['date']).copy().sort_values(['date', 'player_id'])
        mv_merge = mv_sorted[['player_id', 'date', 'value']].rename(columns={'date': 'mv_date'})
        
        # Current Value (As Of)
        result = pd.merge_asof(
            result,
            mv_merge.sort_values(['mv_date', 'player_id']),
            left_on='week_start',
            right_on='mv_date',
            by='player_id',
            direction='backward'
        )
        
        result['market_value'] = result['value'].fillna(0)
        result = result.drop(columns=['mv_date', 'value'])
        
        # Value 6 months ago (Trend)
        # Create a lagged date in panel
        result['date_6m_ago'] = result['week_start'] - pd.Timedelta(days=180)
        
        # Merge again using date_6m_ago
        # merge_asof requires left side to be sorted.
        # It is sorted by week_start, so date_6m_ago is also sorted.
        
        result = pd.merge_asof(
            result,
            mv_merge.sort_values(['mv_date', 'player_id']),
            left_on='date_6m_ago',
            right_on='mv_date',
            by='player_id',
            direction='backward'
        )
        
        result['market_value_6m'] = result['value'].fillna(0)
        result = result.drop(columns=['date_6m_ago', 'mv_date', 'value'])
        
        # Features
        result['log_market_value'] = np.log1p(result['market_value'])
        
        # Trend: (Current - Old) / (Old + eps) is percent change?
        # Or simple ratio.
        # User asked for "6-12 month slope".
        eps = 10000
        result['market_value_trend'] = (result['market_value'] - result['market_value_6m']) / (result['market_value_6m'] + eps)
    
    else:
        result['log_market_value'] = 0
        result['market_value_trend'] = 0
        
    return result

def add_injury_history_features(
    panel: pd.DataFrame,
    injuries: pd.DataFrame,
    windows_days: Tuple[int, ...] = (30, 180, 365),
    min_days_out: int = 7,
) -> pd.DataFrame:
    """
    Adds strictly-prior injury history features using the injury table.

    Definitions (strictly prior):
    - Counts/sums use injuries with `injury_start` in [t-window, t)
    - Recency uses last `injury_start` / `injury_end` with date < t
    """
    logger.info("Adding Injury History Features...")

    if injuries is None or injuries.empty:
        result = panel.copy()
        for w in windows_days:
            result[f'injuries_last_{w}d'] = 0
            result[f'days_out_last_{w}d'] = 0.0
        result['days_since_last_injury_start'] = 9999
        result['days_since_last_injury_end'] = 9999
        result['last_injury_days_out'] = 0.0
        return result

    required = {'player_id', 'injury_start', 'injury_end', 'days_out'}
    missing = required - set(injuries.columns)
    if missing:
        raise ValueError(f"Injuries table missing required columns: {sorted(missing)}")

    needed_players = panel['player_id'].unique()

    inj = injuries[injuries['player_id'].isin(needed_players)].copy()
    inj = inj.dropna(subset=['injury_start'])
    inj['days_out'] = pd.to_numeric(inj['days_out'], errors='coerce')
    inj = inj[inj['days_out'].fillna(0) >= min_days_out]

    # If no injuries, return zeros
    if inj.empty:
        result = panel.copy()
        for w in windows_days:
            result[f'injuries_last_{w}d'] = 0
            result[f'days_out_last_{w}d'] = 0.0
        result['days_since_last_injury_start'] = 9999
        result['days_since_last_injury_end'] = 9999
        result['last_injury_days_out'] = 0.0
        return result

    inj = inj.sort_values(['player_id', 'injury_start'])

    # Pre-build per-player arrays for fast numpy searchsorted in the main loop.
    starts_by_player: Dict[int, np.ndarray] = {}
    ends_by_player: Dict[int, np.ndarray] = {}
    days_by_player: Dict[int, np.ndarray] = {}
    prefix_days_by_player: Dict[int, np.ndarray] = {}

    for pid, g in inj.groupby('player_id', sort=False):
        starts = g['injury_start'].to_numpy(dtype='datetime64[ns]')
        days = g['days_out'].fillna(0).to_numpy(dtype=float)
        prefix_days = np.concatenate(([0.0], np.cumsum(days)))

        ends = g['injury_end'].to_numpy(dtype='datetime64[ns]')
        ends = ends[~pd.isna(ends)]
        ends.sort()

        starts_by_player[int(pid)] = starts
        days_by_player[int(pid)] = days
        prefix_days_by_player[int(pid)] = prefix_days
        ends_by_player[int(pid)] = ends

    ordered = panel.sort_values(['player_id', 'week_start']).reset_index()
    orig_index = ordered['index'].to_numpy()
    pids = ordered['player_id'].to_numpy()
    weeks = ordered['week_start'].to_numpy(dtype='datetime64[ns]')

    n = len(ordered)
    # Use int32 to avoid overflow if someone sets huge windows.
    counts = {w: np.zeros(n, dtype=np.int32) for w in windows_days}
    sums = {w: np.zeros(n, dtype=np.float32) for w in windows_days}
    days_since_start = np.full(n, 9999, dtype=np.int32)
    days_since_end = np.full(n, 9999, dtype=np.int32)
    last_injury_days_out = np.zeros(n, dtype=np.float32)

    unique_pids, start_idx = np.unique(pids, return_index=True)
    end_idx = np.r_[start_idx[1:], n]

    for i, pid in enumerate(unique_pids):
        s = slice(start_idx[i], end_idx[i])
        weeks_pid = weeks[s]

        starts = starts_by_player.get(int(pid))
        if starts is None or len(starts) == 0:
            continue

        days = days_by_player[int(pid)]
        prefix_days = prefix_days_by_player[int(pid)]

        # Counts & sums in windows
        for w in windows_days:
            left = weeks_pid - np.timedelta64(w, 'D')
            right = weeks_pid
            i_right = np.searchsorted(starts, right, side='left')
            i_left = np.searchsorted(starts, left, side='left')
            counts[w][s] = i_right - i_left
            sums[w][s] = (prefix_days[i_right] - prefix_days[i_left]).astype(np.float32)

        # Recency from injury_start
        i_last_start = np.searchsorted(starts, weeks_pid, side='left') - 1
        valid_start = i_last_start >= 0
        if valid_start.any():
            last_start = starts[i_last_start[valid_start]]
            days_since_start[s][valid_start] = ((weeks_pid[valid_start] - last_start) / np.timedelta64(1, 'D')).astype(np.int32)
            last_injury_days_out[s][valid_start] = days[i_last_start[valid_start]].astype(np.float32)

        # Recency from injury_end
        ends = ends_by_player.get(int(pid))
        if ends is not None and len(ends) > 0:
            i_last_end = np.searchsorted(ends, weeks_pid, side='left') - 1
            valid_end = i_last_end >= 0
            if valid_end.any():
                last_end = ends[i_last_end[valid_end]]
                days_since_end[s][valid_end] = ((weeks_pid[valid_end] - last_end) / np.timedelta64(1, 'D')).astype(np.int32)

    feat = pd.DataFrame(index=orig_index)
    for w in windows_days:
        feat[f'injuries_last_{w}d'] = counts[w]
        feat[f'days_out_last_{w}d'] = sums[w]
    feat['days_since_last_injury_start'] = days_since_start
    feat['days_since_last_injury_end'] = days_since_end
    feat['last_injury_days_out'] = last_injury_days_out

    return panel.join(feat, how='left')
