import pandas as pd
import numpy as np
from pathlib import Path

def get_basic_stats(df):
    """
    Returns shape, column list, dtypes, duplicates, memory usage.
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    }
    return stats

def get_missingness(df):
    """
    Returns missing % per column.
    """
    missing = df.isnull().mean() * 100
    missing = missing.sort_values(ascending=False)
    
    # Flag columns with > 50% missing
    flagged = missing[missing > 50].index.tolist()
    
    return {
        'missing_pct': missing.to_dict(),
        'high_missing_cols': flagged
    }

def get_numeric_summary(df):
    """
    Summary for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}
    
    summary = numeric_df.describe().transpose().to_dict('index')
    
    # Top 5 largest values for important numeric columns
    outliers = {}
    for col in numeric_df.columns:
        outliers[col] = numeric_df[col].nlargest(5).tolist()
        
    return {
        'summary': summary,
        'top_5': outliers
    }

def get_categorical_summary(df):
    """
    Summary for categorical/object columns.
    """
    cat_df = df.select_dtypes(include=['object', 'category'])
    if cat_df.empty:
        return {}
    
    summary = {}
    for col in cat_df.columns:
        unique_count = cat_df[col].nunique()
        top_10 = cat_df[col].value_counts().head(10).to_dict()
        
        # Flag high cardinality
        is_id_candidate = unique_count / len(df) > 0.9 if len(df) > 0 else False
        
        summary[col] = {
            'unique_count': unique_count,
            'top_10': top_10,
            'high_cardinality': is_id_candidate
        }
        
    return summary

def detect_and_parse_dates(df):
    """
    Detect candidate date columns and attempt parsing.
    """
    date_keywords = ['date', 'time', 'from', 'until', 'start', 'end', 'birth']
    candidate_cols = [col for col in df.columns if any(kw in col.lower() for kw in date_keywords)]
    
    results = {}
    for col in candidate_cols:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            success_rate = (parsed.notnull().sum() / len(df)) * 100 if len(df) > 0 else 0
            
            if success_rate > 10: # Only report if it seems like it worked somewhat
                results[col] = {
                    'success_rate': success_rate,
                    'min': parsed.min(),
                    'max': parsed.max(),
                    'invalid_count': parsed.isnull().sum()
                }
        except:
            continue
            
    return results

def detect_id_candidates(df):
    """
    Detect columns likely to be IDs.
    """
    id_keywords = ['_id', 'player_id', 'game_id', 'club_id', 'team_id', 'competition_id']
    candidates = [col for col in df.columns if any(kw in col.lower() for kw in id_keywords) or col.lower() == 'id']
    
    results = {}
    for col in candidates:
        nunique = df[col].nunique()
        nrows = len(df)
        uniqueness_ratio = nunique / nrows if nrows > 0 else 0
        null_rate = df[col].isnull().mean() * 100
        
        results[col] = {
            'null_rate': null_rate,
            'uniqueness_ratio': uniqueness_ratio,
            'is_likely_pk': uniqueness_ratio == 1.0 and null_rate == 0
        }
        
    return results

def run_football_checks(df, table_name):
    """
    Specific checks for injury types, appearances, etc.
    """
    checks = {}
    
    # 8.1 Injuries tables
    start_cols = [c for c in df.columns if 'start' in c.lower() or 'from' in c.lower()]
    end_cols = [c for c in df.columns if 'end' in c.lower() or 'until' in c.lower()]
    
    if start_cols and end_cols:
        start_col = start_cols[0]
        end_col = end_cols[0]
        try:
            start_dt = pd.to_datetime(df[start_col], errors='coerce')
            end_dt = pd.to_datetime(df[end_col], errors='coerce')
            days_out = (end_dt - start_dt).dt.days
            
            checks['injuries'] = {
                'negative_days': int((days_out < 0).sum()),
                'zero_days': int((days_out == 0).sum()),
                'time_loss_7plus': int((days_out >= 7).sum()),
                'median_days': float(days_out.median()) if not days_out.dropna().empty else 0
            }
        except:
            pass
            
    # 8.2 Appearances / minutes
    min_cols = [c for c in df.columns if 'minute' in c.lower()]
    if min_cols:
        min_col = min_cols[0]
        try:
            minutes = pd.to_numeric(df[min_col], errors='coerce')
            checks['minutes'] = {
                'zero_minutes_pct': (minutes == 0).mean() * 100,
                'min': float(minutes.min()),
                'max': float(minutes.max()),
                'median': float(minutes.median())
            }
        except:
            pass

    # 8.3 Players table (height/weight)
    height_cols = [c for c in df.columns if 'height' in c.lower()]
    weight_cols = [c for c in df.columns if 'weight' in c.lower()]
    
    if height_cols:
        h_col = height_cols[0]
        h_vals = pd.to_numeric(df[h_col], errors='coerce')
        checks['height'] = {
            'outliers': int(((h_vals < 140) | (h_vals > 220)).sum()),
            'missing_pct': h_vals.isnull().mean() * 100
        }
        
    if weight_cols:
        w_col = weight_cols[0]
        w_vals = pd.to_numeric(df[w_col], errors='coerce')
        checks['weight'] = {
            'outliers': int(((w_vals < 45) | (w_vals > 120)).sum()),
            'missing_pct': w_vals.isnull().mean() * 100
        }
    
    return checks

def check_multi_table_integrity(datasets_dfs):
    """
    Cheap join-coverage checks for known keys if they exist in the set.
    datasets_dfs is a dict: table_name -> df
    """
    integrity = []
    
    # Check appearances and games
    if 'appearances' in datasets_dfs and 'games' in datasets_dfs:
        app_game_ids = datasets_dfs['appearances'].get('game_id')
        game_ids = datasets_dfs['games'].get('game_id')
        if app_game_ids is not None and game_ids is not None:
            coverage = app_game_ids.isin(game_ids).mean() * 100
            integrity.append(f"Appearances -> Games (game_id) coverage: {coverage:.2f}%")

    # Check appearances and players
    if 'appearances' in datasets_dfs and 'players' in datasets_dfs:
        app_player_ids = datasets_dfs['appearances'].get('player_id')
        player_ids = datasets_dfs['players'].get('player_id')
        if app_player_ids is not None and player_ids is not None:
            coverage = app_player_ids.isin(player_ids).mean() * 100
            integrity.append(f"Appearances -> Players (player_id) coverage: {coverage:.2f}%")
            
    return integrity
