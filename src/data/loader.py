
import pandas as pd
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_date_parser(series: pd.Series) -> pd.Series:
    """Coerces a series to datetime, logging invalid rates."""
    parsed = pd.to_datetime(series, errors='coerce')
    n_total = len(series)
    n_nan = parsed.isna().sum()
    if n_total > 0:
        pct_nan = (n_nan / n_total) * 100
        if pct_nan > 1:
            logger.warning(f"Date parse missingness: {pct_nan:.2f}% ({n_nan}/{n_total}) in {series.name}")
    return parsed

def scan_headers(file_path: Path) -> List[str]:
    """Reads only the first line of a CSV to get headers."""
    try:
        return pd.read_csv(file_path, nrows=0).columns.tolist()
    except Exception as e:
        logger.warning(f"Failed to read headers from {file_path}: {e}")
        return []

def detect_tables(root_path: Path, cache_path: Path = None) -> Dict[str, Path]:
    """
    Scans a directory for tables matching required signatures.
    Uses a header-only scan for efficiency.
    Returns: Dict[start_key -> file_path]
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading schema manifest from {cache_path}")
        with open(cache_path, 'r') as f:
            str_map = json.load(f)
            return {k: Path(v) for k, v in str_map.items()}

    logger.info(f"Scanning {root_path} for tables...")
    
    # Signatures: map known table types to required columns (set of strings)
    signatures = {
        'injuries': {'player_id', 'from_date', 'days_missed'}, # football-datasets typical
        'injuries_alt': {'player_id', 'injury_start', 'days_missed'}, # possible variant
        'injuries_irrazional': {'player_id', 'from', 'until', 'Days'}, # irrazional schema
        'profiles': {'player_id', 'height', 'position', 'date_of_birth'},
        'profiles_alt': {'player_id', 'height', 'primary_position'},
        'market_value': {'player_id', 'date', 'market_value_in_eur'},
        'market_value_alt': {'player_id', 'date_unix', 'value'},
        'transfers': {'player_id', 'transfer_date', 'from_team_id'}
    }
    
    found_tables = {}
    csv_files = list(root_path.rglob("*.csv"))
    # Filter out "latest" market value to prefer historical
    csv_files = [p for p in csv_files if "player_latest_market_value" not in p.name and "player_latest_market_value" not in str(p.parent)]
    
    for csv_file in csv_files:
        cols = set(scan_headers(csv_file))
        for key, req_cols in signatures.items():
            if req_cols.issubset(cols):
                # Prefer shorter paths (root files) if duplicates, or just taking first match
                if key not in found_tables:
                    logger.info(f"Found {key} -> {csv_file.name}")
                    found_tables[key] = csv_file
    
    # Normalize keys: e.g. injuries_alt -> injuries
    normalized = {}
    for k, v in found_tables.items():
        norm_key = k.split('_alt')[0]
        if norm_key not in normalized:
            normalized[norm_key] = v
            
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump({k: str(v) for k, v in normalized.items()}, f, indent=2)
            
    return normalized

def load_backbone(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Loads core player-scores tables: appearances, games, players, competitions.
    """
    required = ['appearances', 'games', 'players', 'competitions']
    dfs = {}
    
    for name in required:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Backbone table missing: {path}")
        
        logger.info(f"Loading backbone: {name}")
        dfs[name] = pd.read_csv(path)
        
        # Standardize standard cols
        if 'date' in dfs[name].columns:
            dfs[name]['date'] = robust_date_parser(dfs[name]['date'])
            
    # Explicit mapping for games.date -> game_date to avoid confusion
    if 'date' in dfs['games'].columns:
        dfs['games'] = dfs['games'].rename(columns={'date': 'game_date'})
        
    return dfs

def load_labels(data_dir: Path, signatures_cache: Path = None) -> Dict[str, pd.DataFrame]:
    """
    Loads label/enrichment tables from football-datasets using schema detection.
    """
    table_map = detect_tables(data_dir, signatures_cache)
    dfs = {}
    
    # Load Injuries
    if 'injuries' in table_map:
        path = table_map['injuries']
        logger.info(f"Loading injuries from {path}")
        df = pd.read_csv(path)
        
        # Rename commonly used columns to standard
        rename_map = {
            'from_date': 'injury_start',
            'from': 'injury_start',
            'start_date': 'injury_start',
            'end_date': 'injury_end',
            'until_date': 'injury_end',
            'until': 'injury_end',
            'days': 'days_out',
            'Days': 'days_out',
            'days_missed': 'days_out'
        }
        df = df.rename(columns=rename_map)
        
        # Parse dates
        if 'injury_start' in df.columns:
            df['injury_start'] = robust_date_parser(df['injury_start'])
        if 'injury_end' in df.columns:
            df['injury_end'] = robust_date_parser(df['injury_end'])
            
        dfs['injuries'] = df
    else:
        logger.warning("No injuries table found in labels dataset!")

    # Load Profiles
    if 'profiles' in table_map:
        path = table_map['profiles']
        logger.info(f"Loading profiles from {path}")
        df = pd.read_csv(path)
        
        # Parse DoB
        if 'date_of_birth' in df.columns:
            df['date_of_birth'] = robust_date_parser(df['date_of_birth'])
            
        dfs['profiles'] = df
        
    # Load Transfers
    if 'transfers' in table_map:
        path = table_map['transfers']
        logger.info(f"Loading transfers from {path}")
        df = pd.read_csv(path)
        if 'transfer_date' in df.columns:
            df['transfer_date'] = robust_date_parser(df['transfer_date'])
        dfs['transfers'] = df
        
    # Load Market Values
    if 'market_value' in table_map:
        path = table_map['market_value']
        logger.info(f"Loading market values from {path}")
        df = pd.read_csv(path)
        # Handle date_unix or date
        if 'date' in df.columns:
            df['date'] = robust_date_parser(df['date'])
        elif 'date_unix' in df.columns:
            # Check if it's really unix (int) or string iso. Sample showed string iso '2023-12-19'.
            # Robust parser handles both.
            df['date'] = robust_date_parser(df['date_unix'])
            df = df.drop(columns=['date_unix'])
            
        dfs['market_values'] = df
            
    return dfs

if __name__ == "__main__":
    import sys
    # Default to current dir if no arg provided
    root = Path(".")
    print(f"Running Loader Inspection on {root.resolve()}...")
    
    # 1. Detect Tables
    tables = detect_tables(root)
    print("\n--- Detected Tables ---")
    for k, v in tables.items():
        print(f"[{k}] -> {v.name} ({v.stat().st_size / 1024:.1f} KB)")
        
    # 2. Check Backbone
    try:
        backbone = load_backbone(root / "data/raw/davidcariboo:player-scores")
        print("\n--- Backbone Data ---")
        for k, df in backbone.items():
            print(f"[{k}] Shape: {df.shape}")
    except Exception as e:
        print(f"\n[!] Backbone Check Skipped/Failed: {e}")
        
    print("\nLoader check complete.")
