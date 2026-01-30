import pandas as pd
from rapidfuzz import process, fuzz

def resolve_entities(injury_names, player_df, threshold=80):
    """
    Maps injury player names to player_ids using fuzzy matching.
    
    Args:
        injury_names: List or Series of unique names from injury dataset.
        player_df: DataFrame containing 'player_id' and 'name' (or 'player_name').
        threshold: Score threshold for automatic acceptance (conceptually).
        
    Returns:
        DataFrame with columns ['injury_name', 'candidate_name', 'match_score', 'player_id']
    """
    
    # Ensure player names are strings
    player_names = player_df['name'].dropna().unique()
    player_lookup = player_df.set_index('name')['player_id'].to_dict()
    
    results = []
    
    count = 0
    total = len(injury_names)
    
    for inj_name in injury_names:
        if not isinstance(inj_name, str):
            continue
            
        # Extract best match
        match = process.extractOne(inj_name, player_names, scorer=fuzz.token_sort_ratio)
        
        if match:
            candidate_name, score, index = match
            player_id = player_lookup.get(candidate_name)
            
            results.append({
                'injury_name': inj_name,
                'candidate_name': candidate_name,
                'match_score': score,
                'player_id': player_id
            })
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} names...")
            
    return pd.DataFrame(results)
