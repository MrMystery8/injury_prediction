import os
import re
import pandas as pd
from pathlib import Path
import logging

def safe_slug(name):
    """
    Normalizes a name to a safe slug.
    Rules: lowercase, spaces -> _, - -> _, remove non-alphanumerics except _.
    """
    name = name.lower()
    name = re.sub(r'[\s\-]+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name

def discover_datasets(data_root):
    """
    Discovers datasets in the data_root.
    Returns a list of dicts: {'id': dataset_id, 'path': path, 'is_dir': bool}
    """
    data_root = Path(data_root)
    datasets = []
    
    if not data_root.exists():
        return datasets
        
    for child in data_root.iterdir():
        if child.name.startswith('.'):
            continue
            
        dataset_name = child.stem if child.is_file() else child.name
        dataset_id = safe_slug(dataset_name)
        
        datasets.append({
            'id': dataset_id,
            'name': dataset_name,
            'path': child,
            'is_dir': child.is_dir()
        })
        
    return datasets

def load_table(file_path, sample_rows=None):
    """
    Loads a single table (CSV or Parquet).
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    try:
        if extension == '.csv':
            if sample_rows:
                # Load first few rows to get schema, then potentially sample
                # For basic EDA we might want a representative sample if file is huge
                # But specification says Cap heavy tables
                df = pd.read_csv(file_path, low_memory=False, encoding_errors="replace", on_bad_lines="skip", nrows=sample_rows)
            else:
                df = pd.read_csv(file_path, low_memory=False, encoding_errors="replace", on_bad_lines="skip")
        elif extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return None, f"Unsupported extension: {extension}"
            
        return df, None
    except Exception as e:
        return None, str(e)

def get_dataset_tables(dataset_info):
    """
    Returns a list of tables for a given dataset.
    """
    tables = []
    path = dataset_info['path']
    
    if dataset_info['is_dir']:
        # Recursive search for csv/parquet
        for ext in ['*.csv', '*.parquet']:
            for file_path in path.rglob(ext):
                tables.append({
                    'name': file_path.stem,
                    'path': file_path
                })
    else:
        if path.suffix.lower() in ['.csv', '.parquet']:
            tables.append({
                'name': path.stem,
                'path': path
            })
            
    return tables
