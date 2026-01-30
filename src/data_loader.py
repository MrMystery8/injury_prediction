import pandas as pd
import os
import glob

def load_all_csvs(data_dir):
    """
    Loads all CSV files from the specified directory into a dictionary of DataFrames.
    Keys are the filenames (without extension).
    """
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    datasets = {}
    
    for file_path in csv_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_csv(file_path)
            datasets[filename] = df
            print(f"Loaded {filename}: {df.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return datasets

def standardize_columns(df):
    """
    Converts columns to snake_case and strips whitespace.
    """
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('-', '_')
                  .str.replace('/', '_')
                  .str.replace('.', ''))
    return df

def standardize_datasets(datasets):
    """
    Standardizes column names for all loaded datasets.
    """
    cleaned_datasets = {}
    for name, df in datasets.items():
        cleaned_datasets[name] = standardize_columns(df)
    return cleaned_datasets
