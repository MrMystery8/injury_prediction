import argparse
import logging
from pathlib import Path
from src.loader import discover_datasets, get_dataset_tables, load_table
from src.eda_core import (
    get_basic_stats, get_missingness, get_numeric_summary,
    get_categorical_summary, detect_and_parse_dates,
    detect_id_candidates, run_football_checks,
    check_multi_table_integrity
)
from src.reporting import (
    write_markdown_report, generate_table_report,
    generate_dataset_report, generate_index
)

def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "pipeline.log"),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Loader + Basic EDA")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw datasets")
    parser.add_argument("--out", default="reports", help="Path to output reports")
    parser.add_argument("--sample-rows", type=int, default=200000, help="Cap rows for heavy tables")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    reports_root = Path(args.out)
    reports_root.mkdir(parents=True, exist_ok=True)

    setup_logging(reports_root)
    logging.info(f"Starting EDA pipeline. Discovery in: {data_root}")

    datasets = discover_datasets(data_root)
    logging.info(f"Discovered {len(datasets)} datasets.")

    index_data = []

    for ds in datasets:
        ds_id = ds['id']
        ds_report_dir = reports_root / ds_id
        ds_report_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Processing dataset: {ds_id}")
        
        tables = get_dataset_tables(ds)
        ds_tables_info = []
        ds_dfs = {}
        
        load_log = f"# Load Log: {ds_id}\n\n"
        load_log += "| Table | Status | Rows | Cols | Memory (MB) | Note |\n"
        load_log += "|---|---|---|---|---|---|\n"

        for table in tables:
            t_name = table['name']
            t_path = table['path']
            
            df, err = load_table(t_path, sample_rows=args.sample_rows)
            
            if df is not None:
                logging.info(f"  Loaded table: {t_name} ({len(df)} rows)")
                ds_dfs[t_name] = df
                
                # Run EDA
                eda_results = {
                    'stats': get_basic_stats(df),
                    'missing': get_missingness(df),
                    'numeric': get_numeric_summary(df),
                    'categorical': get_categorical_summary(df),
                    'dates': detect_and_parse_dates(df),
                    'ids': detect_id_candidates(df),
                    'football': run_football_checks(df, t_name)
                }
                
                # Generate per-table report
                t_report_md = generate_table_report(t_name, ds_id, eda_results)
                write_markdown_report(ds_report_dir / "tables" / f"{t_name}.md", t_report_md)
                
                # Track for dataset report
                table_info = {
                    'name': t_name,
                    'path': t_path,
                    'rows': len(df),
                    'cols': len(df.columns)
                }
                ds_tables_info.append(table_info)
                
                load_log += f"| {t_name} | Success | {len(df)} | {len(df.columns)} | {eda_results['stats']['memory_usage']:.2f} | |\n"
            else:
                logging.warning(f"  Failed to load table {t_name}: {err}")
                load_log += f"| {t_name} | Failed | - | - | - | {err} |\n"

        write_markdown_report(ds_report_dir / "load_log.md", load_log)

        # Multi-table integrity
        integrity_results = check_multi_table_integrity(ds_dfs)
        
        # Dataset report
        ds_report_md = generate_dataset_report(ds_id, ds, ds_tables_info, integrity_results)
        write_markdown_report(ds_report_dir / "EDA.md", ds_report_md)
        
        # Index entry
        key_tables = []
        if any('injury' in t['name'].lower() for t in ds_tables_info): key_tables.append("Injuries")
        if any('player' in t['name'].lower() for t in ds_tables_info): key_tables.append("Players")
        if any('appearance' in t['name'].lower() for t in ds_tables_info): key_tables.append("Appearances")
        
        index_data.append({
            'id': ds_id,
            'name': ds['name'],
            'tables': len(ds_tables_info),
            'key_tables': key_tables
        })

    # Global Index
    index_md = generate_index(index_data)
    write_markdown_report(reports_root / "index.md", index_md)
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
