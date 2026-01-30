import os
from pathlib import Path
from tabulate import tabulate

def write_markdown_report(output_path, content):
    """
    Helper to write markdown content to a file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

def format_table(df, title=None):
    """
    Formats a dataframe as a markdown table.
    """
    md = ""
    if title:
        md += f"### {title}\n\n"
    md += tabulate(df, headers='keys', tablefmt='github', showindex=False)
    md += "\n\n"
    return md

def generate_table_report(table_name, dataset_id, eda_results):
    """
    Generates markdown for a single table.
    """
    md = f"# Table Report: {table_name}\n\n"
    
    # Overview
    stats = eda_results['stats']
    md += "## Overview\n"
    md += f"- **Rows**: {stats['shape'][0]}\n"
    md += f"- **Columns**: {stats['shape'][1]}\n"
    md += f"- **Duplicates**: {stats['duplicates']}\n"
    md += f"- **Memory Usage**: {stats['memory_usage']:.2f} MB\n\n"
    
    # Missingness
    missing = eda_results['missing']
    md += "## Missingness\n"
    missing_data = []
    for col, pct in list(missing['missing_pct'].items())[:10]: # Top 10
        missing_data.append({'Column': col, 'Missing %': f"{pct:.2f}%"})
    md += tabulate(missing_data, headers='keys', tablefmt='github', showindex=False)
    md += "\n\n"
    
    if missing['high_missing_cols']:
        md += f"**Warning**: High missingness (>50%) in: {', '.join(missing['high_missing_cols'])}\n\n"
        
    # Numeric
    if eda_results['numeric']:
        md += "## Numeric Summary\n"
        num_summary = []
        for col, s in eda_results['numeric']['summary'].items():
            num_summary.append({
                'Column': col,
                'Mean': f"{s['mean']:.2f}",
                'Std': f"{s['std']:.2f}",
                'Min': s['min'],
                'Max': s['max']
            })
        md += tabulate(num_summary, headers='keys', tablefmt='github', showindex=False)
        md += "\n\n"
        
    # Categorical
    if eda_results['categorical']:
        md += "## Categorical Summary\n"
        cat_data = []
        for col, s in eda_results['categorical'].items():
            cat_data.append({
                'Column': col,
                'Unique': s['unique_count'],
                'High Card.': 'Yes' if s['high_cardinality'] else 'No'
            })
        md += tabulate(cat_data, headers='keys', tablefmt='github', showindex=False)
        md += "\n\n"
        
    # Dates
    if eda_results['dates']:
        md += "## Date Parsing\n"
        date_data = []
        for col, s in eda_results['dates'].items():
            date_data.append({
                'Column': col,
                'Success %': f"{s['success_rate']:.2f}%",
                'Min': s['min'],
                'Max': s['max']
            })
        md += tabulate(date_data, headers='keys', tablefmt='github', showindex=False)
        md += "\n\n"
        
    # Football specific
    if eda_results['football']:
        md += "## Football Specific Checks\n"
        for check_name, data in eda_results['football'].items():
            md += f"### {check_name.capitalize()}\n"
            for k, v in data.items():
                md += f"- **{k}**: {v}\n"
            md += "\n"
            
    return md

def generate_dataset_report(dataset_id, dataset_info, tables_info, integrity_results):
    """
    Generates the main EDA.md for a dataset.
    """
    md = f"# Dataset Report: {dataset_id}\n\n"
    
    md += "## Dataset Overview\n"
    md += f"- **Path**: `{dataset_info['path']}`\n"
    md += f"- **Total Tables**: {len(tables_info)}\n\n"
    
    md += "## Files & Tables Discovered\n"
    table_list = []
    for t in tables_info:
        # Link to table report
        table_list.append({
            'Table Name': f"[{t['name']}](tables/{t['name']}.md)",
            'Rows': t.get('rows', 'N/A'),
            'Cols': t.get('cols', 'N/A'),
            'File': t['path'].name
        })
    md += tabulate(table_list, headers='keys', tablefmt='github', showindex=False)
    md += "\n\n"
    
    if integrity_results:
        md += "## Multi-table Integrity Checks\n"
        for res in integrity_results:
            md += f"- {res}\n"
        md += "\n"
        
    md += "## Relevance to Injury Prediction\n"
    md += "This dataset contains "
    has_injuries = any('injury' in t['name'].lower() for t in tables_info)
    has_players = any('player' in t['name'].lower() for t in tables_info)
    has_appearances = any('appearance' in t['name'].lower() or 'game' in t['name'].lower() for t in tables_info)
    
    relevance = []
    if has_injuries: relevance.append("injury episodes")
    if has_players: relevance.append("player attributes")
    if has_appearances: relevance.append("match logs/appearances")
    
    if relevance:
        md += ", ".join(relevance) + "."
    else:
        md += "generic football data."
    md += "\n\n"
    
    return md

def generate_index(dataset_reports):
    """
    Generates the main index.md.
    dataset_reports: list of dicts {'id': id, 'name': name, 'tables': count, 'key_tables': list}
    """
    md = "# EDA Reports Index\n\n"
    
    summary_data = []
    for r in dataset_reports:
        summary_data.append({
            'Dataset ID': f"[{r['id']}]({r['id']}/EDA.md)",
            'Tables': r['tables'],
            'Key Content': ", ".join(r['key_tables'])
        })
        
    md += tabulate(summary_data, headers='keys', tablefmt='github', showindex=False)
    return md
