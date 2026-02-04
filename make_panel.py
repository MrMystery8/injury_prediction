
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.loader import load_backbone, load_labels
from src.data.processing import (
    filter_top5, clean_injuries,
    build_weekly_panel_top5_club_cohort,
    add_workload_features, add_labels_and_exclusions,
    add_congestion_features, add_context_features, add_injury_history_features
)
from src.observability.io import write_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_quality_report(panel, backbone_coverage, gate_status, output_path):
    """
    Generates a markdown report of the pipeline execution decision log.
    """
    report = f"""# Player-Week Panel: Data Quality Report

## 1. Pipeline Decisions
- **Backbone**: `davidcariboo/player-scores`
- **Labels**: `football-datasets` (Injuries)
- **Top-5 Filter**: Applied (Competition IDs + Country/Tier Fallback)
- **Windowing**: Global In-Season Weeks (No Player Min/Max gaps)

## 2. Gates & Validation
### ID Overlap Gate
- **Status**: {gate_status['status']}
- **Backbone Players (Top-5)**: {gate_status['n_backbone']}
- **Label Players**: {gate_status['n_labels']}
- **Join Overlap**: {gate_status['overlap_pct']:.2f}% (Threshold: 70%)

### Ineligible Sanity
- **Total Rows**: {len(panel)}
- **Ineligible Rows**: {panel['ineligible'].sum()} ({panel['ineligible'].mean()*100:.2f}%)
- **Target Count**: {panel['target'].sum()} ({panel['target'].mean()*100:.2f}%)

## 3. Sample Data
### Positive Labels (Injury in next 30d)
"""
    pos_sample = panel[panel['target'] == 1].head(5)
    report += pos_sample.to_markdown() + "\n\n"
    
    report += "### Ineligible Rows (Already Injured)\n"
    inel_sample = panel[panel['ineligible'] == 1].head(5)
    report += inel_sample.to_markdown() + "\n\n"
    
    report += "### Negative Rows (Healthy High-Workload)\n"
    neg_sample = panel[(panel['target'] == 0) & (panel['ineligible'] == 0)].sort_values('minutes_last_4w', ascending=False).head(5)
    report += neg_sample.to_markdown() + "\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="data/raw/davidcariboo:player-scores")
    parser.add_argument("--labels", default="data/raw/xfkzujqjvx97n:football-datasets")
    parser.add_argument("--out", default="data/processed/panel.parquet")
    parser.add_argument("--report", default="reports/data_quality_report.md")
    parser.add_argument("--interim-dir", default="data/interim", help="Write inspectable interim parquet tables here.")
    parser.add_argument("--manifest", default="reports/manifest/panel_build_manifest.json", help="JSON manifest for observability.")
    args = parser.parse_args()
    
    backbone_dir = Path(args.backbone)
    labels_dir = Path(args.labels)
    output_path = Path(args.out)
    report_path = Path(args.report)
    interim_dir = Path(args.interim_dir)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Backbone
    logger.info("Step 1: Loading Backbone...")
    bb_dfs = load_backbone(backbone_dir)

    # Save inspectable backbone tables
    try:
        bb_dfs["appearances"].to_parquet(interim_dir / "backbone_appearances.parquet", index=False)
        bb_dfs["games"].to_parquet(interim_dir / "backbone_games.parquet", index=False)
        bb_dfs["players"].to_parquet(interim_dir / "backbone_players.parquet", index=False)
        bb_dfs["competitions"].to_parquet(interim_dir / "backbone_competitions.parquet", index=False)
        logger.info(f"Wrote interim backbone tables to {interim_dir}")
    except Exception as e:
        logger.warning(f"Failed writing interim backbone tables: {e}")
    
    # 2. Filter Top-5
    logger.info("Step 2: Filtering Top-5...")
    games_top5, apps_top5 = filter_top5(bb_dfs['games'], bb_dfs['appearances'], bb_dfs['competitions'])

    try:
        games_top5.to_parquet(interim_dir / "backbone_games_top5.parquet", index=False)
        apps_top5.to_parquet(interim_dir / "backbone_appearances_top5.parquet", index=False)
    except Exception as e:
        logger.warning(f"Failed writing interim Top-5 tables: {e}")

    # 3. Build Panel Skeleton (Top-5 club cohort; league calendar weeks)
    logger.info("Step 3: Building Weekly Panel...")
    panel = build_weekly_panel_top5_club_cohort(games_top5, bb_dfs["games"], bb_dfs["appearances"])
    top5_player_ids = set(panel["player_id"].unique())
    logger.info(f"Top-5 Club Cohort Players: {len(top5_player_ids)}")
    logger.info(f"Panel Skeleton: {len(panel)} rows")

    # 4. Load Labels & Clean
    logger.info("Step 4: Loading Labels...")
    lbl_dfs = load_labels(labels_dir)
    injuries = clean_injuries(lbl_dfs['injuries'])

    # Save inspectable label tables (cleaned where applicable)
    try:
        injuries.to_parquet(interim_dir / "labels_injuries.parquet", index=False)
        if "profiles" in lbl_dfs:
            lbl_dfs["profiles"].to_parquet(interim_dir / "labels_profiles.parquet", index=False)
        if "transfers" in lbl_dfs and lbl_dfs["transfers"] is not None:
            lbl_dfs["transfers"].to_parquet(interim_dir / "labels_transfers.parquet", index=False)
        if "market_values" in lbl_dfs and lbl_dfs["market_values"] is not None:
            lbl_dfs["market_values"].to_parquet(interim_dir / "labels_market_values.parquet", index=False)
        logger.info(f"Wrote interim label tables to {interim_dir}")
    except Exception as e:
        logger.warning(f"Failed writing interim label tables: {e}")
    
    # 5. GATE: ID Overlap
    logger.info("Step 5: Running ID Overlap Gate...")
    label_player_ids = set(injuries['player_id'].unique())
    intersect = top5_player_ids.intersection(label_player_ids)
    overlap_pct = len(intersect) / len(top5_player_ids) * 100
    
    gate_status = {
        'n_backbone': len(top5_player_ids),
        'n_labels': len(label_player_ids),
        'overlap_pct': overlap_pct,
        'status': 'PASS'
    }
    
    logger.info(f"ID Overlap: {overlap_pct:.2f}% ({len(intersect)}/{len(top5_player_ids)})")
    
    if overlap_pct < 70:
        gate_status['status'] = 'FAIL'
        logger.error(f"CRITICAL: ID Overlap {overlap_pct:.2f}% is below 70% threshold.")
        logger.error("Action: ABORTING. Please switch label source to 'irrazional/transfermarkt-injuries' or implement ID mapping.")
        
        # Write report even on fail
        generate_quality_report(pd.DataFrame(), 0, gate_status, report_path)
        write_json(
            manifest_path,
            {
                "status": "FAIL",
                "gate_status": gate_status,
                "paths": {
                    "backbone_dir": str(backbone_dir),
                    "labels_dir": str(labels_dir),
                    "output_panel": str(output_path),
                    "report_md": str(report_path),
                    "interim_dir": str(interim_dir),
                },
            },
        )
        exit(1)
        
    # 6. Features
    # Use ALL appearances for workload (physiologically correct to include Cup matches)
    logger.info("Step 6a: Adding Workload Features (All Competitions)...")
    panel = add_workload_features(panel, bb_dfs['appearances'], bb_dfs['games'])
    
    logger.info("Step 6b: Adding Congestion Features (Short-term matches/load)...")
    panel = add_congestion_features(panel, bb_dfs['appearances'], bb_dfs['games'])
    
    logger.info("Step 6c: Adding Context Features (Transfers & Market Value)...")
    panel = add_context_features(
        panel, 
        lbl_dfs.get('transfers'), 
        lbl_dfs.get('market_values')
    )
    
    # 7. Labels
    logger.info("Step 7: Adding Labels & Exclusions (with label_known check)...")
    # need profiles for label_known
    profiles_raw = lbl_dfs.get('profiles')
    panel = add_labels_and_exclusions(panel, injuries, profiles_raw)

    # 7b. Injury history features (strictly prior)
    logger.info("Step 7b: Adding Injury History Features...")
    panel = add_injury_history_features(panel, injuries, windows_days=(30, 180, 365), min_days_out=7)
    
    # 8. Enrichment (Profiles)
    if 'profiles' in lbl_dfs:
        logger.info("Step 8: enriching with profiles...")
        profiles = lbl_dfs['profiles']
        # Join strict
        # Clean profiles?
        cols_to_use = ['player_id', 'height', 'position', 'date_of_birth']
        # check avail cols
        avail = [c for c in cols_to_use if c in profiles.columns]
        panel = panel.merge(profiles[avail], on='player_id', how='left')

        # Clean height: 0 / unrealistic values -> NaN
        if 'height' in panel.columns:
            panel['height'] = pd.to_numeric(panel['height'], errors='coerce')
            panel.loc[(panel['height'] <= 0) | (panel['height'] < 140) | (panel['height'] > 210), 'height'] = np.nan
        
        # Calc Age
        if 'date_of_birth' in panel.columns:
            panel['date_of_birth'] = pd.to_datetime(panel['date_of_birth'], errors='coerce')
            panel['age'] = (panel['week_start'] - panel['date_of_birth']).dt.days / 365.25
        
    # 9. Structural Fixes (Right Censoring & Active Window)
    logger.info("Step 9: Adding Structural Flags...")
    from src.data.processing import add_structural_flags
    panel = add_structural_flags(panel, injuries, bb_dfs['appearances'], bb_dfs['games'])

    # 10. Final Validation
    logger.info("Step 10: Final Sanity Checks...")
    n_pos = panel['target'].sum()
    n_inel = panel['ineligible'].sum()
    logger.info(f"Positives: {n_pos} ({n_pos/len(panel)*100:.2f}%)")
    logger.info(f"Ineligible: {n_inel} ({n_inel/len(panel)*100:.2f}%)")
    
    if n_pos == 0:
        logger.warning("WARNING: 0 Positives found! Logic check needed.")
        
    # 10. Save
    logger.info(f"Saving to {output_path}...")
    panel.to_parquet(output_path)
    
    generate_quality_report(panel, overlap_pct, gate_status, report_path)
    write_json(
        manifest_path,
        {
            "status": "PASS",
            "gate_status": gate_status,
            "counts": {
                "backbone_games": int(len(bb_dfs["games"])),
                "backbone_appearances": int(len(bb_dfs["appearances"])),
                "top5_games": int(len(games_top5)),
                "top5_appearances": int(len(apps_top5)),
                "panel_rows": int(len(panel)),
                "panel_players": int(panel["player_id"].nunique()) if "player_id" in panel.columns else None,
                "panel_positives": int(n_pos),
                "panel_ineligible": int(n_inel),
            },
            "paths": {
                "backbone_dir": str(backbone_dir),
                "labels_dir": str(labels_dir),
                "output_panel": str(output_path),
                "report_md": str(report_path),
                "interim_dir": str(interim_dir),
            },
        },
    )
    logger.info("Pipeline Completed.")

if __name__ == "__main__":
    main()
