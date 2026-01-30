# Injury Prediction (Top-5 Leagues 2020-2025)

## Overview
This project builds a player-based model to predict time-loss injuries (≥7 days) in the next 30 days.

## Setup

1. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```

   Recommended (UI deps for inspection):
   ```bash
   pip install -r requirements-ui.txt
   ```

2. **Data**:
   Place the following Kaggle datasets in `data/raw/`:
   - `davidcariboo/player-scores`
   - `xfkzujqjvx97n/football-datasets` (injuries, profiles, market values, transfers)
   - `irrazional/transfermarkt-injuries` (optional backup label source)
   - `sananmuzaffarov/european-football-injuries-2020-2025` (optional; name-based, not used in main pipeline)

   Alternatively, if you have the Kaggle API set up (`~/.kaggle/kaggle.json`), the scripts can download them automatically (implementation pending).

3. **Running**:
   - Dataset EDA (raw): `python3 run_eda.py --data-root data/raw --out reports`
   - Build player-week panel: `python3 make_panel.py` (writes `data/processed/panel.parquet`)
   - Verify panel (leakage / label checks): `python3 verify_panel.py --panel data/processed/panel.parquet`
   - Audit evaluation protocol (manifest + negative control): `python3 audit_eval.py --panel data/processed/panel.parquet` (also writes `reports/manifest/audit_results.json`)
   - Train model: `python3 train_model.py --panel data/processed/panel.parquet`
   - Quick eval export for Model Studio: `python3 quick_eval.py --panel data/processed/panel.parquet --print-manifest` (also writes eval artifacts to `reports/manifest/` + `reports/figures/`)

## Observability (Inspectable Pipeline Artifacts)
All stage-wise outputs go under:
- `reports/manifest/` (JSON/CSV manifests)
- `reports/samples/` (CSV/parquet samples)
- `reports/figures/` (PNG figures)
- `reports/html/` (static HTML trace views)

## Streamlit UI (recommended for inspection)
Install UI deps:
- `pip install -r requirements-ui.txt`

Run:
- `streamlit run ui/app.py`

## React Model Studio (optional)
The React app lives in `model-studio/`. `node_modules/` is intentionally not committed.

Run locally:
- `cd model-studio && npm ci`
- `npm run dev`

## Detailed documentation
- Full table schemas, join logic, feature provenance, model evaluation, and a Griezmann lineage walkthrough: `docs/PIPELINE_DATA_MODEL_GUIDE.md`
- GitHub upload checklist (what not to commit, how to reproduce): `docs/GITHUB_UPLOAD_CHECKLIST.md`

Runbook:
1. Raw schema + file discovery: `python3 inspect_raw.py`
2. Build canonical panel (+ interim parquet tables): `python3 make_panel.py` (also writes `reports/manifest/panel_build_manifest.json`)
3. Panel integrity: `python3 verify_panel.py --panel data/processed/panel.parquet` and `python3 audit_eval.py --panel data/processed/panel.parquet`
4. Export inspectable artifacts: `python3 inspect_panel.py && python3 inspect_joins.py && python3 feature_report.py`
5. Export model eval artifacts for dashboards: `python3 quick_eval.py --panel data/processed/panel.parquet --print-manifest`
6. Player lineage trace (repeat for a few IDs): `python3 trace_player.py --player-id <ID>`

## Structure
- `data/`: Raw and processed data.
- `src/`: Source code for the pipeline.
- `notebooks/`: EDA and experiments.
- `reports/`: Generated figures and metrics.
