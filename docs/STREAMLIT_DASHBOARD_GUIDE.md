# Streamlit Dashboard Guide (Reproducible)

This repo includes a Streamlit “Observability Studio” so you can run the pipeline, inspect artifacts, trace a player, and (now) **score player-weeks** with the **constrained** model.

## 1) Start the dashboard

From the repo root:

```bash
./.venv/bin/streamlit run ui/app.py
```

If `streamlit` is missing from your venv, install UI deps:

```bash
./.venv/bin/pip install -r requirements-ui.txt
```

## 2) One-time pipeline steps (to generate required artifacts)

In the dashboard’s **Pipeline Runner** tab you can click:

1. **Build Panel** → runs `make_panel.py` and writes:
   - `data/processed/panel.parquet`
   - plus inspectable tables under `data/interim/` and `reports/manifest/`
2. **Quick Eval (+manifest)** → runs `quick_eval.py` and writes:
   - `reports/manifest/eval_manifest.json`
   - `reports/manifest/model_features.csv`
   - `reports/manifest/feature_importance_validation.csv`

You can also run these from the terminal:

```bash
./.venv/bin/python make_panel.py
./.venv/bin/python quick_eval.py --panel data/processed/panel.parquet --print-manifest
```

## 3) What each tab is for

### Pipeline Runner
- “One-click” buttons for the allowlisted scripts (safe to run from the UI).
- Artifact status table showing what exists and when it was updated.

### Panel Explorer
- Filters and preview queries via DuckDB (avoids full parquet loads).
- Lets you sanity-check `target`, `ineligible`, `position`, etc. interactively.

### Joins / Features / Eval
- Renders the main observability artifacts from `reports/manifest/`:
  - join coverage samples
  - missingness
  - the exact model feature list
  - evaluation metrics + audit results

### Player Trace + Risk (Constrained Model)
This tab now supports two related workflows:

**A) Generate trace artifacts**
1. Lookup the player by **name** (from `data/interim/backbone_players.parquet`) or enter a digits-only `player_id`.
2. Click **Generate trace** → runs `trace_player.py` and shows:
   - timeline image (`reports/figures/player_<id>_timeline.png`)
   - timeline HTML path (`reports/html/player_<id>_timeline.html`)
   - tail of panel rows (`reports/samples/player_<id>_panel_rows.csv`)

**B) Score player-weeks (probabilities)**
1. Enter/select a `player_id`.
2. Click **Score player risk**.
3. The app trains/loads (cached) the same constrained model family used by `quick_eval.py`:
   - strict training mask (`ineligible==0`, `not_censored==1`, `active_player==1`, and `label_known==1` if present)
   - time-based split (train/valid/test)
4. It displays:
   - a risk-over-time plot for the selected player
   - a table (tail) similar to the example shown in `docs/PIPELINE_DATA_MODEL_GUIDE.md`
   - a CSV download of the scored tail rows

## 4) “Why do I need Quick Eval if the dashboard can score?”

You don’t *have* to run `quick_eval.py` to see per-player scoring in the dashboard.

But `quick_eval.py` is still the canonical script that:
- produces the evaluation artifacts under `reports/manifest/`
- records the exact feature list and permutation importance for write-up/reporting

