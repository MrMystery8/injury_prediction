# Injury Risk Pipeline — Constrained Model Guide

This document is a companion to `docs/PIPELINE_DATA_MODEL_GUIDE.md`.

It describes the **same pipeline** and label definition, but a **constrained modeling setup** where the model is allowed to use only:

- **Age**
- **Player position**
- **Minutes played**
- **Days since last played (rest)**
- **Injury history**

If you want the full “first-time reader” walkthrough of joins, leakage policies, and artifacts, start with `docs/PIPELINE_DATA_MODEL_GUIDE.md` and then return here.

---

## 1) What stays the same

The constrained model does **not** change:

- The prediction task: for each `(player_id, week_start)`, predict whether a **time-loss injury (≥7 days out)** starts in the **next 30 days**
- The leakage policy: features are computed **strictly prior** to `week_start`
- The strict modeling mask (eligibility):
  - `ineligible == 0`
  - `not_censored == 1`
  - `active_player == 1`
  - and if present: `label_known == 1`
- The time-based split:
  - Train: `week_start < 2023-06-01`
  - Valid: `2023-06-01 <= week_start < 2024-01-01`
  - Test: `week_start >= 2024-01-01`

---

## 2) What changes (the constrained architecture)

### 2.0 Cohort definition (Top‑5 league clubs, all competitions)

We keep the modeling focus on the **Top‑5 domestic leagues**, but we treat players as “in-scope” if they are affiliated with a Top‑5 league **club** in a given season.

Concretely:
- The panel’s `competition_id` stays one of `{GB1, ES1, L1, IT1, FR1}` (the player’s primary Top‑5 league for that season).
- Workload/rest features still use **all competitions** from the backbone appearances (league + cups + European competitions + internationals), so non‑league minutes affect the risk score.

Implementation: `src/data/processing.py` (`build_weekly_panel_top5_club_cohort`) and `make_panel.py`.

### 2.1 New/updated feature: rest (`days_since_last_played`)

We add a rest proxy to the panel:

- `days_since_last_played`: days between `week_start` and the most recent prior match date where `minutes_played > 0`
- Strictly prior (no leakage): only matches with `game_date < week_start` count
- Sentinel for “never played before this date in the backbone extract”: `9999`

Implementation: `src/data/processing.py` (inside `add_workload_features`).

### 2.2 The **only** features the model can use

The constrained model uses:

**A) Age**
- `age`

**B) Position**
- one-hot dummies from `position` (columns `pos_*`)

**C) Minutes played (workload)**
- `minutes_last_4w` (sum of minutes in the last 28 days, strictly prior)

**D) Rest**
- `days_since_last_played`

**E) Injury history (strictly prior)**
- `injuries_last_365d`
- `days_out_last_365d`
- `days_since_last_injury_start`
- `days_since_last_injury_end`
- `last_injury_days_out`

The definitive list written by evaluation is:
- `reports/manifest/model_features.csv`

Code locations:
- feature selection: `quick_eval.py`, `audit_eval.py`, `train_model.py`
- shared constants/utilities: `src/constrained_model.py`

---

## 3) Reproducible run steps

All commands below assume you use the repo venv (`./.venv/bin/python`).

### 3.1 Build the panel

```bash
./.venv/bin/python make_panel.py
```

Key output:
- `data/processed/panel.parquet`

### 3.2 Evaluate the constrained model

```bash
./.venv/bin/python quick_eval.py --panel data/processed/panel.parquet --print-manifest
./.venv/bin/python audit_eval.py --panel data/processed/panel.parquet
```

Key outputs:
- `reports/manifest/eval_manifest.json` (metrics + split manifest)
- `reports/manifest/audit_results.json` (sanity checks + negative control)
- `reports/manifest/model_features.csv` (exact features used)

### 3.3 Use the dashboard to trace + score players

See: `docs/STREAMLIT_DASHBOARD_GUIDE.md`

---

## 4) Current performance snapshot (constrained features)

From the latest local run on this repo state:

- Test prevalence: **0.0989**
- Average Precision (AP): **0.1451**
- ROC-AUC: **0.6291**
- Brier score: **0.0875**
- Precision@Top5%: **0.1793**
- Recall@Top5%: **0.0906**

Interpretation (high level):
- Performance drops slightly vs the unconstrained baseline (expected) because context/congestion/market features are removed.
- The model still concentrates injuries in the top-ranked alerts better than random ranking (AP > prevalence).

---

## 5) Practical improvements that still respect the feature constraint

If you want to improve results without leaving the allowed feature groups:

### 5.1 Minutes played (still “minutes”)
- Add additional windows: `minutes_last_1w`, `minutes_last_2w`, `minutes_last_8w`
- Add ratio/shape: `acwr` (acute:chronic) using only minutes

### 5.2 Rest (still “rest”)
- Add “min rest in last 14 days”
- Add “days since last match” variants that treat 0-minute appearances differently (if present in source)

### 5.3 Injury history (still “injury history”)
- Add shorter windows: 30d and 180d counts/days-out
- Add “injury-type history” if `injury_reason` is reliable (e.g., muscle vs impact)

### 5.4 Data handling (still allowed)
- Impute `age` with median age (not 0) and add a missingness indicator
- Keep an explicit “position unknown” bucket instead of silently all-zeros
