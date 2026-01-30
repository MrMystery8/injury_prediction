# GitHub Upload Checklist (this repo)

This repo is designed to **not** commit large local artifacts (datasets, parquet outputs, virtualenvs, node_modules). GitHub should contain code + lightweight docs only.

## 1) Confirm ignores are in place
Key paths that must remain local:
- `.venv/`
- `data/**` (raw Kaggle datasets + processed parquet)
- `reports/**` (generated artifacts)
- `model-studio/node_modules/`

These are already covered by `.gitignore`.

## 2) Recommended: upload via Git (not zip)
If you upload a zip of the folder, you can accidentally include local data and environments. Use Git so `.gitignore` is enforced.

## 3) Initialize a repo and check what will be committed
From repo root:
```bash
git init
git add -A
git status
```

You should **not** see `data/`, `reports/`, `.venv/`, or `model-studio/node_modules/` staged.

If you want to double-check ignored files:
```bash
git status --ignored
```

## 4) React app reproducibility
`model-studio/node_modules/` is not committed. On a fresh clone:
```bash
cd model-studio
npm ci
npm run dev
```

## 5) Python reproducibility (suggested)
Recommended workflow:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

Then place datasets under `data/raw/` and run the pipeline per `README.md`.

## 6) Create the remote and push
```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

