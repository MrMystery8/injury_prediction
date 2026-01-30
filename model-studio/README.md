# Model Studio (React + Vite)

This is an optional dashboard UI. The main inspection/debug UI is Streamlit (`ui/app.py`).

## Run locally
`node_modules/` is intentionally not committed. Recreate it with:

```bash
cd model-studio
npm ci
npm run dev
```

## Data inputs
The app reads exported artifacts written by:
- `python3 quick_eval.py --panel data/processed/panel.parquet --print-manifest`

In particular:
- `model-studio/public/model_data.json` (gitignored; generated locally)
