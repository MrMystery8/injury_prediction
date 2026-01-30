import argparse
import json
import os
from pathlib import Path

_mpl_dir = Path("reports") / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib
matplotlib.use("Agg")  # headless/sandbox-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, brier_score_loss, roc_auc_score
from sklearn.inspection import permutation_importance

from src.observability.paths import ensure_reports_dirs
from src.observability.io import write_json

def _manifest(df: pd.DataFrame, features: list[str], train_mask: pd.Series, valid_mask: pd.Series, test_mask: pd.Series) -> dict:
    def split_stats(mask: pd.Series) -> dict:
        part = df.loc[mask]
        if part.empty:
            return {"rows": 0, "prev": None, "min_week": None, "max_week": None}
        return {
            "rows": int(len(part)),
            "prev": float(part["target"].mean()),
            "min_week": str(part["week_start"].min()),
            "max_week": str(part["week_start"].max()),
        }

    return {
        "rows": int(len(df)),
        "prev": float(df["target"].mean()) if len(df) else None,
        "features": int(len(features)),
        "splits": {
            "train": split_stats(train_mask),
            "valid": split_stats(valid_mask),
            "test": split_stats(test_mask),
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--out", default="model-studio/public/model_data.json")
    parser.add_argument("--print-manifest", action="store_true")
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args()

    print("Loading Panel...")
    panel = pd.read_parquet(args.panel)
    
    # Strict Mask
    mask = (panel['ineligible'] == 0) & (panel['not_censored'] == 1) & (panel['active_player'] == 1)
    if 'label_known' in panel.columns: mask &= (panel['label_known'] == 1)
    
    df = panel[mask].copy()
    
    # Feature eng setup
    if 'position' in df.columns:
        df = pd.get_dummies(df, columns=['position'], prefix='pos')
        pos_dummies = [c for c in df.columns if c.startswith('pos_')]
    else:
        pos_dummies = []
        
    candidates = [
        'minutes_last_1w', 'minutes_last_2w', 'minutes_last_4w', 'minutes_last_8w', 'acwr',
        'matches_last_3d', 'matches_last_7d', 'matches_last_14d', 'matches_last_28d',
        'minutes_last_10d', 'max_minutes_14d',
        'away_matches_last_28d', 'away_minutes_last_28d', 'away_match_share_28d',
        'yellow_cards_last_28d', 'red_cards_last_56d',
        'days_since_transfer', 'log_market_value', 'market_value_trend',
        'injuries_last_30d', 'injuries_last_180d', 'injuries_last_365d',
        'days_out_last_30d', 'days_out_last_180d', 'days_out_last_365d',
        'days_since_last_injury_start', 'days_since_last_injury_end', 'last_injury_days_out',
        'age', 'height'
    ]
    features = [c for c in candidates if c in df.columns] + pos_dummies
    
    # Split
    train_mask = df['week_start'] < '2023-06-01'
    valid_mask = (df['week_start'] >= '2023-06-01') & (df['week_start'] < '2024-01-01')
    test_mask = df['week_start'] >= '2024-01-01'

    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, 'target']
    X_valid = df.loc[valid_mask, features].fillna(0)
    y_valid = df.loc[valid_mask, 'target']
    X_test = df.loc[test_mask, features].fillna(0)
    y_test = df.loc[test_mask, 'target']

    eval_manifest = _manifest(df, features, train_mask, valid_mask, test_mask)
    if args.print_manifest:
        print("Eval manifest:", json.dumps(eval_manifest, indent=2))

    # Train
    print("Training...")
    neg, pos = np.bincount(y_train)
    w_train = np.ones(len(y_train))
    w_train[y_train == 1] = neg / pos
    
    hgb = HistGradientBoostingClassifier(
        loss='log_loss', learning_rate=0.05, max_iter=500, max_depth=6,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=42, verbose=0
    )
    hgb.fit(X_train, y_train, sample_weight=w_train)
    
    # Calibrate
    print("Calibrating...")
    val_probs = hgb.predict_proba(X_valid)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(val_probs, y_valid)
    
    # Eval
    print("Evaluating...")
    test_probs = iso.transform(hgb.predict_proba(X_test)[:, 1])
    
    # Metrics
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    pr_auc_trap = auc(rec, prec)
    ap = average_precision_score(y_test, test_probs)
    roc_auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    
    k = max(1, int(len(y_test) * 0.05))
    top_k_idx = np.argsort(test_probs)[-k:]
    hits = y_test.iloc[top_k_idx].sum()
    top5_prec = hits / k
    recall_at_top5 = hits / max(1, y_test.sum())
    
    metrics = {
        "average_precision": ap,
        "pr_auc_trap": pr_auc_trap,
        "roc_auc": roc_auc,
        "brier": brier,
        "precision_top5": top5_prec,
        "recall_top5": recall_at_top5,
        "test_prevalence": y_test.mean()
    }
    print("Metrics:", metrics)
    
    # Importance
    print("Importance...")
    X_val_s = X_valid.sample(5000, random_state=42)
    y_val_s = y_valid.loc[X_val_s.index]
    r = permutation_importance(hgb, X_val_s, y_val_s, n_repeats=5, random_state=42, scoring='average_precision')
    
    imp_df = pd.DataFrame(
        {
            "feature": features,
            "importance": r.importances_mean,
            "importance_std": r.importances_std,
            "n_repeats": 5,
            "scoring": "average_precision",
        }
    ).sort_values("importance", ascending=False)

    imp_top = imp_df.head(30).copy()
    imp_top10 = imp_df.head(10).copy()

    # Reports artifacts (CSV/PNG/JSON)
    out_dirs = ensure_reports_dirs(args.reports_root)
    out_manifest = out_dirs["manifest"]
    out_figures = out_dirs["figures"]

    pd.DataFrame({"feature": features}).to_csv(out_manifest / "model_features.csv", index=False)
    imp_df.to_csv(out_manifest / "feature_importance_validation.csv", index=False)
    write_json(out_manifest / "eval_manifest.json", {"manifest": eval_manifest, "metrics": metrics})

    if not imp_top.empty:
        plt.figure(figsize=(10, 8))
        imp_plot = imp_top.sort_values("importance")
        plt.barh(imp_plot["feature"], imp_plot["importance"])
        plt.title("Top 30 Permutation Importance (Validation)")
        plt.tight_layout()
        plt.savefig(out_figures / "feature_importance_top30.png", dpi=160)
        plt.close()
        
    # Output Data
    data = {
        "metrics": metrics,
        "manifest": eval_manifest,
        "feature_importance": imp_top10[["feature", "importance"]].to_dict(orient="records"),
        "calibration_curve": [] # Placeholder, or compute histogram
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {out_path}")
    print(f"Wrote {out_manifest / 'eval_manifest.json'}")
    print(f"Wrote {out_manifest / 'feature_importance_validation.csv'}")

if __name__ == "__main__":
    main()
