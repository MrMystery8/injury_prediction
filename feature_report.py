from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Set

_mpl_dir = Path("reports") / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib
matplotlib.use("Agg")  # headless/sandbox-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.observability.paths import ensure_reports_dirs


EXCLUDE: Set[str] = {
    "player_id",
    "game_id",
    "team_id",
    "club_id",
    "competition_id",
    "season",
    "week_start",
    "target",
    "ineligible",
    "label_known",
    "not_censored",
    "active_player",
}


def _numeric_stats(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def _read_feature_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return set()
    return set(df["feature"].astype(str).tolist())


def _plot_barh(out_path: Path, title: str, x: Iterable[float], y: Iterable[str]) -> None:
    plt.figure(figsize=(10, 8))
    plt.barh(list(y), list(x))
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args()

    out_dirs = ensure_reports_dirs(args.reports_root)
    out_m = out_dirs["manifest"]
    out_f = out_dirs["figures"]

    df = pd.read_parquet(args.panel)
    model_features = _read_feature_list(out_m / "model_features.csv")

    rows = []
    for c in df.columns:
        if c in EXCLUDE:
            continue

        rec = {
            "feature": c,
            "dtype": str(df[c].dtype),
            "missing_pct": float(df[c].isna().mean()),
            "n_unique": int(df[c].nunique(dropna=True)),
            "is_model_feature": int(c in model_features) if model_features else np.nan,
        }
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            rec.update(_numeric_stats(df[c]))
        else:
            rec.update({"mean": np.nan, "std": np.nan, "min": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan})
        rows.append(rec)

    cat_df = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    cat_df.to_csv(out_m / "feature_catalog.csv", index=False)

    top_miss = cat_df.head(30).sort_values("missing_pct")
    _plot_barh(out_f / "feature_missingness_top30.png", "Top 30 Feature Missingness", top_miss["missing_pct"], top_miss["feature"])

    imp_path = out_m / "feature_importance_validation.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        if {"feature", "importance"}.issubset(imp_df.columns) and not imp_df.empty:
            top_imp = imp_df.sort_values("importance", ascending=False).head(30).sort_values("importance")
            _plot_barh(
                out_f / "feature_importance_top30.png",
                "Top 30 Permutation Importance (Validation)",
                top_imp["importance"],
                top_imp["feature"],
            )

    print(f"Wrote {out_m / 'feature_catalog.csv'}")
    print(f"Wrote {out_f / 'feature_missingness_top30.png'}")


if __name__ == "__main__":
    main()
