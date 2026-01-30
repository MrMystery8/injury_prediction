from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.observability.paths import ensure_reports_dirs


def _safe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--sample-size", type=int, default=50_000)
    args = parser.parse_args()

    out_dirs = ensure_reports_dirs(args.reports_root)
    out_samples = out_dirs["samples"]
    out_manifest = out_dirs["manifest"]

    df = pd.read_parquet(args.panel)

    df.head(200).to_csv(out_samples / "panel_head.csv", index=False)

    pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}).to_csv(
        out_manifest / "panel_columns.csv", index=False
    )

    missingness = pd.DataFrame(
        {
            "column": df.columns,
            "missing_pct": df.isna().mean().values,
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    ).sort_values("missing_pct", ascending=False)
    missingness.to_csv(out_manifest / "missingness.csv", index=False)

    _safe_sample(df, args.sample_size).to_parquet(out_samples / "panel_sample_50k.parquet", index=False)

    if "target" in df.columns and df["target"].notna().any():
        pos = df[df["target"] == 1]
        if not pos.empty:
            _safe_sample(pos, 500).to_csv(out_samples / "panel_positives_500.csv", index=False)

    if "ineligible" in df.columns and df["ineligible"].notna().any():
        inel = df[df["ineligible"] == 1]
        if not inel.empty:
            _safe_sample(inel, 500).to_csv(out_samples / "panel_ineligible_500.csv", index=False)

    print(f"Wrote {out_samples / 'panel_head.csv'}")
    print(f"Wrote {out_samples / 'panel_sample_50k.parquet'}")
    print(f"Wrote {out_manifest / 'panel_columns.csv'}")
    print(f"Wrote {out_manifest / 'missingness.csv'}")


if __name__ == "__main__":
    main()

