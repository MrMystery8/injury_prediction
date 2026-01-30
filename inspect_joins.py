from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.observability.paths import ensure_reports_dirs


def _uniq_keys(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns:
        raise KeyError(f"Missing join key '{key}' in columns: {list(df.columns)[:30]}...")
    return df[[key]].dropna().drop_duplicates()


def audit_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    name: str,
    reports_root: Path,
    max_unmatched: int = 2000,
) -> Dict[str, Any]:
    out_dirs = ensure_reports_dirs(reports_root)
    out_samples = out_dirs["samples"]

    left_k = _uniq_keys(left, key)
    right_k = _uniq_keys(right, key)

    l2r = left_k.merge(right_k, on=key, how="left", indicator=True)
    r2l = right_k.merge(left_k, on=key, how="left", indicator=True)

    l_stats = l2r["_merge"].value_counts(dropna=False).to_dict()
    r_stats = r2l["_merge"].value_counts(dropna=False).to_dict()

    l2r[l2r["_merge"] == "left_only"].head(max_unmatched).to_csv(
        out_samples / f"unmatched_{name}_left.csv", index=False
    )
    r2l[r2l["_merge"] == "left_only"].head(max_unmatched).to_csv(
        out_samples / f"unmatched_{name}_right.csv", index=False
    )

    return {
        "join": name,
        "key": key,
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "left_unique": int(left_k[key].nunique()),
        "right_unique": int(right_k[key].nunique()),
        "both_left": int(l_stats.get("both", 0)),
        "left_only": int(l_stats.get("left_only", 0)),
        "coverage_left": float(l_stats.get("both", 0) / max(1, len(left_k))),
        "both_right": int(r_stats.get("both", 0)),
        "right_only": int(r_stats.get("left_only", 0)),
        "coverage_right": float(r_stats.get("both", 0) / max(1, len(right_k))),
    }


def _load_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim", default="data/interim", help="Directory with interim parquet tables.")
    parser.add_argument("--reports-root", default="reports", help="Reports root.")
    parser.add_argument("--key", default="player_id", help="Join key (default: player_id).")
    args = parser.parse_args()

    interim = Path(args.interim)
    reports_root = Path(args.reports_root)

    players = pd.read_parquet(interim / "backbone_players.parquet")
    injuries = pd.read_parquet(interim / "labels_injuries.parquet")
    profiles = _load_optional(interim / "labels_profiles.parquet")
    transfers = _load_optional(interim / "labels_transfers.parquet")
    market_values = _load_optional(interim / "labels_market_values.parquet")

    rows: List[Dict[str, Any]] = []
    rows.append(audit_merge(players, injuries, args.key, "players_vs_injuries", reports_root))
    if profiles is not None:
        rows.append(audit_merge(players, profiles, args.key, "players_vs_profiles", reports_root))
    if transfers is not None:
        rows.append(audit_merge(players, transfers, args.key, "players_vs_transfers", reports_root))
    if market_values is not None:
        rows.append(audit_merge(players, market_values, args.key, "players_vs_market_values", reports_root))

    out_dirs = ensure_reports_dirs(reports_root)
    pd.DataFrame(rows).to_csv(out_dirs["manifest"] / "join_stats.csv", index=False)

    # Required baseline samples (backbone vs label injuries)
    players_k = _uniq_keys(players, args.key)
    injuries_k = _uniq_keys(injuries, args.key)
    bb_unmatched = players_k.merge(injuries_k, on=args.key, how="left", indicator=True)
    bb_unmatched = bb_unmatched[bb_unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    bb_unmatched.head(2000).to_csv(out_dirs["samples"] / "unmatched_backbone_players.csv", index=False)

    lbl_unmatched = injuries_k.merge(players_k, on=args.key, how="left", indicator=True)
    lbl_unmatched = lbl_unmatched[lbl_unmatched["_merge"] == "left_only"].drop(columns=["_merge"])
    lbl_unmatched.head(2000).to_csv(out_dirs["samples"] / "unmatched_label_players.csv", index=False)

    # matched examples: players with injury record
    common = _uniq_keys(players, args.key).merge(_uniq_keys(injuries, args.key), on=args.key, how="inner")
    examples = common.sample(min(500, len(common)), random_state=42) if len(common) else common
    examples.to_csv(out_dirs["samples"] / "matched_examples.csv", index=False)

    print(f"Wrote {out_dirs['manifest'] / 'join_stats.csv'}")
    print(f"Wrote {out_dirs['samples'] / 'matched_examples.csv'}")
    print(f"Wrote {out_dirs['samples'] / 'unmatched_backbone_players.csv'}")
    print(f"Wrote {out_dirs['samples'] / 'unmatched_label_players.csv'}")


if __name__ == "__main__":
    main()
