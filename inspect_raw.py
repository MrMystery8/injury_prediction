from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.observability.paths import ensure_reports_dirs
from src.observability.io import write_json


def _count_data_rows(csv_path: Path) -> int | None:
    try:
        n_lines = 0
        with csv_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                n_lines += chunk.count(b"\n")
        # heuristic: header row + data rows; if file ends without newline this may undercount by 1
        return max(0, n_lines - 1)
    except Exception:
        return None


def list_csv_headers(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*.csv")):
        rec: Dict[str, Any] = {"path": str(p)}
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
            rec.update({"n_cols": int(len(cols)), "cols": cols})
        except Exception as e:
            rec.update({"error": str(e), "n_cols": None, "cols": []})

        rec["size_bytes"] = p.stat().st_size if p.exists() else None
        rec["n_rows"] = _count_data_rows(p)
        rows.append(rec)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", help="Root directory to scan for CSVs.")
    parser.add_argument("--reports-root", default="reports", help="Reports root (writes to reports/manifest).")
    args = parser.parse_args()

    data_root = Path(args.root)
    out_dirs = ensure_reports_dirs(args.reports_root)
    out_manifest = out_dirs["manifest"]

    rows = list_csv_headers(data_root)
    df = pd.DataFrame(rows)
    if "cols" in df.columns:
        df["cols"] = df["cols"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    df.to_csv(out_manifest / "raw_tables_overview.csv", index=False)
    write_json(out_manifest / "raw_schema_manifest.json", rows)

    print(f"Wrote {out_manifest / 'raw_tables_overview.csv'}")
    print(f"Wrote {out_manifest / 'raw_schema_manifest.json'}")


if __name__ == "__main__":
    main()

