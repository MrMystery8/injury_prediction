from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Cmd:
    label: str
    argv: list[str]


PY = sys.executable

COMMANDS: dict[str, Cmd] = {
    "inspect_raw": Cmd("Inspect Raw (schemas)", [PY, "inspect_raw.py"]),
    "make_panel": Cmd("Build Panel", [PY, "make_panel.py"]),
    "verify_panel": Cmd(
        "Verify Panel",
        [PY, "verify_panel.py", "--panel", "data/processed/panel.parquet"],
    ),
    "inspect_joins": Cmd("Inspect Joins", [PY, "inspect_joins.py"]),
    "inspect_panel": Cmd("Inspect Panel", [PY, "inspect_panel.py"]),
    "feature_report": Cmd("Feature Report", [PY, "feature_report.py"]),
    "quick_eval": Cmd(
        "Quick Eval (+manifest)",
        [PY, "quick_eval.py", "--panel", "data/processed/panel.parquet", "--print-manifest"],
    ),
    "audit_eval": Cmd("Audit Eval", [PY, "audit_eval.py", "--panel", "data/processed/panel.parquet"]),
}
