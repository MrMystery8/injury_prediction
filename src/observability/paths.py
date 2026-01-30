from __future__ import annotations

from pathlib import Path
from typing import Dict


REPORT_SUBDIRS = ("manifest", "samples", "figures", "html")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_reports_dirs(root: Path | str = "reports") -> Dict[str, Path]:
    root_path = Path(root)
    out = {name: root_path / name for name in REPORT_SUBDIRS}
    ensure_dir(root_path)
    for p in out.values():
        ensure_dir(p)
    return out

