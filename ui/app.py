from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from allowlist import COMMANDS

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None

try:
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None


st.set_page_config(page_title="Injury Risk Studio", layout="wide")

ROOT = Path(".")
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "manifest"
SAMPLES = REPORTS / "samples"
FIGURES = REPORTS / "figures"
HTML = REPORTS / "html"
PANEL = ROOT / "data" / "processed" / "panel.parquet"


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def _duckdb_conn():
    if duckdb is None:
        return None
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")
    return con


def _panel_exists_or_stop() -> None:
    if not PANEL.exists():
        st.warning(f"Missing `{PANEL}`. Run `make_panel.py` first.")
        st.stop()


def _artifact_status() -> pd.DataFrame:
    files = [
        MANIFEST / "panel_build_manifest.json",
        MANIFEST / "join_stats.csv",
        MANIFEST / "panel_columns.csv",
        MANIFEST / "missingness.csv",
        MANIFEST / "feature_catalog.csv",
        MANIFEST / "model_features.csv",
        MANIFEST / "eval_manifest.json",
        MANIFEST / "audit_results.json",
        MANIFEST / "feature_importance_validation.csv",
        SAMPLES / "panel_sample_50k.parquet",
    ]
    rows = []
    for f in files:
        rows.append(
            {
                "file": str(f),
                "exists": bool(f.exists()),
                "modified": _fmt_ts(f.stat().st_mtime) if f.exists() else "",
                "size_kb": round(f.stat().st_size / 1024, 1) if f.exists() else None,
            }
        )
    return pd.DataFrame(rows)


def _run_cmd(argv: list[str], tail_chars: int = 15_000) -> None:
    st.write("Running:", " ".join(argv))
    out_box = st.empty()
    err_box = st.empty()

    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_buf = ""
    stderr_buf = ""

    # Read incrementally; update UI with a running tail.
    while True:
        out = proc.stdout.readline() if proc.stdout else ""
        err = proc.stderr.readline() if proc.stderr else ""

        if out:
            stdout_buf += out
            stdout_buf = stdout_buf[-tail_chars:]
            out_box.code(stdout_buf, language="text")
        if err:
            stderr_buf += err
            stderr_buf = stderr_buf[-tail_chars:]
            err_box.code(stderr_buf, language="text")

        if (not out) and (not err) and proc.poll() is not None:
            break

    rc = proc.wait()
    if rc == 0:
        st.success(f"Done (exit {rc})")
    else:
        st.error(f"Failed (exit {rc})")


def _trace_player_cmd(player_id: str) -> list[str] | None:
    # Keep tight: only digits to avoid surprise IDs and to keep filenames stable.
    if not re.fullmatch(r"\d{1,18}", player_id.strip()):
        return None
    return [sys.executable, "trace_player.py", "--player-id", player_id.strip()]


def _duckdb_query(where_sql: str, limit: int = 1000, cols: Iterable[str] | None = None) -> pd.DataFrame:
    con = _duckdb_conn()
    if con is None:
        st.error("Missing dependency: `duckdb`. Install `duckdb` to use the Panel Explorer.")
        st.stop()

    _panel_exists_or_stop()
    select_cols = "*" if cols is None else ", ".join([f'"{c}"' for c in cols])
    q = f"SELECT {select_cols} FROM read_parquet('{PANEL.as_posix()}') {where_sql} LIMIT {int(limit)}"
    return con.execute(q).df()


def _duckdb_count(where_sql: str) -> int:
    con = _duckdb_conn()
    if con is None:
        return 0
    if not PANEL.exists():
        return 0
    q = f"SELECT COUNT(*) AS n FROM read_parquet('{PANEL.as_posix()}') {where_sql}"
    return int(con.execute(q).df().iloc[0]["n"])


def _duckdb_group(where_sql: str, group_col: str, limit: int = 200) -> pd.DataFrame:
    con = _duckdb_conn()
    if con is None:
        st.error("Missing dependency: `duckdb`. Install `duckdb` to use aggregations.")
        st.stop()

    _panel_exists_or_stop()
    gc = f'"{group_col}"'
    q = f"""
    SELECT
      {gc} AS group_key,
      COUNT(*) AS n,
      AVG(target) AS prevalence
    FROM read_parquet('{PANEL.as_posix()}')
    {where_sql}
    GROUP BY {gc}
    ORDER BY n DESC
    LIMIT {int(limit)}
    """
    return con.execute(q).df()


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if px is None:
        st.info("Install `plotly` for interactive charts.")
        return
    fig = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)


def _plot_barh(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if px is None:
        st.info("Install `plotly` for interactive charts.")
        return
    fig = px.bar(df, x=x, y=y, orientation="h", title=title)
    st.plotly_chart(fig, use_container_width=True)


st.title("Player-Week Injury Risk — Observability Studio")

tabs = st.tabs(["Pipeline Runner", "Panel Explorer", "Joins", "Features", "Eval", "Player Trace", "Artifacts"])


with tabs[0]:
    st.subheader("One-click pipeline steps (allowlisted)")

    cols = st.columns(4)
    keys = list(COMMANDS.keys())
    tail = st.slider("stdout/stderr tail (chars)", 2000, 50000, 15000, 1000)
    for i, k in enumerate(keys):
        with cols[i % 4]:
            if st.button(COMMANDS[k].label, use_container_width=True):
                _run_cmd(COMMANDS[k].argv, tail_chars=int(tail))

    st.divider()
    st.subheader("Artifact status")
    st.dataframe(_artifact_status(), use_container_width=True)


with tabs[1]:
    st.subheader("Panel Explorer (DuckDB, no full load)")
    if duckdb is None:
        st.error("Missing dependency: `duckdb`. Install `duckdb` to use this tab.")
        st.stop()

    sample = _read_parquet(SAMPLES / "panel_sample_50k.parquet")
    if sample.empty:
        st.warning("Missing panel sample. Run `inspect_panel.py` first.")
        st.stop()

    # Sidebar filters are derived from the sample, but queries are run against the full parquet.
    with st.sidebar:
        st.header("Filters")

        # Date
        if "week_start" in sample.columns:
            w = pd.to_datetime(sample["week_start"], errors="coerce")
            w0, w1 = w.min(), w.max()
            dr = st.date_input("week_start range", (w0.date(), w1.date()))
        else:
            dr = None

        def _tri(col: str) -> str:
            return st.selectbox(col, ["all", "1", "0"], index=0)

        target_opt = _tri("target") if "target" in sample.columns else "all"
        ineligible_opt = _tri("ineligible") if "ineligible" in sample.columns else "all"
        active_opt = _tri("active_player") if "active_player" in sample.columns else "all"
        notc_opt = _tri("not_censored") if "not_censored" in sample.columns else "all"
        label_opt = _tri("label_known") if "label_known" in sample.columns else "all"

        # League-ish
        league_col = None
        for c in ("competition_code", "competition_id"):
            if c in sample.columns:
                league_col = c
                break

        if league_col is not None:
            leagues = sorted(sample[league_col].dropna().astype(str).unique().tolist())
            league_pick = st.multiselect(league_col, leagues, default=[])
        else:
            league_pick = []

        # Position
        pos_col = "position" if "position" in sample.columns else None
        if pos_col is not None:
            positions = sorted(sample[pos_col].dropna().astype(str).unique().tolist())
            pos_pick = st.multiselect("position", positions, default=[])
        else:
            pos_pick = []

        limit = st.slider("Row preview limit", 100, 5000, 1000, 100)

    where = []
    if dr and len(dr) == 2 and "week_start" in sample.columns:
        a, b = dr
        where.append(f"week_start >= DATE '{a}' AND week_start <= DATE '{b}'")
    if target_opt != "all":
        where.append(f"target = {int(target_opt)}")
    if ineligible_opt != "all":
        where.append(f"ineligible = {int(ineligible_opt)}")
    if active_opt != "all":
        where.append(f"active_player = {int(active_opt)}")
    if notc_opt != "all":
        where.append(f"not_censored = {int(notc_opt)}")
    if label_opt != "all" and "label_known" in sample.columns:
        where.append(f"label_known = {int(label_opt)}")
    if league_col is not None and league_pick:
        allowed = ", ".join([f"'{x}'" for x in league_pick])
        where.append(f"CAST({league_col} AS VARCHAR) IN ({allowed})")
    if pos_col is not None and pos_pick:
        allowed = ", ".join([f"'{x}'" for x in pos_pick])
        where.append(f"CAST(position AS VARCHAR) IN ({allowed})")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    st.caption(where_sql if where_sql else "(no filters)")

    c0, c1, c2 = st.columns(3)
    c0.metric("Filtered rows", f"{_duckdb_count(where_sql):,}")
    c1.metric("Panel file", "present" if PANEL.exists() else "missing")
    strict_mask = "WHERE ineligible=0 AND not_censored=1 AND active_player=1" + (" AND label_known=1" if "label_known" in sample.columns else "")
    c2.metric("Strict rows", f"{_duckdb_count(strict_mask):,}")

    preview = _duckdb_query(where_sql, limit=int(limit))
    st.dataframe(preview, use_container_width=True, height=420)

    st.divider()
    st.subheader("Aggregations")
    candidates = [c for c in ("season", "competition_code", "competition_id", "position") if c in sample.columns]
    group_col = st.selectbox("Group by", candidates or list(sample.columns), index=0)
    agg = _duckdb_group(where_sql, group_col)
    st.dataframe(agg, use_container_width=True, height=380)
    _plot_bar(agg.head(30), x="group_key", y="prevalence", title=f"Prevalence by {group_col} (top 30 by n)")

    st.divider()
    st.subheader("Download filtered preview")
    max_dl = st.number_input("Max rows to export", min_value=1000, max_value=200000, value=20000, step=1000)
    dl_df = _duckdb_query(where_sql, limit=int(max_dl))
    st.download_button("Download CSV", dl_df.to_csv(index=False).encode("utf-8"), file_name="panel_filtered.csv")


with tabs[2]:
    st.subheader("Join Explorer")
    join_stats = _read_csv(MANIFEST / "join_stats.csv")
    if join_stats.empty:
        st.warning("Missing `reports/manifest/join_stats.csv`. Run `inspect_joins.py`.")
    else:
        st.dataframe(join_stats, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ub = _read_csv(SAMPLES / "unmatched_backbone_players.csv")
        st.write("Unmatched backbone players (sample)")
        st.dataframe(ub.head(500), use_container_width=True, height=420)
    with c2:
        ul = _read_csv(SAMPLES / "unmatched_label_players.csv")
        st.write("Unmatched label players (sample)")
        st.dataframe(ul.head(500), use_container_width=True, height=420)


with tabs[3]:
    st.subheader("Feature Explorer")
    feat_cat = _read_csv(MANIFEST / "feature_catalog.csv")
    miss = _read_csv(MANIFEST / "missingness.csv")
    feats = _read_csv(MANIFEST / "model_features.csv")
    imp = _read_csv(MANIFEST / "feature_importance_validation.csv")

    c0, c1 = st.columns(2)
    with c0:
        st.write("Model features (definitive)")
        st.dataframe(feats, use_container_width=True, height=260)
        st.write("Missingness")
        st.dataframe(miss.head(300), use_container_width=True, height=360)
    with c1:
        st.write("Feature catalog")
        st.dataframe(feat_cat.head(300), use_container_width=True, height=360)
        st.write("Importance (validation)")
        st.dataframe(imp.head(100), use_container_width=True, height=260)

    if not imp.empty:
        imp_col = "importance" if "importance" in imp.columns else "importance_mean" if "importance_mean" in imp.columns else None
        if imp_col:
            top = imp.sort_values(imp_col, ascending=False).head(30).copy()
            top = top.sort_values(imp_col, ascending=True)
            _plot_barh(top, x=imp_col, y="feature", title="Top 30 Permutation Importance (Validation)")


with tabs[4]:
    st.subheader("Model Evaluation")
    eval_m = _read_json(MANIFEST / "eval_manifest.json")
    audit = _read_json(MANIFEST / "audit_results.json")

    c0, c1 = st.columns(2)
    with c0:
        st.write("Eval manifest + metrics")
        st.json(eval_m)
    with c1:
        st.write("Audit results")
        st.json(audit)

    if isinstance(eval_m.get("metrics"), dict):
        m = eval_m["metrics"]
        cols = st.columns(5)
        cols[0].metric("AP", f"{m.get('average_precision', float('nan')):.4f}" if m.get("average_precision") is not None else "")
        cols[1].metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.4f}" if m.get("roc_auc") is not None else "")
        cols[2].metric("Brier", f"{m.get('brier', float('nan')):.4f}" if m.get("brier") is not None else "")
        cols[3].metric("Prec@Top5%", f"{m.get('precision_top5', float('nan')):.4f}" if m.get("precision_top5") is not None else "")
        cols[4].metric("Rec@Top5%", f"{m.get('recall_top5', float('nan')):.4f}" if m.get("recall_top5") is not None else "")


with tabs[5]:
    st.subheader("Player Trace")
    st.caption("Digits-only player_id (safety). Runs `trace_player.py` and shows the generated outputs.")

    player_id = st.text_input("player_id", "")
    cmd = _trace_player_cmd(player_id) if player_id.strip() else None

    c0, c1 = st.columns([1, 2])
    with c0:
        if st.button("Generate trace", use_container_width=True, disabled=(cmd is None)):
            if cmd is None:
                st.error("Invalid player_id. Use digits only.")
            else:
                _run_cmd(cmd, tail_chars=15000)

    with c1:
        if player_id.strip():
            html_path = HTML / f"player_{player_id.strip()}_timeline.html"
            png_path = FIGURES / f"player_{player_id.strip()}_timeline.png"
            if png_path.exists():
                st.image(str(png_path), use_container_width=True)
            if html_path.exists():
                st.write("HTML trace:", f"`{html_path}`")

            rows_path = SAMPLES / f"player_{player_id.strip()}_panel_rows.csv"
            if rows_path.exists():
                st.write("Panel rows (tail)")
                st.dataframe(_read_csv(rows_path).tail(60), use_container_width=True, height=280)


with tabs[6]:
    st.subheader("Artifacts")
    st.dataframe(_artifact_status(), use_container_width=True)

    st.divider()
    for folder in (MANIFEST, SAMPLES, FIGURES, HTML):
        st.write(f"`{folder}`")
        if folder.exists():
            files = sorted([p.name for p in folder.iterdir() if p.is_file()])
            st.write(files[:250])
        else:
            st.write("(missing)")
