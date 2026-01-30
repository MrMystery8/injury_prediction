from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

_mpl_dir = Path("reports") / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib
matplotlib.use("Agg")  # headless/sandbox-safe
import matplotlib.pyplot as plt
import pandas as pd

from src.observability.paths import ensure_reports_dirs


def _pick_minutes_feature(cols: list[str]) -> str | None:
    for c in ("minutes_last_4w", "minutes_last_8w", "minutes_last_2w", "minutes_last_1w"):
        if c in cols:
            return c
    return None


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _render_timeline_png(
    player_panel: pd.DataFrame,
    injuries: pd.DataFrame,
    minutes_col: str | None,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))

    x = player_panel["week_start"]
    if minutes_col and minutes_col in player_panel.columns:
        y = player_panel[minutes_col]
        ax.plot(x, y, color="tab:blue", linewidth=1.5, label=minutes_col)

    if "target" in player_panel.columns and minutes_col and minutes_col in player_panel.columns:
        tmask = player_panel["target"] == 1
        if tmask.any():
            ax.scatter(
                player_panel.loc[tmask, "week_start"],
                player_panel.loc[tmask, minutes_col],
                s=18,
                color="tab:red",
                label="target==1",
                zorder=3,
            )

    if not injuries.empty:
        for _, r in injuries.iterrows():
            start = r.get("injury_start")
            end = r.get("injury_end")
            if pd.notna(start) and pd.notna(end):
                ax.axvspan(start, end, alpha=0.15, color="tab:red")
            if pd.notna(start):
                ax.axvline(start, alpha=0.3, color="tab:red", linewidth=1)

    ax.set_title("Player Timeline (Workload + Injury Windows)")
    ax.set_xlabel("week_start")
    ax.set_ylabel("minutes")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _embed_png_base64(png_path: Path) -> str:
    b = png_path.read_bytes()
    return base64.b64encode(b).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-id", required=True, help="Player identifier (int or string).")
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--interim", default="data/interim")
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args()

    out_dirs = ensure_reports_dirs(args.reports_root)
    out_samples = out_dirs["samples"]
    out_html = out_dirs["html"]
    out_fig = out_dirs["figures"]

    panel = pd.read_parquet(args.panel)
    if "week_start" in panel.columns:
        panel["week_start"] = _to_dt(panel["week_start"])

    pid_raw = args.player_id
    pid: object = pid_raw
    if str(pid_raw).isdigit():
        try:
            pid = int(pid_raw)
        except Exception:
            pid = pid_raw

    player_panel = panel[panel["player_id"] == pid].copy()
    if player_panel.empty:
        player_panel = panel[panel["player_id"].astype(str) == str(pid_raw)].copy()
    if player_panel.empty:
        raise SystemExit(f"No panel rows found for player_id={pid_raw}")
    player_panel = player_panel.sort_values("week_start")

    interim = Path(args.interim)
    apps = pd.read_parquet(interim / "backbone_appearances.parquet")
    games = pd.read_parquet(interim / "backbone_games.parquet")
    injuries = pd.read_parquet(interim / "labels_injuries.parquet")

    if "game_date" in games.columns:
        games["game_date"] = _to_dt(games["game_date"])
    if "date" in apps.columns:
        apps["date"] = _to_dt(apps["date"])

    player_matches = apps[apps["player_id"] == pid].copy()
    if player_matches.empty:
        player_matches = apps[apps["player_id"].astype(str) == str(pid_raw)].copy()
    if "game_id" in player_matches.columns and "game_id" in games.columns:
        player_matches = player_matches.merge(
            games[[c for c in ["game_id", "game_date", "competition_id", "season"] if c in games.columns]],
            on="game_id",
            how="left",
        )
    if "game_date" in player_matches.columns:
        player_matches = player_matches.sort_values("game_date")

    player_inj = injuries[injuries["player_id"] == pid].copy()
    if player_inj.empty:
        player_inj = injuries[injuries["player_id"].astype(str) == str(pid_raw)].copy()
    for c in ("injury_start", "injury_end"):
        if c in player_inj.columns:
            player_inj[c] = _to_dt(player_inj[c])
    if "injury_start" in player_inj.columns:
        player_inj = player_inj.sort_values("injury_start")

    out_samples.mkdir(parents=True, exist_ok=True)
    safe_id = str(pid_raw).replace("/", "_")
    player_panel.to_csv(out_samples / f"player_{safe_id}_panel_rows.csv", index=False)
    player_matches.to_csv(out_samples / f"player_{safe_id}_matches.csv", index=False)
    player_inj.to_csv(out_samples / f"player_{safe_id}_injuries.csv", index=False)

    minutes_col = _pick_minutes_feature(list(player_panel.columns))
    out_png = out_fig / f"player_{safe_id}_timeline.png"
    _render_timeline_png(player_panel, player_inj, minutes_col, out_png)

    png_b64 = _embed_png_base64(out_png)
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Player Trace</title>",
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px 8px;font-size:12px} th{background:#f6f6f6} img{max-width:100%}</style>",
        "</head><body>",
        f"<h1>Player Trace: {args.player_id}</h1>",
        f"<p><b>Panel rows</b>: {len(player_panel):,} | <b>Matches</b>: {len(player_matches):,} | <b>Injuries</b>: {len(player_inj):,}</p>",
        f"<img src='data:image/png;base64,{png_b64}' />",
        "<h2>Panel rows (tail 60)</h2>",
        player_panel.tail(60).to_html(index=False, escape=False),
        "<h2>Matches (tail 60)</h2>",
        player_matches.tail(60).to_html(index=False, escape=False),
        "<h2>Injuries</h2>",
        (player_inj.to_html(index=False, escape=False) if not player_inj.empty else "<p>No injuries found.</p>"),
        "</body></html>",
    ]

    out_path = out_html / f"player_{safe_id}_timeline.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_samples / f'player_{safe_id}_panel_rows.csv'}")


if __name__ == "__main__":
    main()
