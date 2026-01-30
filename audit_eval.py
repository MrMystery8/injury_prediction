import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score

from src.data.loader import load_labels
from src.data.processing import clean_injuries
from src.observability.io import write_json


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(r, p))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def _split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = df["week_start"] < "2023-06-01"
    valid_mask = (df["week_start"] >= "2023-06-01") & (df["week_start"] < "2024-01-01")
    test_mask = df["week_start"] >= "2024-01-01"
    return train_mask, valid_mask, test_mask


def _strict_mask(panel: pd.DataFrame) -> pd.DataFrame:
    mask = (panel["ineligible"] == 0) & (panel["not_censored"] == 1) & (panel["active_player"] == 1)
    if "label_known" in panel.columns:
        mask &= panel["label_known"] == 1
    return panel.loc[mask].copy()


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "position" in df.columns:
        df = pd.get_dummies(df, columns=["position"], prefix="pos")
        pos_dummies = [c for c in df.columns if c.startswith("pos_")]
    else:
        pos_dummies = []

    candidates = [
        # Workload
        "minutes_last_1w",
        "minutes_last_2w",
        "minutes_last_4w",
        "minutes_last_8w",
        "acwr",
        # Congestion
        "matches_last_3d",
        "matches_last_7d",
        "matches_last_14d",
        "matches_last_28d",
        "minutes_last_10d",
        "max_minutes_14d",
        # Travel / discipline proxies
        "away_matches_last_28d",
        "away_minutes_last_28d",
        "away_match_share_28d",
        "yellow_cards_last_28d",
        "red_cards_last_56d",
        # Context
        "days_since_transfer",
        "log_market_value",
        "market_value_trend",
        # Injury history
        "injuries_last_30d",
        "injuries_last_180d",
        "injuries_last_365d",
        "days_out_last_30d",
        "days_out_last_180d",
        "days_out_last_365d",
        "days_since_last_injury_start",
        "days_since_last_injury_end",
        "last_injury_days_out",
        # Demographics
        "age",
        "height",
    ]
    features = [c for c in candidates if c in df.columns] + pos_dummies
    return df, features


def _fit_and_score(df: pd.DataFrame, features: list[str], seed: int = 42) -> dict:
    train_mask, valid_mask, test_mask = _split(df)

    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, "target"].to_numpy()
    X_valid = df.loc[valid_mask, features].fillna(0)
    y_valid = df.loc[valid_mask, "target"].to_numpy()
    X_test = df.loc[test_mask, features].fillna(0)
    y_test = df.loc[test_mask, "target"].to_numpy()

    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError("No positives in train or test after masking/splitting.")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    w_train = np.ones(len(y_train), dtype=float)
    w_train[y_train == 1] = neg / max(1, pos)

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=500,
        max_depth=6,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    hgb.fit(X_train, y_train, sample_weight=w_train)

    val_probs = hgb.predict_proba(X_valid)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(val_probs, y_valid)

    test_probs = iso.transform(hgb.predict_proba(X_test)[:, 1])

    return {
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "test_prev": float(y_test.mean()),
        "average_precision": average_precision(y_test, test_probs),
        "pr_auc_trap": pr_auc(y_test, test_probs),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
    }


def _audit_label_horizon(df: pd.DataFrame, injuries: pd.DataFrame, n: int = 200) -> dict:
    positives = df[df["target"] == 1].copy()
    if positives.empty:
        return {"checked": 0, "ok": True, "errors": 0}

    sample = positives.sample(min(n, len(positives)), random_state=42)
    errors = 0
    deltas = []

    for _, row in sample.iterrows():
        pid = row["player_id"]
        t = row["week_start"]
        matches = injuries[
            (injuries["player_id"] == pid)
            & (injuries["days_out"] >= 7)
            & (injuries["injury_start"] > t)
            & (injuries["injury_start"] <= t + pd.Timedelta(days=30))
        ]
        if matches.empty:
            errors += 1
            continue
        delta = int((matches.iloc[0]["injury_start"] - t).days)
        deltas.append(delta)
        if not (1 <= delta <= 30):
            errors += 1

    return {
        "checked": int(len(sample)),
        "ok": errors == 0,
        "errors": int(errors),
        "delta_min": int(min(deltas)) if deltas else None,
        "delta_max": int(max(deltas)) if deltas else None,
        "delta_mean": float(np.mean(deltas)) if deltas else None,
    }


def _audit_injury_history_strict_prior(panel: pd.DataFrame, injuries: pd.DataFrame, n: int = 50) -> dict:
    candidates = panel[panel.get("injuries_last_365d", 0) > 0].copy()
    if candidates.empty:
        return {"checked": 0, "ok": True, "errors": 0}

    sample = candidates.sample(min(n, len(candidates)), random_state=42)
    errors = 0

    for _, row in sample.iterrows():
        pid = row["player_id"]
        t = row["week_start"]

        # recompute strictly prior (start in [t-365d, t))
        w0 = t - pd.Timedelta(days=365)
        subset = injuries[(injuries["player_id"] == pid) & (injuries["days_out"] >= 7)].copy()
        subset = subset.dropna(subset=["injury_start"])
        subset = subset[(subset["injury_start"] >= w0) & (subset["injury_start"] < t)]

        exp_count = int(len(subset))
        got_count = int(row["injuries_last_365d"])
        if exp_count != got_count:
            errors += 1
            continue

        # check strictness explicitly
        if not subset.empty and not (subset["injury_start"] < t).all():
            errors += 1

    return {"checked": int(len(sample)), "ok": errors == 0, "errors": int(errors)}


def _negative_control(df: pd.DataFrame, features: list[str]) -> dict:
    train_mask, valid_mask, test_mask = _split(df)

    X_train = df.loc[train_mask, features].fillna(0)
    y_train = df.loc[train_mask, "target"].to_numpy()
    X_valid = df.loc[valid_mask, features].fillna(0)
    y_valid = df.loc[valid_mask, "target"].to_numpy()
    X_test = df.loc[test_mask, features].fillna(0)
    y_test = df.loc[test_mask, "target"].to_numpy()

    rng = np.random.RandomState(42)
    y_train_shuf = y_train.copy()
    rng.shuffle(y_train_shuf)
    y_valid_shuf = y_valid.copy()
    rng.shuffle(y_valid_shuf)

    neg = int((y_train_shuf == 0).sum())
    pos = int((y_train_shuf == 1).sum())
    w_train = np.ones(len(y_train_shuf), dtype=float)
    w_train[y_train_shuf == 1] = neg / max(1, pos)

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=500,
        max_depth=6,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    hgb.fit(X_train, y_train_shuf, sample_weight=w_train)

    val_probs = hgb.predict_proba(X_valid)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    # fit calibrator on shuffled validation labels (keeps the negative control strict)
    iso.fit(val_probs, y_valid_shuf)

    probs = iso.transform(hgb.predict_proba(X_test)[:, 1])

    # also compute what happens if we shuffle y_test for a pure evaluation sanity check
    y_test_shuf = y_test.copy()
    rng.shuffle(y_test_shuf)
    return {
        "train_shuffled_average_precision": average_precision(y_test, probs),
        "train_shuffled_pr_auc_trap": pr_auc(y_test, probs),
        "test_label_shuffled_average_precision": average_precision(y_test_shuf, probs),
        "test_label_shuffled_pr_auc_trap": pr_auc(y_test_shuf, probs),
        "test_prev": float(y_test.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--labels", default="data/raw/xfkzujqjvx97n:football-datasets")
    parser.add_argument("--out", default="reports/manifest/audit_results.json")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    labels_path = Path(args.labels)
    out_path = Path(args.out)

    panel = pd.read_parquet(panel_path)
    df = _strict_mask(panel)
    df, features = _build_features(df)

    # Load injuries for audits (same loader + cleaner as the pipeline)
    lbl = load_labels(labels_path)
    injuries = clean_injuries(lbl["injuries"])

    manifest = {
        "panel": str(panel_path),
        "rows_after_mask": int(len(df)),
        "features": int(len(features)),
        "splits": {
            "train": int((df["week_start"] < "2023-06-01").sum()),
            "valid": int(((df["week_start"] >= "2023-06-01") & (df["week_start"] < "2024-01-01")).sum()),
            "test": int((df["week_start"] >= "2024-01-01").sum()),
        },
    }

    score = _fit_and_score(df, features)
    horizon = _audit_label_horizon(df[df["week_start"] >= "2024-01-01"], injuries)
    inj_hist = _audit_injury_history_strict_prior(panel, injuries)
    neg_ctl = _negative_control(df, features)

    out = {
        "manifest": manifest,
        "score": score,
        "audit_label_horizon_test": horizon,
        "audit_injury_history_strict_prior": inj_hist,
        "negative_control": neg_ctl,
    }
    print(json.dumps(out, indent=2))
    write_json(out_path, out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
