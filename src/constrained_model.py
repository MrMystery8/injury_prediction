from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, brier_score_loss, roc_auc_score


# Constrained feature groups (assignment):
# - age
# - player position (one-hot)
# - minutes played (workload proxy)
# - days since last played (rest proxy)
# - injury history
BASE_FEATURES: list[str] = [
    "minutes_last_4w",
    "days_since_last_played",
    "injuries_last_365d",
    "days_out_last_365d",
    "days_since_last_injury_start",
    "days_since_last_injury_end",
    "last_injury_days_out",
    "age",
]


def strict_mask(panel: pd.DataFrame) -> pd.Series:
    mask = (panel["ineligible"] == 0) & (panel["not_censored"] == 1) & (panel["active_player"] == 1)
    if "label_known" in panel.columns:
        mask &= panel["label_known"] == 1
    return mask


def split_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = df["week_start"] < "2023-06-01"
    valid_mask = (df["week_start"] >= "2023-06-01") & (df["week_start"] < "2024-01-01")
    test_mask = df["week_start"] >= "2024-01-01"
    return train_mask, valid_mask, test_mask


def add_position_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "position" not in df.columns:
        return df, []
    out = pd.get_dummies(df, columns=["position"], prefix="pos")
    pos_dummies = [c for c in out.columns if c.startswith("pos_")]
    return out, pos_dummies


def constrained_feature_cols(df_after_dummies: pd.DataFrame, pos_dummies: list[str]) -> list[str]:
    features = [c for c in BASE_FEATURES if c in df_after_dummies.columns]
    features += pos_dummies
    return features


def align_feature_frame(df_after_dummies: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df_after_dummies.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0
    return out.loc[:, feature_cols].fillna(0)


@dataclass(frozen=True)
class FitResult:
    model: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    feature_cols: list[str]
    metrics: dict[str, float]


def fit_hgb_isotonic(
    df_strict: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int = 42,
) -> FitResult:
    train_mask, valid_mask, test_mask = split_masks(df_strict)

    X_train = align_feature_frame(df_strict.loc[train_mask], feature_cols)
    y_train = df_strict.loc[train_mask, "target"].to_numpy()
    X_valid = align_feature_frame(df_strict.loc[valid_mask], feature_cols)
    y_valid = df_strict.loc[valid_mask, "target"].to_numpy()
    X_test = align_feature_frame(df_strict.loc[test_mask], feature_cols)
    y_test = df_strict.loc[test_mask, "target"].to_numpy()

    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError("No positives in train or test after masking/splitting.")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    w_train = np.ones(len(y_train), dtype=float)
    w_train[y_train == 1] = neg / max(1, pos)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=500,
        max_depth=6,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    val_probs = model.predict_proba(X_valid)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(val_probs, y_valid)

    test_probs = iso.transform(model.predict_proba(X_test)[:, 1])
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    pr_auc_trap = float(auc(rec, prec))

    k = max(1, int(len(y_test) * 0.05))
    top_k_idx = np.argsort(test_probs)[-k:]
    hits = float(y_test[top_k_idx].sum())
    precision_top5 = float(hits / k)
    recall_top5 = float(hits / max(1, y_test.sum()))

    metrics = {
        "average_precision": float(average_precision_score(y_test, test_probs)),
        "pr_auc_trap": pr_auc_trap,
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "brier": float(brier_score_loss(y_test, test_probs)),
        "precision_top5": precision_top5,
        "recall_top5": recall_top5,
        "test_prevalence": float(y_test.mean()),
    }

    return FitResult(model=model, calibrator=iso, feature_cols=list(feature_cols), metrics=metrics)


def predict_risk(df: pd.DataFrame, fit: FitResult) -> np.ndarray:
    df_d, _ = add_position_dummies(df)
    X = align_feature_frame(df_d, fit.feature_cols)
    return fit.calibrator.transform(fit.model.predict_proba(X)[:, 1])
