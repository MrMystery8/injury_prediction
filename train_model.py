import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, brier_score_loss, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from src.constrained_model import BASE_FEATURES

def load_panel(path):
    logger.info(f"Loading Panel from {path}...")
    df = pd.read_parquet(path)
    return df

def bootstrap_metrics(y_true, y_prob, n_boot=50):
    """
    Computes 95% CI for Average Precision (PR-AUC) and Precision@Top5%.
    """
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    
    aps = []
    top5_precs = []
    
    for i in range(n_boot):
        boot_idx = rng.choice(indices, len(indices), replace=True)
        y_t = y_true.iloc[boot_idx].values
        y_p = y_prob[boot_idx]
        
        if y_t.sum() == 0: continue
        
        # Average Precision (PR-AUC)
        aps.append(average_precision_score(y_t, y_p))
        
        # Top 5% Precision
        k = max(1, int(len(y_t) * 0.05))
        top_k_idx = np.argsort(y_p)[-k:]
        hits = y_t[top_k_idx].sum()
        top5_precs.append(hits / k)
        
    stats = {
        'ap_mean': np.mean(aps),
        'ap_lower': np.percentile(aps, 2.5),
        'ap_upper': np.percentile(aps, 97.5),
        'top5_prec_mean': np.mean(top5_precs),
        'top5_prec_lower': np.percentile(top5_precs, 2.5),
        'top5_prec_upper': np.percentile(top5_precs, 97.5)
    }
    return stats

def evaluate_model(y_true, y_prob, name="Model"):
    # Standard Metrics
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Bootstrap
    boot = bootstrap_metrics(y_true, y_prob, n_boot=50)
    
    print(f"\n--- {name} Results ---")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"Brier Score:   {brier:.4f}")
    print(f"Avg Precision: {ap:.4f} (95% CI: {boot['ap_lower']:.4f} - {boot['ap_upper']:.4f})")
    print(f"Prec @ Top 5%: {boot['top5_prec_mean']:.2%} (95% CI: {boot['top5_prec_lower']:.2%} - {boot['top5_prec_upper']:.2%})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=6)
    args = parser.parse_args()
    
    panel = load_panel(args.panel)
    
    # 1. Structural Filters (Strict)
    logger.info("Applying Strict Training Mask...")
    # Mask: eligible & label_known & not_censored & active
    # Note: 'label_known' might be missing if older panel, check first
    mask = (panel['ineligible'] == 0) & (panel['not_censored'] == 1) & (panel['active_player'] == 1)
    if 'label_known' in panel.columns:
        mask = mask & (panel['label_known'] == 1)
    
    modeling_df = panel[mask].copy()
    logger.info(f"Modeling Rows: {len(modeling_df)} ({len(modeling_df)/len(panel):.2%})")
    
    # 2. Features
    # Identify available features
    all_cols = modeling_df.columns
    
    # One-Hot Position
    if 'position' in modeling_df.columns:
        modeling_df = pd.get_dummies(modeling_df, columns=['position'], prefix='pos')
        pos_dummies = [c for c in modeling_df.columns if c.startswith('pos_')]
    else:
        pos_dummies = []
        
    # Constrained feature set (assignment): age, position, minutes played, rest, injury history
    candidates = list(BASE_FEATURES)
    
    features = [c for c in candidates if c in modeling_df.columns] + pos_dummies
    logger.info(f"Features ({len(features)}): {features}")
    
    # 3. Splits
    train_mask = modeling_df['week_start'] < '2023-06-01'
    valid_mask = (modeling_df['week_start'] >= '2023-06-01') & (modeling_df['week_start'] < '2024-01-01')
    test_mask = modeling_df['week_start'] >= '2024-01-01'
    
    X_train = modeling_df.loc[train_mask, features].fillna(0)
    y_train = modeling_df.loc[train_mask, 'target']
    
    X_valid = modeling_df.loc[valid_mask, features].fillna(0)
    y_valid = modeling_df.loc[valid_mask, 'target']
    
    X_test = modeling_df.loc[test_mask, features].fillna(0)
    y_test = modeling_df.loc[test_mask, 'target']
    
    logger.info(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    # 4. HGBT Training
    logger.info("Training Base HistGradientBoostingClassifier...")
    
    # 4a. Sample Weights for Balance (Train)
    neg, pos = np.bincount(y_train)
    scale_pos = neg / pos
    w_train = np.ones(len(y_train))
    w_train[y_train == 1] = scale_pos
    
    # Base Model
    hgb = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=0
    )
    
    # Fit Base on Train (with weights)
    hgb.fit(X_train, y_train, sample_weight=w_train)
    
    # 5. Calibration (Isotonic - Manual Implementation)
    # CalibratedClassifierCV 'prefit' is failing in this env. Using IsotonicRegression directly.
    logger.info("Calibrating on Validation Set (IsotonicRegression)...")
    from sklearn.isotonic import IsotonicRegression
    
    # Get raw scores (probabilities) from base model
    val_probs_base = hgb.predict_proba(X_valid)[:, 1]
    
    # Fit Isotonic
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_reg.fit(val_probs_base, y_valid)
    
    # 6. Evaluation
    logger.info("Evaluating on Test Set...")
    test_probs_base = hgb.predict_proba(X_test)[:, 1]
    y_prob = iso_reg.transform(test_probs_base)
    
    # Get Test Baseline
    test_prev = y_test.mean()
    logger.info(f"Test Set Prevalence (Baseline): {test_prev:.2%}")
    
    evaluate_model_block_boot(
        y_test, 
        y_prob, 
        modeling_df.loc[test_mask, 'player_id'], 
        baseline=test_prev, 
        name="HGBT_Calib_Manual"
    )
    
    # 7. Feature Importance (Permutation on Val)
    logger.info("Computing Feature Importance (on Validation)...")
    sample_size = min(10000, len(X_valid))
    X_val_sample = X_valid.sample(sample_size, random_state=42)
    y_val_sample = y_valid.loc[X_val_sample.index]
    
    # Wrapper to allow permutation importance to work with the pipeline
    class CalibratedPipeline:
        def __init__(self, base, calibrator):
            self.base = base
            self.calibrator = calibrator
        def predict(self, X):
            p = self.base.predict_proba(X)[:, 1]
            return self.calibrator.transform(p)
        def score(self, X, y):
            # Dummy score for permutation importance (we specify scoring='average_precision' anyway)
            return 0 
            
    # Custom scorer for permutation importance that uses the pipeline
    # actually permutation_importance expects an estimator with predict/predict_proba
    # Simple workaround: Just compute importance on the BASE model. 
    # The rank ordering shouldn't change much with isotonic calibration (monotonic transform).
    # Computing on base model is standard proxy.
    
    r = permutation_importance(
        hgb, X_val_sample[features], y_val_sample,
        n_repeats=10, random_state=42, n_jobs=-1, scoring='average_precision'
    )
    
    imp = pd.DataFrame({
        'feature': features,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n--- Feature Importance (Validation Set) ---")
    print(imp.head(15).to_markdown(index=False))

def evaluate_model_block_boot(y_true, y_prob, group_ids, baseline, name="Model", n_boot=1000):
    """
    Block Bootstrap by group_ids (player_id) to account for autocorrelation.
    """
    # Metrics
    trues = np.array(y_true)
    probs = np.array(y_prob)
    groups = np.array(group_ids)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Point Estimates
    p, r, _ = precision_recall_curve(trues, probs)
    pt_prauc = auc(r, p)
    pt_roc = roc_auc_score(trues, probs)
    pt_brier = brier_score_loss(trues, probs)
    
    # Top 5%
    k = max(1, int(len(trues) * 0.05))
    top_k_idx = np.argsort(probs)[-k:]
    pt_top5_prec = trues[top_k_idx].sum() / k
    
    print(f"\n--- {name} Results (Test Prev={baseline:.2%}) ---")
    print(f"ROC-AUC:       {pt_roc:.4f}")
    print(f"Brier Score:   {pt_brier:.4f}")
    
    # Bootstrap
    logger.info(f"Running Block Bootstrap ({n_boot} iterations, {n_groups} blocks)...")
    rng = np.random.RandomState(42)
    
    stats_prauc = []
    stats_top5 = []
    
    for _ in range(n_boot):
        # Sample groups with replacement
        boot_groups = rng.choice(unique_groups, n_groups, replace=True)
        
        # We need to reconstruct the indices. 
        # Optimized approach: Pre-map group -> indices
        # But for simplicity/readability in script: use a boolean mask or index list
        # Speed hack: Do it once out of loop? 
        # Re-implementation:
        # 1. Create a DF
        # 2. Sample DF groups
        pass 
    
    # Pandas implementation is cleaner for blocking
    df_boot = pd.DataFrame({'y': trues, 'p': probs, 'g': groups})
    
    for i in range(n_boot):
        # Sample groups
        sampled_groups = rng.choice(unique_groups, n_groups, replace=True)
        # This is slow if we do it naively.
        # Faster: Get indices for each group
        # Just use row bootstrap if blocking is too slow? User specifically asked for Block.
        # Let's try a simpler Block:
        # Just sample indices? No, must keep clusters.
        # Hack: Since n_boot=1000 might be slow in python loop, let's do 200 for speed 
        # or optimize.  
        # Actually, let's Stick to User Request: 1000 is requested.
        # Optimization:
        # Group indices map
        pass

    # ... (Actually, implementing efficient block bootstrap in pure python script is verbose)
    # Let's write a helper function inside 
    
    grp_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    
    for i in range(n_boot):
        # Resample groups
        resampled_gs = rng.choice(unique_groups, n_groups, replace=True)
        # Collect indices
        boot_idx = np.concatenate([grp_to_idx[g] for g in resampled_gs])
        
        y_b = trues[boot_idx]
        p_b = probs[boot_idx]
        
        if y_b.sum() == 0: continue
        
        # PR AUC
        pr, re, _ = precision_recall_curve(y_b, p_b)
        stats_prauc.append(auc(re, pr))
        
        # Top 5
        k_b = max(1, int(len(y_b) * 0.05))
        top_k = np.argsort(p_b)[-k_b:]
        stats_top5.append(y_b[top_k].sum() / k_b)
        
    # CIs
    print(f"PR-AUC:        {pt_prauc:.4f} (95% CI: {np.percentile(stats_prauc, 2.5):.4f} - {np.percentile(stats_prauc, 97.5):.4f})")
    print(f"Prec @ Top 5%: {pt_top5_prec:.2%} (95% CI: {np.percentile(stats_top5, 2.5):.2%} - {np.percentile(stats_top5, 97.5):.2%})")

if __name__ == "__main__":
    main()
