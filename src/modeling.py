import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import matplotlib.pyplot as plt

def time_split_train_test(df, date_col='week_start', test_size_months=6):
    """
    Splits data into train and test based on time.
    """
    max_date = df[date_col].max()
    split_date = max_date - pd.DateOffset(months=test_size_months)
    
    train = df[df[date_col] < split_date]
    test = df[df[date_col] >= split_date]
    
    return train, test, split_date

def train_baselines(train_df, test_df, feature_cols, target_col='target'):
    """
    Trains Logistic Regression and LightGBM.
    Returns metrics and models.
    """
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    metrics = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]
    
    metrics['lr'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_lr),
        'pr_auc': average_precision_score(y_test, y_pred_lr),
        'brier': brier_score_loss(y_test, y_pred_lr)
    }
    
    # 2. LightGBM
    lgbm = lgb.LGBMClassifier(class_weight='balanced', n_estimators=100)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict_proba(X_test)[:, 1]
    
    metrics['lgbm'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_lgbm),
        'pr_auc': average_precision_score(y_test, y_pred_lgbm),
        'brier': brier_score_loss(y_test, y_pred_lgbm)
    }
    
    return metrics, lr, lgbm

def plot_feature_importance(model, feature_names, save_path):
    """
    Plots feature importance for Tree model.
    """
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        imp.head(20).plot(kind='barh')
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(save_path)
    pass
