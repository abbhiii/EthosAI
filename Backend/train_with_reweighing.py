import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from train_and_predict import load_and_clean, preprocess

# Paths
BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "../data/adult.csv")
RANDOM_SEED = 42

def per_group_metrics(y_true, y_pred, groups):
    """Return dict of group -> (TPR, FPR, n, positive_rate)"""
    out = {}
    for g, mask in groups.items():
        if mask.sum() == 0:
            out[g] = {"tpr": None, "fpr": None, "n": int(mask.sum()), "pos_rate": None}
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        # ensure binary labels 0/1
        try:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        except Exception:
            # fallback if confusion doesn't fit (single class)
            tn = fp = fn = tp = 0
            for yt_i, yp_i in zip(yt, yp):
                if yt_i == 0 and yp_i == 0: tn += 1
                if yt_i == 0 and yp_i == 1: fp += 1
                if yt_i == 1 and yp_i == 0: fn += 1
                if yt_i == 1 and yp_i == 1: tp += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        out[g] = {"tpr": tpr, "fpr": fpr, "n": int(mask.sum()), "pos_rate": float(yt.mean()) if len(yt)>0 else None}
    return out

def compute_reweighing_weights(df, sensitive_col='sex', target_col='target_bin'):
    """
    Compute sample weights using Kamiran & Calders style reweighing:
    weight(g,y) = (P(g) * P(y)) / P(g,y)
    where probabilities are estimated from counts.
    """
    N = len(df)
    # counts
    # ensure target_bin exists
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in df")
    # sensitive as str
    df['_s'] = df[sensitive_col].astype(str)
    counts_g = df['_s'].value_counts().to_dict()  # count per group
    counts_y = df[target_col].value_counts().to_dict()  # count per label
    counts_gy = df.groupby(['_s', target_col]).size().to_dict()  # count per (g,y)
    weights = np.ones(N, dtype=float)
    for i, row in df.reset_index(drop=True).iterrows():
        g = row['_s']
        y = int(row[target_col])
        cg = counts_g.get(g, 0)
        cy = counts_y.get(y, 0)
        cgy = counts_gy.get((g, y), 0)
        # Prevent division by zero
        if cgy == 0 or cg == 0 or cy == 0:
            w = 1.0
        else:
            p_g = cg / N
            p_y = cy / N
            p_gy = cgy / N
            w = (p_g * p_y) / p_gy
        weights[i] = w
    return weights

def main():
    print("Loading and preprocessing data...")
    df = load_and_clean(DATA_PATH)
    X, y, feature_names = preprocess(df)
    # train/test split (deterministic)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    # build groups mapping for metrics by sensitive attribute on the original df index
    sensitive_col = 'sex'
    # Build boolean masks over test set indices for each sensitive group
    df_reset = df.reset_index(drop=True)
    unique_groups = df_reset[sensitive_col].astype(str).unique().tolist()
    groups_test = {}
    for g in unique_groups:
        mask = df_reset.loc[idx_test, sensitive_col].astype(str) == g
        groups_test[g] = mask.values  # boolean numpy array aligned to test set

    # Baseline model (no weights)
    print("\\nTraining baseline Logistic Regression (no sample weights)...")
    model_base = LogisticRegression(max_iter=300)
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"Baseline Accuracy: {acc_base:.4f}")
    print("Baseline classification report (weighted avg):")
    print(classification_report(y_test, y_pred_base))
    base_group_stats = per_group_metrics(y_test, y_pred_base, groups_test)
    print("\\nBaseline per-group metrics:")
    for g, stats in base_group_stats.items():
        print(f"{g}: TPR={stats['tpr']}, FPR={stats['fpr']}, n={stats['n']}, pos_rate={stats['pos_rate']}")

    # Compute reweighing weights on training set rows (use df indices from idx_train)
    df_train = df_reset.loc[idx_train].reset_index(drop=True)
    weights_train = compute_reweighing_weights(df_train, sensitive_col=sensitive_col, target_col='target_bin')

    # Train weighted model using sample_weight
    print("\\nTraining reweighed Logistic Regression (sample_weight)...")
    model_rw = LogisticRegression(max_iter=300)
    model_rw.fit(X_train, y_train, sample_weight=weights_train)
    y_pred_rw = model_rw.predict(X_test)
    acc_rw = accuracy_score(y_test, y_pred_rw)
    print(f"Reweighed Accuracy: {acc_rw:.4f}")
    print("Reweighed classification report (weighted avg):")
    print(classification_report(y_test, y_pred_rw))
    rw_group_stats = per_group_metrics(y_test, y_pred_rw, groups_test)
    print("\\nReweighed per-group metrics:")
    for g, stats in rw_group_stats.items():
        print(f"{g}: TPR={stats['tpr']}, FPR={stats['fpr']}, n={stats['n']}, pos_rate={stats['pos_rate']}")

    # Compute gaps and improvements for the two groups we care about (Female vs Male)
    g1 = 'Female' if 'Female' in base_group_stats else list(base_group_stats.keys())[0]
    g2 = 'Male' if 'Male' in base_group_stats else (list(base_group_stats.keys())[1] if len(base_group_stats)>1 else None)
    if g2 is not None:
        tpr_gap_base = None
        tpr_gap_rw = None
        if base_group_stats[g1]['tpr'] is not None and base_group_stats[g2]['tpr'] is not None:
            tpr_gap_base = abs(base_group_stats[g2]['tpr'] - base_group_stats[g1]['tpr'])
        if rw_group_stats[g1]['tpr'] is not None and rw_group_stats[g2]['tpr'] is not None:
            tpr_gap_rw = abs(rw_group_stats[g2]['tpr'] - rw_group_stats[g1]['tpr'])
        print(f"\\nTPR gap (base) between {g2} and {g1}: {tpr_gap_base}")
        print(f"TPR gap (reweighed) between {g2} and {g1}: {tpr_gap_rw}")
        if tpr_gap_base is not None and tpr_gap_rw is not None:
            reduction = (tpr_gap_base - tpr_gap_rw) / tpr_gap_base * 100 if tpr_gap_base != 0 else 0.0
            print(f"TPR gap reduction: {reduction:.2f}%")
    else:
        print("Unable to compute gender gap (group labels missing).")

if __name__ == "__main__":
    main()
