import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from train_and_predict import load_and_clean, preprocess

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "perm_outputs")
os.makedirs(OUT, exist_ok=True)

def train_model():
    df = load_and_clean(os.path.join(BASE, "../data/adult.csv"))
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    return model, X_test, y_test, feature_names, df, X_train

def main():
    print("Training model and computing permutation importance...")
    model, X_test, y_test, feature_names, df, X_train = train_model()
    # permutation importance (n_repeats small for speed)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    stds = result.importances_std
    feat_imp = list(zip(feature_names, importances, stds))
    feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)
    top10 = feat_imp_sorted[:10]
    print("\\nTop 10 features by permutation importance (mean decrease in score):")
    for f, imp, sd in top10:
        print(f"{f}: {imp:.6f} ± {sd:.6f}")
    # Save CSV
    fi_df = pd.DataFrame(feat_imp_sorted, columns=["feature", "importance_mean", "importance_std"])
    fi_df.to_csv(os.path.join(OUT, "permutation_importance.csv"), index=False)
    print(f"Saved permutation_importance.csv to {os.path.join(OUT, 'permutation_importance.csv')}")
    # Plot bar chart
    top_feats = feat_imp_sorted[:15]
    names = [t[0] for t in top_feats][::-1]
    vals = [t[1] for t in top_feats][::-1]
    plt.figure(figsize=(10,6))
    plt.barh(range(len(vals)), vals, xerr=[t[2] for t in top_feats][::-1], align='center')
    plt.yticks(range(len(vals)), names)
    plt.xlabel("Permutation importance (mean decrease in score)")
    plt.title("Top features by permutation importance")
    plt.tight_layout()
    plot_path = os.path.join(OUT, "permutation_importance_bar.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved bar plot to {plot_path}")
    # Partial dependence plot for the top numeric feature (if numeric exists)
    # choose first numeric feature in preprocess order if present
    numeric_candidates = [n for n in feature_names if n in ("age","education_num","capital_gain","capital_loss","hours_per_week")]
    if len(numeric_candidates) > 0:
        feat = numeric_candidates[0]
        idx = feature_names.index(feat)
        # PDP expects the original X_train (2D) and feature index
        try:
            fig, ax = plt.subplots(figsize=(6,4))
            display = PartialDependenceDisplay.from_estimator(model, X_train, [idx], feature_names=feature_names, ax=ax)
            pdp_path = os.path.join(OUT, f"pdp_{feat}.png")
            fig.savefig(pdp_path, dpi=150, bbox_inches="tight")
            print(f"Saved PDP for {feat} to {pdp_path}")
        except Exception as e:
            print("Could not compute PDP:", e)
    else:
        print("No numeric candidate found for PDP.")

    # save small local table: top features and signs via coefficient if available
    try:
        coefs = model.coef_[0]
        coef_map = dict(zip(feature_names, coefs))
        local_table = []
        for f, imp, sd in top10:
            sign = "pos" if coef_map.get(f,0) >=0 else "neg"
            local_table.append((f, imp, sd, sign, coef_map.get(f,0)))
        local_df = pd.DataFrame(local_table, columns=["feature","importance_mean","importance_std","coef_sign","coef_value"])
        local_df.to_csv(os.path.join(OUT, "top10_feature_summary.csv"), index=False)
        print(f"Saved top10_feature_summary.csv to {os.path.join(OUT, 'top10_feature_summary.csv')}")
    except Exception as e:
        print("Could not save coefficient summary:", e)

    print("\\nDone. Inspect outputs in:", OUT)

if __name__ == '__main__':
    main()
