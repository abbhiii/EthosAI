import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from train_and_predict import load_and_clean, preprocess
from sklearn.model_selection import train_test_split

# paths
BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "../data/adult.csv")
PRED_PATH = os.path.join(BASE, "predicted_adult.csv")
OUT_DIR = os.path.join(BASE, "shap_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def load_model_and_data():
    # load data and model
    df = load_and_clean(DATA_PATH)
    X, y, feature_names = preprocess(df)
    # train/test split same deterministic way as train_and_predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # fit a model same as before so we have access to it (or load saved model if you persisted)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    return model, X_test, y_test, feature_names, df

def compute_shap_for_linear(model, X_sample, feature_names):
    """Use LinearExplainer for linear models — fast and exact for linear models."""
    explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)
    # shap_values shape: (n_features,) or (n_classes, n_samples, n_features) depending; handle binary
    return explainer, shap_values

def main():
    print("Loading model and data...")
    model, X_test, y_test, feature_names, df = load_model_and_data()
    # Use a sample for SHAP computation to save time (we'll compute global on 2000 rows)
    sample_size = min(2000, X_test.shape[0])
    X_sample = X_test[:sample_size]
    print(f"Computing SHAP values on sample_size={sample_size} ...")
    explainer, shap_vals = compute_shap_for_linear(model, X_sample, feature_names)
    # shap_vals might be shaped (n_samples, n_features) for binary; ensure shape
    shap_arr = np.array(shap_vals)
    # global importance: mean absolute shap per feature
    mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
    feat_imp = list(zip(feature_names, mean_abs_shap))
    feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)
    top10 = feat_imp_sorted[:10]
    print("\\nTop 10 features by mean(|SHAP|):")
    for name, val in top10:
        print(f"{name}: {val:.6f}")

    # Save a CSV of global feature importances
    fi_df = pd.DataFrame(feat_imp_sorted, columns=["feature", "mean_abs_shap"])
    fi_df.to_csv(os.path.join(OUT_DIR, "global_feature_importance_shap.csv"), index=False)
    print(f"Saved global feature importances to {os.path.join(OUT_DIR, 'global_feature_importance_shap.csv')}")

    # Save shap values for the first 20 test rows as examples (local explanations)
    n_local = min(20, X_sample.shape[0])
    shap_local = shap_arr[:n_local, :]
    local_df = pd.DataFrame(shap_local, columns=feature_names)
    local_df['true'] = y_test[:n_local]
    local_df.to_csv(os.path.join(OUT_DIR, "local_shap_values_sample.csv"), index=False)
    print(f"Saved local SHAP values for {n_local} rows to {os.path.join(OUT_DIR, 'local_shap_values_sample.csv')}")

    # Plot global importance bar chart
    plt.figure(figsize=(10,6))
    top_feats = feat_imp_sorted[:15]
    names = [t[0] for t in top_feats][::-1]
    vals = [t[1] for t in top_feats][::-1]
    plt.barh(range(len(vals)), vals, align='center')
    plt.yticks(range(len(vals)), names)
    plt.xlabel("mean(|SHAP value|)")
    plt.title("Top features by mean |SHAP value|")
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "global_shap_bar.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved SHAP bar plot to {plot_path}")

    # Save one force plot for the first test instance (requires JS for interactive; we save as matplotlib fallback)
    try:
        fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_arr[0], feature_names=feature_names, show=False)
        # waterfall_legacy returns a matplotlib figure if show=False
        png_path = os.path.join(OUT_DIR, "shap_force_example0.png")
        fig.savefig(png_path, bbox_inches='tight', dpi=150)
        print(f"Saved example waterfall plot to {png_path}")
    except Exception as e:
        print("Could not create waterfall plot (matplotlib fallback failed):", e)

    print("\\nDone. Inspect files in the folder:", OUT_DIR)

if __name__ == "__main__":
    main()
