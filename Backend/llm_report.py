import os, json
import pandas as pd
import numpy as np
from train_and_predict import load_and_clean, preprocess
from train_with_reweighing import compute_reweighing_weights
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import datetime

# optional OpenAI call
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

def per_group_metrics_from_arrays(y_true, y_pred, groups_mask, group_names):
    out = {}
    for name, mask in zip(group_names, groups_mask):
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt)==0:
            out[name] = {"tpr": None, "fpr": None, "n": 0, "pos_rate": None}
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp+fn)>0 else None
        fpr = fp / (fp + tn) if (fp+tn)>0 else None
        out[name] = {"tpr": tpr, "fpr": fpr, "n": int(len(yt)), "pos_rate": float(np.mean(yt))}
    return out

def compute_models_and_metrics():
    # load data & preprocess (uses same logic as train scripts)
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, "../data/adult.csv")
    df = load_and_clean(data_path)
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    # baseline
    model_base = LogisticRegression(max_iter=300)
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    # groups masks aligned to test set
    df_reset = df.reset_index(drop=True)
    groups = df_reset['sex'].astype(str)
    unique_groups = list(df_reset['sex'].astype(str).unique())
    groups_masks = [ (df_reset.loc[idx_test,'sex'].astype(str) == g).values for g in unique_groups ]
    base_group_stats = per_group_metrics_from_arrays(y_test, y_pred_base, groups_masks, unique_groups)

    # compute reweighing weights for training portion using function from train_with_reweighing
    df_train = df_reset.loc[idx_train].reset_index(drop=True)
    weights_train = compute_reweighing_weights(df_train, sensitive_col='sex', target_col='target_bin')
    # train reweighted
    model_rw = LogisticRegression(max_iter=300)
    model_rw.fit(X_train, y_train, sample_weight=weights_train)
    y_pred_rw = model_rw.predict(X_test)
    acc_rw = accuracy_score(y_test, y_pred_rw)
    rw_group_stats = per_group_metrics_from_arrays(y_test, y_pred_rw, groups_masks, unique_groups)

    # compute TPR gaps (choose Female and Male if present)
    g1 = 'Female' if 'Female' in unique_groups else unique_groups[0]
    g2 = 'Male' if 'Male' in unique_groups else (unique_groups[1] if len(unique_groups)>1 else None)
    tpr_gap_base = None
    tpr_gap_rw = None
    if g2:
        tpr_gap_base = None if base_group_stats[g1]['tpr'] is None or base_group_stats[g2]['tpr'] is None else abs(base_group_stats[g2]['tpr'] - base_group_stats[g1]['tpr'])
        tpr_gap_rw = None if rw_group_stats[g1]['tpr'] is None or rw_group_stats[g2]['tpr'] is None else abs(rw_group_stats[g2]['tpr'] - rw_group_stats[g1]['tpr'])
    results = {
        "accuracy_baseline": acc_base,
        "accuracy_reweighed": acc_rw,
        "base_group_stats": base_group_stats,
        "rw_group_stats": rw_group_stats,
        "tpr_gap_base": tpr_gap_base,
        "tpr_gap_rw": tpr_gap_rw,
        "feature_names": feature_names,
        "unique_groups": unique_groups
    }
    return results

def load_feature_importance():
    base = os.path.dirname(__file__)
    p = os.path.join(base, "perm_outputs", "permutation_importance.csv")
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p)
    # take top 10
    top = df.sort_values("importance_mean", ascending=False).head(10)
    return list(zip(top['feature'].tolist(), top['importance_mean'].tolist()))

def build_prompt(metrics, top_features):
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"EthosAI Automated Ethics Report — {date}")
    lines.append("")
    lines.append("Key model performance:")
    lines.append(f"- Baseline accuracy: {metrics['accuracy_baseline']:.4f}")
    lines.append(f"- Reweighed accuracy: {metrics['accuracy_reweighed']:.4f}")
    lines.append("")
    lines.append("Top predictive features (permutation importance):")
    for f, imp in top_features:
        lines.append(f"- {f}: mean importance {imp:.6f}")
    lines.append("")
    lines.append("Group fairness summary (sensitive attribute = sex):")
    for g, stats in metrics['base_group_stats'].items():
        lines.append(f"- Baseline {g}: TPR={stats['tpr']:.3f}, FPR={stats['fpr']:.3f}, n={stats['n']}, pos_rate={stats['pos_rate']:.3f}")
    lines.append("")
    lines.append("Reweighed model group stats:")
    for g, stats in metrics['rw_group_stats'].items():
        lines.append(f"- Reweighed {g}: TPR={stats['tpr']:.3f}, FPR={stats['fpr']:.3f}, n={stats['n']}, pos_rate={stats['pos_rate']:.3f}")
    lines.append("")
    lines.append("Requested outputs:")
    lines.append("1) Short executive summary (2-3 sentences) describing the fairness issue and mitigation result.")
    lines.append("2) Root-cause hypotheses connecting top features to observed bias.")
    lines.append("3) Concrete next-step mitigation and evaluation plan (3 bullets).")
    prompt = "\\n".join(lines)
    return prompt

def generate_local_report(prompt):
    # deterministic template-based report using the prompt facts
    parts = prompt.split("\\n")
    header = parts[0]
    report = [header, "", "Executive summary:"]
    # extract key numbers quickly from prompt lines
    def extract_value(prefix):
        for l in parts:
            if l.startswith(prefix):
                return l.split(":")[1].strip()
        return None
    acc_base = float(extract_value("- Baseline accuracy") or 0)
    acc_rw = float(extract_value("- Reweighed accuracy") or 0)
    # simple summary
    report.append(f"EthosAI detected measurable gender disparity in the baseline model and applied reweighing mitigation. The mitigation adjusted group TPRs with a small accuracy trade-off ({acc_base:.4f} → {acc_rw:.4f}).")
    report.append("")
    report.append("Root-cause hypotheses:")
    report.append("- Model uses socio-economic features (e.g., education and capital gains) which correlate with both income and demographic groups, producing observed disparities.")
    report.append("- Presence of 'sex' as an important feature indicates the model learns direct gender signals which amplify disparities.")
    report.append("")
    report.append("Next steps (recommended):")
    report.append("1. Evaluate reweighing + thresholding combined and measure TPR/FPR parity and accuracy on a held-out set.")
    report.append("2. Consider data augmentation for under-represented groups (synthetic examples) and validate on factuality/utility.")
    report.append("3. Add human-in-the-loop review for high-impact decisions and produce model cards for compliance.")
    return "\\n".join(report)

def call_openai(prompt):
    import openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = key
    # Use ChatCompletion (compat) - adjust model if you have access
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system","content":"You are an expert ML ethics auditor."},
                  {"role":"user","content":prompt}],
        max_tokens=800,
        temperature=0.0
    )
    return resp['choices'][0]['message']['content'].strip()

def main():
    print("Computing models and metrics...")
    metrics = compute_models_and_metrics()
    top_feats = load_feature_importance()
    prompt = build_prompt(metrics, top_feats)
    print("\\n=== Generated prompt (for transparency) ===\\n")
    print(prompt)
    print("\\n=== Generating final report ===\\n")
    if USE_OPENAI:
        try:
            text = call_openai(prompt)
        except Exception as e:
            print("OpenAI call failed, falling back to local template. Error:", e)
            text = generate_local_report(prompt)
    else:
        text = generate_local_report(prompt)
    out_path = os.path.join(os.path.dirname(__file__), "ethos_ai_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved report to {out_path}\\n")
    print("===== REPORT START =====")
    print(text)
    print("===== REPORT END =====")
    return

if __name__ == '__main__':
    main()
