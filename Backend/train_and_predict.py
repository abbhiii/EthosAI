import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/adult.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "predicted_adult.csv")
RANDOM_SEED = 42

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(".", "_") for c in df.columns]
    df = df.applymap(lambda x: x.strip().replace('"', '') if isinstance(x, str) else x)
    df["target_bin"] = df["income"].apply(lambda v: 1 if ">50K" in str(v) else 0)
    return df

def preprocess(df):
    numeric_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    cat_cols = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    X_cat = ohe.fit_transform(df[cat_cols])
    X_cat_cols = ohe.get_feature_names_out(cat_cols)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric_cols])
    X = np.hstack([X_num, X_cat])
    feature_names = list(numeric_cols) + list(X_cat_cols)
    return X, df["target_bin"].values, feature_names

def main():
    df = load_and_clean(DATA_PATH)
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Macro avg:", report["macro avg"])
    print("Weighted avg:", report["weighted avg"])
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    coefs = model.coef_[0]
    top_features = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("\n🔥 Top 10 features:")
    for f, c in top_features:
        print(f"{f}: {c:.4f}")
    df["pred"] = model.predict(X)
    sensitive = "sex"
    groups = df[sensitive].unique().tolist()
    print("\n📊 Per-group TPR/FPR:")
    for g in groups:
        sub = df[df[sensitive] == g]
        tn, fp, fn, tp = confusion_matrix(sub["target_bin"], sub["pred"]).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        print(f"{g}: TPR={tpr:.3f}, FPR={fpr:.3f}, n={len(sub)}")
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved predictions to {OUT_PATH}")

if __name__ == "__main__":
    main()
