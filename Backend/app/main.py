from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import Optional
from sklearn.metrics import confusion_matrix
import uvicorn
import os
import sys
import subprocess
import numpy as np

app = FastAPI(title="EthosAI - Fairness & Report API", version="0.2.2")

# CORS for Next.js dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def map_income_to01(series: pd.Series) -> pd.Series:
    """
    Robust mapping for Adult 'income' labels. Works regardless of quotes/spaces/punctuation.
    1 if string contains '>50k', else 0 if string contains '<=50k', else 0.
    """
    s = series.astype(str).str.lower().str.replace('"', '', regex=False).str.replace(' ', '', regex=False)
    is_pos = s.str.contains('>50k', regex=False)
    is_neg = s.str.contains('<=50k', regex=False)
    out = np.where(is_pos, 1, np.where(is_neg, 0, 0)).astype(int)
    return pd.Series(out, index=series.index)

def to01_generic(series: pd.Series) -> pd.Series:
    """Fallback mapper for arbitrary binary targets, using common positives."""
    s = series.astype(str).str.strip().str.lower().str.replace('"', '', regex=False)
    positives = { "1","true","yes","y","positive",">50k",">50k.","income>50k","gt50k","gt50k." }
    return s.apply(lambda x: 1 if (x in positives or '>50k' in x) else 0).astype(int)

def compute_basic_fairness(df: pd.DataFrame, target_col: str, sensitive_col: str):
    """
    Minimal fairness report:
      - overall class distribution
      - positive rate by sensitive group (demographic parity)
      - (TPR/FPR filled later if predictions provided)
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in dataset")
    if sensitive_col not in df.columns:
        raise ValueError(f"sensitive_col '{sensitive_col}' not in dataset")

    df = df.copy()

    # Force robust string mapping for labels first; if that fails, use generic mapper.
    y_str = df[target_col].astype(str)
    y_num = map_income_to01(y_str)
    if y_num.sum() == 0 and y_num.mean() == 0:
        # If mapping produced all zeros, fall back to generic
        y_num = to01_generic(y_str)

    df["_target_bin"] = y_num

    groups = df[sensitive_col].astype(str).unique().tolist()

    report = {
        "overall": {
            "n_rows": int(len(df)),
            "positive_rate": float(df["_target_bin"].mean()),
            "class_counts": {str(k): int(v) for k, v in df["_target_bin"].value_counts().to_dict().items()}
        },
        "by_group": {}
    }

    for g in groups:
        sub = df[df[sensitive_col].astype(str) == g]
        if len(sub) == 0:
            continue
        pos_rate = float(sub["_target_bin"].mean())
        report["by_group"][g] = {
            "n": int(len(sub)),
            "positive_rate": pos_rate,
            "tpr": None,
            "fpr": None
        }

    return report

@app.post("/api/upload-dataset")
async def upload_dataset(
    dataset: UploadFile = File(...),
    target_col: str = Form(...),
    sensitive_col: str = Form(...),
    predictions_col: Optional[str] = Form(None)
):
    """
    Upload a CSV dataset and compute a minimal fairness report.

    Form fields:
    - dataset: CSV file
    - target_col: name of the label/target column (binary)
    - sensitive_col: sensitive attribute column (e.g., sex, race)
    - predictions_col (optional): if provided, computes TPR/FPR per group.
    """
    if not dataset.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    contents = await dataset.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {e}")

    # Base report with robust mapping
    try:
        report = compute_basic_fairness(df, target_col, sensitive_col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # If predictions provided, compute TPR/FPR per group
    if predictions_col:
        if predictions_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"predictions_col '{predictions_col}' not in dataset")

        # map true/pred labels robustly
        y_true = map_income_to01(df[target_col])
        if y_true.sum() == 0 and y_true.mean() == 0:
            y_true = to01_generic(df[target_col])

        y_pred = to01_generic(df[predictions_col])

        sens = df[sensitive_col].astype(str).values
        groups = list(report["by_group"].keys())

        for g in groups:
            mask = (sens == g)
            if mask.sum() == 0:
                continue
            yt = y_true[mask].values
            yp = y_pred[mask].values
            # Safe confusion matrix
            tn = fp = fn = tp = 0
            try:
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            except Exception:
                for a, b in zip(yt, yp):
                    if a == 0 and b == 0: tn += 1
                    elif a == 0 and b == 1: fp += 1
                    elif a == 1 and b == 0: fn += 1
                    elif a == 1 and b == 1: tp += 1

            tpr = (tp / (tp + fn)) if (tp + fn) > 0 else None
            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else None
            report["by_group"][g]["tpr"] = tpr
            report["by_group"][g]["fpr"] = fpr

    # Debug info to verify mapping
    _s = df[target_col].astype(str).str.lower().str.replace('"','', regex=False).str.replace(' ','', regex=False)
    _pos = int(_s.str.contains('>50k', regex=False).sum())
    _neg = int(_s.str.contains('<=50k', regex=False).sum())
    _uniq = _s.unique().tolist()[:6]
    return JSONResponse(content={"status": "ok", "report": report, "debug": {"unique": _uniq, "pos": _pos, "neg": _neg}})

@app.post("/api/generate-report")
def generate_report():
    """
    Run llm_report.py (recomputes metrics on YOUR local dataset at ../data/adult.csv)
    and return the report text as JSON.
    """
    # app/ -> backend/
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(BASE_DIR, "llm_report.py")
    report_path = os.path.join(BASE_DIR, "ethos_ai_report.txt")

    if not os.path.exists(script_path):
        return {"status": "error", "message": f"Script not found: {script_path}"}

    try:
        _ = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        return {"status": "error", "stderr": e.stderr, "stdout": e.stdout}

    if not os.path.exists(report_path):
        return {"status": "error", "message": "Report file not found after generation."}

    with open(report_path, "r") as f:
        text = f.read()

    return {"status": "ok", "report": text}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
