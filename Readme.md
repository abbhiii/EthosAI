# EthosAI — Automated AI Fairness & Ethics Engine

EthosAI is a full-stack platform that detects, visualizes, and explains **demographic bias** in machine-learning models.  
Upload any CSV dataset (and optional predictions), and the system computes **Positive Rate, TPR, FPR, and group-level fairness metrics** — then generates an **AI-written ethics report**.

This project demonstrates:  
✅ Responsible AI  
✅ Bias detection & explainability  
✅ Real ML fairness metrics  
✅ Full-stack engineering (FastAPI + Next.js)

---

## 🚀 Features

### ✅ **Bias & Fairness Analytics**
- Positive rate by sensitive attribute (gender, race, etc.)
- Group metrics (TPR, FPR)
- Auto-cleaning of label formats (`<=50k` → 0, `>50k` → 1)
- Supports additional prediction column (e.g., `pred`)

### ✅ **Ethical AI Report (LLM-Generated)**
- Executive summary
- Root-cause hypotheses
- Mitigation recommendations

### ✅ **Interactive Dashboard**
- Upload CSV  
- View grouped statistics  
- Visual charts (TPR / FPR bar charts)  
- View ethics report  

---

## 📊 Example Results (Adult Income Dataset)

| Metric | Value |
|-------|-------|
| **Rows** | 32,561 |
| **Positive Rate** | 0.2408 |
| **Baseline Accuracy** | 0.8529 |
| **Reweighed Accuracy** | 0.8480 |

### Group Fairness (with predictions)
| Group | TPR | FPR |
|-------|---------|-----------|
| **Female** | 0.513 | 0.0226 |
| **Male** | 0.622 | 0.0996 |

---

## 📁 Project Structure
EthosAI/
├── Backend/ # FastAPI + ML fairness engine
│ ├── app/
│ │ └── main.py
│ ├── train_and_predict.py
│ ├── train_with_reweighing.py
│ ├── explain_permutation.py
│ ├── llm_report.py
│ ├── predicted_adult.csv (demo)
│ └── requirements.txt
│
├── Frontend/ # Next.js fairness dashboard
│ ├── app/
│ ├── public/
│ ├── package.json
│ ├── tsconfig.json
│ ├── next.config.ts
│ ├── eslint.config.mjs
│ └── postcss.config.mjs
│
└── README.md


---

## ✅ **Backend Setup (FastAPI)**

```bash
cd Backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at:
👉 http://127.0.0.1:8000

Health check:
```
curl http://127.0.0.1:8000/health
```

✅ Frontend Setup (Next.js + Tailwind)
cd Frontend
npm install
npm run dev


Frontend runs at:
👉 http://localhost:3000

🌟 Why This Project Matters

ML models often behave unfairly across gender, race, age, or region—especially in hiring, finance, and insurance.
EthosAI solves a real-life problem: it automatically detects harmful bias, explains why it exists, and provides actionable next steps.

Recruiters love this because it shows:
✅ You understand ML deeply
✅ You can build real full-stack systems
✅ You care about ethical & responsible AI
