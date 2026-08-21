[README.md](https://github.com/user-attachments/files/28963182/README.md)
# 🛡 PhishGuard AI — Phishing Detection System

AI-powered phishing detection for emails and URLs using fine-tuned DistilBERT,
ensemble ML classifiers (RandomForest + XGBoost), and 40+ handcrafted heuristic rules.
Trained on 671,000 real phishing samples.

---

## 👥 Who Uses What

| User | OS | Setup Script | Launch Script |
|------|----|-------------|---------------|
| Linux | Ubuntu 24.04 Linux | `bash setup.sh` | `bash launch.sh` |
| Windows | Windows 10/11 | `setup.bat` | `launch.bat` |

> The `.sh` files are for **Linux/Mac only**.
> The `.bat` files are for **Windows only**.
> `start.py` works on **both** as an alternative.

---

## 🚀 Quick Start

### 🐧 Linux (Ubuntu 24.04)

```bash
# Step 1 — Go to project folder
cd ~/Desktop/phishing-detector_LLM

# Step 2 — Setup (first time only)
bash setup.sh

# Step 3 — Train on your dataset (first time only)
python3 train.py --dataset MASTER_phishing_dataset.csv

# Step 4 — Launch the app
bash launch.sh
```

Open browser at: **http://localhost:8501**

---

### 🪟 Windows (10/11)

```batch
REM Step 1 — Open Command Prompt in project folder

REM Step 2 — Setup (first time only)
setup.bat

REM Step 3 — Train on your dataset (first time only)
venv\Scripts\python.exe train.py --dataset MASTER_phishing_dataset.csv

REM Step 4 — Launch the app
launch.bat
```

Open browser at: **http://localhost:8501**

---

## 🗂 Project Structure

```
phishing-detector_LLM/
│
├── train.py                    ← Run this first to train on your dataset
├── start.py                    ← Cross-platform alternative (Linux + Windows)
├── check_env.py                ← Check system compatibility before setup
│
├── setup.sh  + launch.sh       ← Linux scripts
├── setup.bat + launch.bat      ← Windows scripts
├── requirements.txt            ← All Python dependencies (no version pins)
│
├── app/
│   └── streamlit_app.py        ← Web UI (dark-mode cybersecurity dashboard)
│
├── core/
│   ├── email_parser.py         ← Parses RFC-822 and plain-text emails
│   ├── heuristics_engine.py    ← 40+ rule-based phishing detection rules
│   ├── transformer_engine.py   ← DistilBERT inference (fine-tuned or zero-shot)
│   ├── url_analyzer.py         ← URL structural analysis (20+ features)
│   └── ensemble.py             ← Combines all signals with weighted voting
│
├── ml/
│   └── classifier.py           ← RandomForest + XGBoost (loads real trained models)
│
├── config/
│   └── settings.py             ← All tunable parameters in one place
│
├── utils/
│   ├── cache.py                ← Two-level disk + memory cache
│   └── logger.py               ← Colored rotating log output
│
├── models/                     ← Auto-created after running train.py
│   ├── distilbert_email/       ← Fine-tuned DistilBERT weights
│   ├── url_rf.joblib           ← Trained RandomForest
│   ├── url_xgb.joblib          ← Trained XGBoost
│   └── training_summary.json  ← F1, AUC, accuracy metrics
│
├── reports/                    ← Auto-created after running train.py
│   └── training_report.png    ← Loss curves, confusion matrix, ROC curves
│
├── tests/
│   └── test_suite.py           ← 43 pytest unit tests
│
└── .streamlit/
    └── config.toml             ← Dark theme, port 8501, no email prompt
```

---

## 🤖 AI Models Used

| Model | Size | Purpose |
|-------|------|---------|
| `distilbert-base-uncased` (fine-tuned) | ~250 MB | Email classifier — trained on 85k real phishing emails |
| `all-MiniLM-L6-v2` | ~80 MB | Semantic embeddings backup |
| `cross-encoder/nli-MiniLM2-L6-H768` | ~90 MB | Zero-shot fallback (used before training) |
| RandomForest (sklearn) | ~5 MB | URL classifier — trained on 114k real phishing URLs |
| XGBoost | ~5 MB | URL classifier ensemble |

---

## 🏋 Training on Your Dataset

Your dataset: **671,694 rows** from CEAS, Enron, Nazario, SpamAssassin, phishing URLs.

### Linux
```bash
# Full training — ~1-2 hours on Intel i5 (most accurate)
python3 train.py --dataset MASTER_phishing_dataset.csv

# Skip DistilBERT, only RF+XGBoost — ~5 minutes
python3 train.py --dataset MASTER_phishing_dataset.csv --skip-bert

# Quick test first — 2 minutes
python3 train.py --dataset MASTER_phishing_dataset.csv --quick --skip-bert
```

### Windows
```batch
venv\Scripts\python.exe train.py --dataset MASTER_phishing_dataset.csv
venv\Scripts\python.exe train.py --dataset MASTER_phishing_dataset.csv --skip-bert
venv\Scripts\python.exe train.py --dataset MASTER_phishing_dataset.csv --quick
```

After training, the app **automatically loads** your trained models on next launch.

---

## 📊 Detection Architecture

```
Email Input
     │
     ├──► DistilBERT (fine-tuned on 85k real emails)  ──► 40% weight
     ├──► Heuristics Engine (40+ rules)               ──► 35% weight
     ├──► URL Analyzer (20+ structural features)       ──► 15% weight
     └──► RF + XGBoost (trained on 114k real URLs)    ──► 10% weight
                              │
                              ▼
                    Final Phishing Score (0–100%)
```

---

## 🖥 System Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| OS | Ubuntu 20.04+ / Windows 10+ | Ubuntu 24.04 + Windows 11 |
| Python | 3.9+ | 3.12.3 |
| RAM | 4 GB available | 8 GB |
| CPU | Intel i5 or better | Intel Core i5 |
| Disk | 3 GB free | 512 GB SSD |
| GPU | Not required | CPU-only |

---

## ⚡ Performance (Intel i5, 8GB RAM)

| Operation | Time |
|-----------|------|
| First model load | ~30-45 sec |
| Email analysis | ~0.8-2.5 sec |
| URL analysis | ~0.05-0.3 sec |
| Cached result | ~1 ms |
| Training with DistilBERT | ~1-2 hours |
| Training skip-bert | ~5 minutes |
| RAM usage | ~1.5-2 GB |

---

## 🧪 Running Tests

### Linux
```bash
source venv/bin/activate
python3 -m pytest tests/test_suite.py -v
```

### Windows
```batch
venv\Scripts\activate
python -m pytest tests\test_suite.py -v
```

Expected: **43 tests passed**

---

## 🔧 Troubleshooting

### Linux

| Problem | Fix |
|---------|-----|
| `localhost refused to connect` | Use `bash launch.sh` not `python3 start.py` |
| `python3-venv not found` | `sudo apt install python3-venv python3-full` |
| Email prompt on startup | Fixed — `headless=true` in `.streamlit/config.toml` |
| Port 8501 in use | `sudo lsof -i :8501` then kill the process |

### Windows

| Problem | Fix |
|---------|-----|
| `torch==2.1.2 not found` | Already fixed in latest `requirements.txt` |
| `streamlit not recognized` | Run `setup.bat` first |
| Port 8501 in use | `netstat -ano \| findstr 8501` then kill the PID |
| Antivirus blocking | Add project folder to antivirus exclusions |

---

## 🔒 Privacy

- All analysis runs **locally** — no data sent externally
- Models download once from HuggingFace, then work **offline**
- Your dataset and emails never leave your computer

---

## 📁 Dataset Summary

| Source | Type | Rows |
|--------|------|------|
| phishing_site_urls | URL | 507,195 |
| phishing_email | Email | 82,077 |
| CEAS_08 | Email | 39,145 |
| Enron | Email | 29,726 |
| SpamAssassin | Email | 5,809 |
| Nigerian_Fraud | Email | 3,319 |
| Nazario | Email | 1,564 |
| Ling | Email | 2,859 |
| **Total** | | **671,694** |

| Label | Count |
|-------|-------|
| 0 — Legitimate | 471,720 |
| 1 — Phishing | 199,974 |

---

## 📝 Academic Notes

This project demonstrates:
- **Transfer learning** — DistilBERT pretrained on BookCorpus + Wikipedia, fine-tuned on phishing data
- **Ensemble methods** — neural network + gradient boosted trees + rule engine
- **Feature engineering** — 20+ handcrafted URL features + learned representations
- **Imbalanced learning** — class weights and probability calibration for 2.4:1 label imbalance
- **Explainability** — token highlighting, feature importance, per-module confidence scores
- **Real dataset** — 671k rows from CEAS, Enron, Nazario, SpamAssassin, phishing URL corpora
