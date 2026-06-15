#!/usr/bin/env python3
"""
PhishGuard AI — Training Pipeline
==================================
Trains on your MASTER_phishing_dataset.csv with:
  1. DistilBERT  — fine-tuned on real email data (replaces zero-shot)
  2. RandomForest + XGBoost — trained on real URL data (replaces synthetic)
  3. Full evaluation: F1, ROC-AUC, confusion matrix, training curves
  4. Saves all models → app loads them automatically on next launch

Usage:
  python3 train.py --dataset MASTER_phishing_dataset.csv
  python3 train.py --dataset MASTER_phishing_dataset.csv --quick   # fast test run
  python3 train.py --dataset MASTER_phishing_dataset.csv --email-samples 20000
"""
import os, sys, csv, random, time, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

csv.field_size_limit(10_000_000)

MODELS_DIR  = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(description="PhishGuard AI Training Pipeline")
    ap.add_argument("--dataset",       required=True,  help="Path to MASTER_phishing_dataset.csv")
    ap.add_argument("--email-samples", type=int, default=20000, help="Email rows to use for DistilBERT (default 20000)")
    ap.add_argument("--url-samples",   type=int, default=60000, help="URL rows for RF/XGBoost (default 60000)")
    ap.add_argument("--epochs",        type=int, default=3,     help="DistilBERT training epochs (default 3)")
    ap.add_argument("--batch-size",    type=int, default=16,    help="DistilBERT batch size (default 16)")
    ap.add_argument("--max-len",       type=int, default=128,   help="Max token length (default 128)")
    ap.add_argument("--lr",            type=float, default=2e-5,help="Learning rate (default 2e-5)")
    ap.add_argument("--quick",         action="store_true",     help="Quick mode: 2000 email samples, 1 epoch")
    ap.add_argument("--skip-bert",     action="store_true",     help="Skip DistilBERT, only train RF/XGBoost")
    ap.add_argument("--seed",          type=int, default=42)
    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load & Split Dataset
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(path: str, email_n: int, url_n: int, seed: int):
    print("\n[1/5] Loading dataset...")
    t0 = time.time()

    email_phish, email_legit = [], []
    url_phish,   url_legit   = [], []

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row["text"].strip()
            lbl = int(row["label"])
            typ = row["type"]
            if len(txt) < 5:
                continue
            if typ == "email":
                (email_phish if lbl == 1 else email_legit).append(txt)
            else:
                (url_phish   if lbl == 1 else url_legit).append(txt)

    rng = random.Random(seed)

    # ── Email: balanced sample ─────────────────────────────────────────────
    half = email_n // 2
    ep = rng.sample(email_phish, min(half, len(email_phish)))
    el = rng.sample(email_legit, min(half, len(email_legit)))
    email_texts  = ep + el
    email_labels = [1]*len(ep) + [0]*len(el)
    combined = list(zip(email_texts, email_labels))
    rng.shuffle(combined)
    email_texts, email_labels = zip(*combined)
    email_texts, email_labels = list(email_texts), list(email_labels)

    # ── URL: balanced sample ───────────────────────────────────────────────
    half_u = url_n // 2
    up = rng.sample(url_phish, min(half_u, len(url_phish)))
    ul = rng.sample(url_legit, min(half_u, len(url_legit)))
    url_texts  = up + ul
    url_labels = [1]*len(up) + [0]*len(ul)
    combined_u = list(zip(url_texts, url_labels))
    rng.shuffle(combined_u)
    url_texts, url_labels = zip(*combined_u)
    url_texts, url_labels = list(url_texts), list(url_labels)

    elapsed = time.time() - t0
    print(f"       Loaded in {elapsed:.1f}s")
    print(f"       Email — phish: {len(ep):,}  legit: {len(el):,}  total: {len(email_texts):,}")
    print(f"       URL   — phish: {len(up):,}  legit: {len(ul):,}  total: {len(url_texts):,}")

    return email_texts, email_labels, url_texts, url_labels


def train_val_test_split(texts, labels, seed=42, val=0.15, test=0.15):
    """Stratified 70/15/15 split."""
    from sklearn.model_selection import train_test_split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=val+test, stratify=labels, random_state=seed
    )
    ratio = test / (val + test)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=ratio, stratify=y_tmp, random_state=seed
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DistilBERT Fine-Tuning (Email Classifier)
# ══════════════════════════════════════════════════════════════════════════════
def train_distilbert(texts_tr, labels_tr, texts_val, labels_val,
                     texts_te, labels_te,
                     epochs=3, batch_size=16, max_len=128, lr=2e-5):
    print("\n[2/5] Fine-tuning DistilBERT on email data...")
    print(f"       Train: {len(texts_tr):,}  Val: {len(labels_val):,}  Test: {len(texts_te):,}")
    print(f"       Epochs: {epochs}  Batch: {batch_size}  MaxLen: {max_len}  LR: {lr}")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                                  precision_score, recall_score, confusion_matrix,
                                  classification_report)

    device = torch.device("cpu")
    print(f"       Device: CPU")

    # ── Dataset ────────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class EmailDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                texts, truncation=True, padding=True,
                max_length=max_len, return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

    print("       Tokenizing...")
    ds_tr  = EmailDataset(texts_tr,  labels_tr)
    ds_val = EmailDataset(texts_val, labels_val)
    ds_te  = EmailDataset(texts_te,  labels_te)

    dl_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size*2)
    dl_te  = DataLoader(ds_te,  batch_size=batch_size*2)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dl_tr) * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                         total_iters=total_steps)

    # ── Training Loop ──────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_f1   = 0.0
    best_path = str(MODELS_DIR / "distilbert_email")

    print(f"\n       {'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Val Acc':<10} {'Time'}")
    print(f"       {'-'*60}")

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        train_losses = []

        for batch_idx, (inputs, lbls) in enumerate(dl_tr):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            lbls   = lbls.to(device)

            outputs = model(**inputs, labels=lbls)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

            # Progress bar every 50 batches
            if (batch_idx + 1) % 50 == 0:
                pct = (batch_idx + 1) / len(dl_tr) * 100
                print(f"\r       Epoch {epoch} [{pct:5.1f}%] loss={np.mean(train_losses):.4f}", end="", flush=True)

        print("\r", end="")

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_losses, val_preds, val_probs, val_true = [], [], [], []
        with torch.no_grad():
            for inputs, lbls in dl_val:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                lbls   = lbls.to(device)
                out    = model(**inputs, labels=lbls)
                val_losses.append(out.loss.item())
                probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
                preds = out.logits.argmax(dim=1).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_true.extend(lbls.cpu().numpy())

        tr_loss  = np.mean(train_losses)
        vl_loss  = np.mean(val_losses)
        vl_f1    = f1_score(val_true, val_preds)
        vl_acc   = accuracy_score(val_true, val_preds)
        elapsed  = time.time() - t_ep

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_f1"].append(vl_f1)
        history["val_acc"].append(vl_acc)

        mark = " ← best" if vl_f1 > best_f1 else ""
        print(f"       {epoch:<6} {tr_loss:<12.4f} {vl_loss:<12.4f} {vl_f1:<10.4f} {vl_acc:<10.4f} {elapsed:.0f}s{mark}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    # ── Test Evaluation ────────────────────────────────────────────────────
    print("\n       Loading best checkpoint for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(best_path).to(device)
    model.eval()

    te_preds, te_probs, te_true = [], [], []
    with torch.no_grad():
        for inputs, lbls in dl_te:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out    = model(**inputs)
            probs  = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            preds  = out.logits.argmax(dim=1).cpu().numpy()
            te_probs.extend(probs)
            te_preds.extend(preds)
            te_true.extend(lbls.numpy())

    metrics = {
        "accuracy":  accuracy_score(te_true, te_preds),
        "f1":        f1_score(te_true, te_preds),
        "precision": precision_score(te_true, te_preds),
        "recall":    recall_score(te_true, te_preds),
        "roc_auc":   roc_auc_score(te_true, te_probs),
        "conf_matrix": confusion_matrix(te_true, te_preds).tolist(),
        "probs":     te_probs,
        "true":      te_true,
        "preds":     te_preds,
    }

    print(f"\n       ── DistilBERT Test Results ──────────────────────")
    print(f"       Accuracy:   {metrics['accuracy']:.4f}")
    print(f"       F1 Score:   {metrics['f1']:.4f}")
    print(f"       Precision:  {metrics['precision']:.4f}")
    print(f"       Recall:     {metrics['recall']:.4f}")
    print(f"       ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"       ─────────────────────────────────────────────────")
    print(f"\n       {classification_report(te_true, te_preds, target_names=['Legitimate','Phishing'])}")
    print(f"       Model saved → {best_path}/")

    return history, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — RF + XGBoost on Real URL Data
# ══════════════════════════════════════════════════════════════════════════════
def extract_url_features(url: str) -> list:
    """Extract numerical features from a URL string."""
    import math, re

    def entropy(s):
        if not s: return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        n = len(s)
        return -sum((v/n)*math.log2(v/n) for v in freq.values())

    url = url.strip()
    if not url.startswith("http"):
        url = "http://" + url

    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        domain = p.netloc.split(":")[0].replace("www.", "")
        path   = p.path
        query  = p.query
    except Exception:
        domain = url
        path   = ""
        query  = ""

    suspicious_tlds = {".tk",".ml",".ga",".cf",".gq",".xyz",".top",".club",
                       ".work",".date",".racing",".download",".stream",".bid",
                       ".click",".win",".loan",".review",".trade",".faith"}
    phish_kw = {"login","signin","secure","account","verify","update","confirm",
                "paypal","ebay","amazon","apple","microsoft","bank","support",
                "password","credential","suspended","urgent","alert","recover"}
    brands   = {"paypal","apple","google","amazon","microsoft","facebook",
                "netflix","ebay","instagram","twitter","chase","bankofamerica"}

    tld = "." + domain.split(".")[-1] if "." in domain else ""
    sub_parts = domain.split(".")[:-2]

    url_lower = url.lower()
    dom_lower = domain.lower()

    ip_re = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

    return [
        min(1.0, len(url)   / 300),
        min(1.0, len(domain)/ 50),
        min(1.0, len(path)  / 200),
        min(1.0, len(sub_parts) / 5),
        float(bool(ip_re.match(domain))),
        float(url.startswith("https")),
        float("@" in url),
        float("//" in path),
        float(any(r in query.lower() for r in ["url=","redirect=","next=","goto="])),
        float(tld in suspicious_tlds),
        float(any(k in url_lower for k in phish_kw)),
        float(any(b in ".".join(sub_parts).lower() for b in brands) or
              any(b in dom_lower.split(".")[0] and dom_lower.split(".")[0] != b
                  for b in brands)),
        float(bool(re.search(r"xn--", url)) or any(ord(c)>127 for c in url)),
        float(url.count(".") > 5),
        float(":" in domain),
        min(1.0, entropy(url)    / 6.0),
        min(1.0, entropy(domain) / 6.0),
        min(1.0, sum(c.isdigit() for c in domain) / max(len(domain),1)),
        min(1.0, sum(1 for c in url if not c.isalnum() and c not in "://.-_?=&%") / max(len(url),1)),
        min(1.0, len(url.split("/")) / 10),
        float(bool(re.search(r"\.(exe|bat|zip|rar|js|vbs|scr|pif)\b", url, re.I))),
    ]


def train_url_classifier(url_texts, url_labels, seed=42):
    print("\n[3/5] Training URL classifier (RF + XGBoost) on real data...")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                                  precision_score, recall_score, confusion_matrix,
                                  classification_report)
    from xgboost import XGBClassifier
    import joblib

    print(f"       Extracting features from {len(url_texts):,} URLs...")
    t0 = time.time()
    X = np.array([extract_url_features(u) for u in url_texts], dtype=np.float32)
    y = np.array(url_labels)
    print(f"       Feature extraction: {time.time()-t0:.1f}s  Shape: {X.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                               stratify=y, random_state=seed)
    print(f"       Train: {len(X_tr):,}  Test: {len(X_te):,}")

    results = {}

    # ── Random Forest ──────────────────────────────────────────────────────
    print("       Training RandomForest (120 trees)...")
    t0 = time.time()
    rf = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=3,
                               class_weight="balanced", n_jobs=-1, random_state=seed),
        cv=3, method="isotonic"
    )
    rf.fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_te)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    results["rf"] = {
        "accuracy":  accuracy_score(y_te, rf_preds),
        "f1":        f1_score(y_te, rf_preds),
        "precision": precision_score(y_te, rf_preds),
        "recall":    recall_score(y_te, rf_preds),
        "roc_auc":   roc_auc_score(y_te, rf_probs),
        "conf_matrix": confusion_matrix(y_te, rf_preds).tolist(),
        "probs": rf_probs, "true": y_te, "preds": rf_preds,
    }
    print(f"       RF done in {time.time()-t0:.1f}s  F1={results['rf']['f1']:.4f}  AUC={results['rf']['roc_auc']:.4f}")

    # ── XGBoost ────────────────────────────────────────────────────────────
    print("       Training XGBoost (100 trees)...")
    t0 = time.time()
    scale = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=scale, eval_metric="logloss",
                         verbosity=0, random_state=seed, n_jobs=-1)
    xgb.fit(X_tr, y_tr)
    xgb_probs = xgb.predict_proba(X_te)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    results["xgb"] = {
        "accuracy":  accuracy_score(y_te, xgb_preds),
        "f1":        f1_score(y_te, xgb_preds),
        "precision": precision_score(y_te, xgb_preds),
        "recall":    recall_score(y_te, xgb_preds),
        "roc_auc":   roc_auc_score(y_te, xgb_probs),
        "conf_matrix": confusion_matrix(y_te, xgb_preds).tolist(),
        "probs": xgb_probs, "true": y_te, "preds": xgb_preds,
    }
    print(f"       XGB done in {time.time()-t0:.1f}s  F1={results['xgb']['f1']:.4f}  AUC={results['xgb']['roc_auc']:.4f}")

    # ── Ensemble vote ──────────────────────────────────────────────────────
    ens_probs = 0.55 * rf_probs + 0.45 * xgb_probs
    ens_preds = (ens_probs >= 0.5).astype(int)
    results["ensemble"] = {
        "accuracy":  accuracy_score(y_te, ens_preds),
        "f1":        f1_score(y_te, ens_preds),
        "roc_auc":   roc_auc_score(y_te, ens_probs),
        "probs": ens_probs, "true": y_te, "preds": ens_preds,
        "conf_matrix": confusion_matrix(y_te, ens_preds).tolist(),
    }

    print(f"\n       ── URL Classifier Results ────────────────────────")
    for name in ["rf", "xgb", "ensemble"]:
        r = results[name]
        print(f"       {name.upper():<10}  Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}")
    print(f"       ─────────────────────────────────────────────────")
    print(f"\n       {classification_report(y_te, ens_preds, target_names=['Legitimate','Phishing'])}")

    # Save models
    rf_path  = str(MODELS_DIR / "url_rf.joblib")
    xgb_path = str(MODELS_DIR / "url_xgb.joblib")
    joblib.dump(rf,  rf_path)
    joblib.dump(xgb, xgb_path)
    print(f"       Models saved → {rf_path}")
    print(f"                    → {xgb_path}")

    # Feature importances (from RF base)
    try:
        feat_names = ["url_len","dom_len","path_len","subdomain_depth","is_ip",
                      "is_https","has_at","double_slash","has_redirect","susp_tld",
                      "susp_keyword","brand_impersonation","unicode_obfusc",
                      "excessive_dots","has_port","url_entropy","dom_entropy",
                      "digit_ratio","special_char_ratio","path_depth","exec_ext"]
        base_rf = rf.calibrated_classifiers_[0].estimator
        importances = dict(zip(feat_names, base_rf.feature_importances_))
        results["feature_importances"] = importances
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Generate Evaluation Report (Plots)
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(bert_history, bert_metrics, url_results, email_n, url_n, epochs):
    print("\n[4/5] Generating evaluation report...")
    from sklearn.metrics import roc_curve
    import seaborn as sns

    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#0d1117",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#c9d1d9",
        "grid.color":        "#21262d",
        "figure.titlesize":  14,
    })

    CYAN  = "#00d4ff"
    GREEN = "#00ff88"
    RED   = "#ff3366"
    ORANGE= "#ff8c00"

    has_bert = bert_history is not None

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#070b14")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Title ────────────────────────────────────────────────────────────
    fig.suptitle("PhishGuard AI — Training Evaluation Report",
                 fontsize=18, fontweight="bold", color="#e2e8f0", y=0.98)

    # ── [0,0-1] DistilBERT Loss Curves ───────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0:2])
    if has_bert:
        ep_range = range(1, len(bert_history["train_loss"]) + 1)
        ax0.plot(ep_range, bert_history["train_loss"], "o-", color=CYAN,   lw=2, ms=6, label="Train Loss")
        ax0.plot(ep_range, bert_history["val_loss"],   "s-", color=ORANGE, lw=2, ms=6, label="Val Loss")
        ax0.set_title("DistilBERT — Training & Validation Loss", color="#e2e8f0", fontsize=11, pad=8)
        ax0.set_xlabel("Epoch"); ax0.set_ylabel("Cross-Entropy Loss")
        ax0.legend(framealpha=0.2); ax0.set_xticks(list(ep_range))
    else:
        ax0.text(0.5, 0.5, "DistilBERT training skipped\n(--skip-bert flag)",
                 ha="center", va="center", color="#475569", fontsize=12)
        ax0.set_title("DistilBERT Loss Curves", color="#e2e8f0", fontsize=11)

    # ── [0,2-3] DistilBERT F1 & Accuracy ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2:4])
    if has_bert:
        ax1.plot(ep_range, bert_history["val_f1"],  "o-", color=GREEN, lw=2, ms=6, label="Val F1")
        ax1.plot(ep_range, bert_history["val_acc"], "s-", color=CYAN,  lw=2, ms=6, label="Val Accuracy")
        ax1.axhline(bert_metrics["f1"],      linestyle="--", color=GREEN,  alpha=0.4, lw=1)
        ax1.axhline(bert_metrics["accuracy"],linestyle="--", color=CYAN,   alpha=0.4, lw=1)
        ax1.set_title("DistilBERT — Validation F1 & Accuracy", color="#e2e8f0", fontsize=11, pad=8)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.05); ax1.legend(framealpha=0.2); ax1.set_xticks(list(ep_range))
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", color="#475569")
        ax1.set_title("DistilBERT F1 & Accuracy", color="#e2e8f0", fontsize=11)

    # ── [1,0] DistilBERT Confusion Matrix ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if has_bert:
        cm = np.array(bert_metrics["conf_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", ax=ax2,
                    cmap="Blues", linewidths=0.5,
                    xticklabels=["Legit","Phish"],
                    yticklabels=["Legit","Phish"],
                    annot_kws={"size":12,"color":"white"})
        ax2.set_title("DistilBERT\nConfusion Matrix (Test)", color="#e2e8f0", fontsize=10)
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    else:
        ax2.text(0.5,0.5,"Skipped",ha="center",va="center",color="#475569")
        ax2.set_title("DistilBERT Confusion Matrix", color="#e2e8f0", fontsize=10)

    # ── [1,1] DistilBERT ROC Curve ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if has_bert:
        fpr, tpr, _ = roc_curve(bert_metrics["true"], bert_metrics["probs"])
        ax3.plot(fpr, tpr, color=CYAN, lw=2, label=f"AUC={bert_metrics['roc_auc']:.4f}")
        ax3.plot([0,1],[0,1],"--",color="#475569",lw=1)
        ax3.fill_between(fpr, tpr, alpha=0.1, color=CYAN)
        ax3.set_title("DistilBERT ROC Curve", color="#e2e8f0", fontsize=10)
        ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
        ax3.legend(framealpha=0.2)
    else:
        ax3.text(0.5,0.5,"Skipped",ha="center",va="center",color="#475569")

    # ── [1,2] RF Confusion Matrix ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    cm_rf = np.array(url_results["rf"]["conf_matrix"])
    sns.heatmap(cm_rf, annot=True, fmt="d", ax=ax4,
                cmap="Greens", linewidths=0.5,
                xticklabels=["Legit","Phish"],
                yticklabels=["Legit","Phish"],
                annot_kws={"size":12,"color":"white"})
    ax4.set_title(f"RandomForest URL\nConfusion Matrix (Test)", color="#e2e8f0", fontsize=10)
    ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")

    # ── [1,3] XGBoost Confusion Matrix ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 3])
    cm_xgb = np.array(url_results["xgb"]["conf_matrix"])
    sns.heatmap(cm_xgb, annot=True, fmt="d", ax=ax5,
                cmap="Oranges", linewidths=0.5,
                xticklabels=["Legit","Phish"],
                yticklabels=["Legit","Phish"],
                annot_kws={"size":12,"color":"white"})
    ax5.set_title(f"XGBoost URL\nConfusion Matrix (Test)", color="#e2e8f0", fontsize=10)
    ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")

    # ── [2,0-1] ROC Curves Comparison ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0:2])
    for name, color, label in [
        ("rf",       GREEN,  "RandomForest"),
        ("xgb",      ORANGE, "XGBoost"),
        ("ensemble", CYAN,   "Ensemble (RF+XGB)"),
    ]:
        r = url_results[name]
        fpr, tpr, _ = roc_curve(r["true"], r["probs"])
        ax6.plot(fpr, tpr, color=color, lw=2,
                 label=f"{label} AUC={r['roc_auc']:.4f}")
    if has_bert:
        fpr_b, tpr_b, _ = roc_curve(bert_metrics["true"], bert_metrics["probs"])
        ax6.plot(fpr_b, tpr_b, color=RED, lw=2,
                 label=f"DistilBERT AUC={bert_metrics['roc_auc']:.4f}")
    ax6.plot([0,1],[0,1],"--",color="#475569",lw=1)
    ax6.set_title("ROC Curves — All Models", color="#e2e8f0", fontsize=11, pad=8)
    ax6.set_xlabel("False Positive Rate"); ax6.set_ylabel("True Positive Rate")
    ax6.legend(framealpha=0.2, fontsize=9)

    # ── [2,2-3] Feature Importances ───────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2:4])
    if "feature_importances" in url_results:
        fi = url_results["feature_importances"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
        names  = [x[0].replace("_"," ") for x in sorted_fi]
        values = [x[1] for x in sorted_fi]
        colors_bar = [RED if v > 0.08 else ORANGE if v > 0.04 else CYAN for v in values]
        bars = ax7.barh(names[::-1], values[::-1], color=colors_bar[::-1], alpha=0.85)
        ax7.set_title("Top 15 Feature Importances (RandomForest URL Classifier)",
                      color="#e2e8f0", fontsize=11, pad=8)
        ax7.set_xlabel("Importance Score")
        for bar, val in zip(bars, values[::-1]):
            ax7.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", ha="left", fontsize=8, color="#94a3b8")

    # ── Summary Text Box ──────────────────────────────────────────────────
    bert_f1  = f"{bert_metrics['f1']:.4f}"  if has_bert else "N/A"
    bert_auc = f"{bert_metrics['roc_auc']:.4f}" if has_bert else "N/A"
    summary = (
        f"Dataset: {email_n:,} emails + {url_n:,} URLs\n"
        f"DistilBERT  F1={bert_f1}  AUC={bert_auc}\n"
        f"RandomForest F1={url_results['rf']['f1']:.4f}  AUC={url_results['rf']['roc_auc']:.4f}\n"
        f"XGBoost      F1={url_results['xgb']['f1']:.4f}  AUC={url_results['xgb']['roc_auc']:.4f}\n"
        f"URL Ensemble F1={url_results['ensemble']['f1']:.4f}  AUC={url_results['ensemble']['roc_auc']:.4f}"
    )
    fig.text(0.01, 0.01, summary, fontsize=9, color="#64748b",
             verticalalignment="bottom", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117",
                       edgecolor="#1e293b", alpha=0.8))

    out_path = str(REPORTS_DIR / "training_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#070b14", edgecolor="none")
    plt.close()
    print(f"       Report saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Save Training Summary JSON
# ══════════════════════════════════════════════════════════════════════════════
def save_summary(bert_metrics, url_results, args):
    import json
    summary = {
        "dataset":        args.dataset,
        "email_samples":  args.email_samples,
        "url_samples":    args.url_samples,
        "epochs":         args.epochs,
        "distilbert": {
            "model":     "distilbert-base-uncased (fine-tuned)",
            "saved_at":  str(MODELS_DIR / "distilbert_email"),
            "accuracy":  bert_metrics.get("accuracy") if bert_metrics else None,
            "f1":        bert_metrics.get("f1")       if bert_metrics else None,
            "precision": bert_metrics.get("precision") if bert_metrics else None,
            "recall":    bert_metrics.get("recall")    if bert_metrics else None,
            "roc_auc":   bert_metrics.get("roc_auc")   if bert_metrics else None,
        } if bert_metrics else {"skipped": True},
        "random_forest": {
            "model":    "RandomForestClassifier (sklearn)",
            "saved_at": str(MODELS_DIR / "url_rf.joblib"),
            "f1":       url_results["rf"]["f1"],
            "roc_auc":  url_results["rf"]["roc_auc"],
        },
        "xgboost": {
            "model":    "XGBClassifier (xgboost)",
            "saved_at": str(MODELS_DIR / "url_xgb.joblib"),
            "f1":       url_results["xgb"]["f1"],
            "roc_auc":  url_results["xgb"]["roc_auc"],
        },
        "url_ensemble": {
            "f1":      url_results["ensemble"]["f1"],
            "roc_auc": url_results["ensemble"]["roc_auc"],
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = str(MODELS_DIR / "training_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"       Summary saved → {path}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    if args.quick:
        args.email_samples = 2000
        args.url_samples   = 5000
        args.epochs        = 1
        print("  [QUICK MODE] email=2000, url=5000, epochs=1")

    total_start = time.time()
    print("\n" + "=" * 62)
    print("  PhishGuard AI — Training Pipeline")
    print("=" * 62)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Email samples: {args.email_samples:,}")
    print(f"  URL samples:   {args.url_samples:,}")
    print(f"  BERT epochs:   {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Max token len: {args.max_len}")
    print("=" * 62)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────
    email_texts, email_labels, url_texts, url_labels = load_dataset(
        args.dataset, args.email_samples, args.url_samples, args.seed
    )

    # ── DistilBERT ─────────────────────────────────────────────────────────
    bert_history, bert_metrics = None, None
    if not args.skip_bert:
        X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
            email_texts, email_labels, seed=args.seed
        )
        bert_history, bert_metrics = train_distilbert(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            epochs=args.epochs, batch_size=args.batch_size,
            max_len=args.max_len, lr=args.lr,
        )
    else:
        print("\n[2/5] DistilBERT skipped (--skip-bert)")

    # ── URL Classifier ─────────────────────────────────────────────────────
    url_results = train_url_classifier(url_texts, url_labels, seed=args.seed)

    # ── Report ─────────────────────────────────────────────────────────────
    generate_report(bert_history, bert_metrics, url_results,
                    len(email_texts), len(url_texts), args.epochs)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[5/5] Saving training summary...")
    save_summary(bert_metrics, url_results, args)

    total = time.time() - total_start
    mins, secs = divmod(int(total), 60)
    print(f"\n{'='*62}")
    print(f"  Training complete in {mins}m {secs}s")
    print(f"  Saved models  → {MODELS_DIR}/")
    print(f"  Saved report  → {REPORTS_DIR}/training_report.png")
    print(f"\n  Next: run the app — it auto-loads your trained models.")
    print(f"        bash launch.sh")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
