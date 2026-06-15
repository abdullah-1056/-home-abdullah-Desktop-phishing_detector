"""
PhishGuard AI - ML Classifier
Loads trained models from disk if available (trained by train.py).
Falls back to synthetic training if no saved models found.
"""
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

from utils.logger import log

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_combined_features(
    url_features_vector: list,
    heuristic_score: float,
    urgency_score: float,
    spoofing_score: float,
    content_score: float,
    link_count: int,
    body_length: int,
    subject_length: int,
) -> np.ndarray:
    email_features = [
        heuristic_score,
        urgency_score,
        spoofing_score,
        content_score,
        min(1.0, link_count / 20.0),
        min(1.0, body_length / 5000.0),
        min(1.0, subject_length / 200.0),
        float(urgency_score > 0.5),
        float(spoofing_score > 0.3),
        float(content_score > 0.4),
        float(link_count > 5),
    ]
    combined = list(url_features_vector) + email_features
    return np.array(combined, dtype=np.float32)


def _generate_synthetic_data(n_phish=800, n_legit=800, seed=42):
    rng = np.random.default_rng(seed)
    phish_url = np.column_stack([
        rng.uniform(0.4,1.0,n_phish), rng.uniform(0.2,0.8,n_phish),
        rng.uniform(0.1,0.6,n_phish), rng.uniform(0.2,0.8,n_phish),
        rng.uniform(0.0,0.5,n_phish),
        rng.binomial(1,0.30,n_phish).astype(float),
        rng.binomial(1,0.35,n_phish).astype(float),
        rng.binomial(1,0.25,n_phish).astype(float),
        rng.binomial(1,0.20,n_phish).astype(float),
        rng.binomial(1,0.45,n_phish).astype(float),
        rng.binomial(1,0.55,n_phish).astype(float),
        rng.binomial(1,0.75,n_phish).astype(float),
        rng.binomial(1,0.40,n_phish).astype(float),
        rng.binomial(1,0.20,n_phish).astype(float),
        rng.binomial(1,0.35,n_phish).astype(float),
        rng.binomial(1,0.15,n_phish).astype(float),
        rng.uniform(0.55,1.0,n_phish), rng.uniform(0.4,1.0,n_phish),
        rng.uniform(0.3,0.9,n_phish),  rng.uniform(0.05,0.3,n_phish),
        rng.uniform(0.1,0.5,n_phish),
    ])
    phish_email = np.column_stack([
        rng.uniform(0.35,0.95,n_phish), rng.uniform(0.40,1.00,n_phish),
        rng.uniform(0.20,0.90,n_phish), rng.uniform(0.30,0.95,n_phish),
        rng.uniform(0.10,0.80,n_phish), rng.uniform(0.01,0.50,n_phish),
        rng.uniform(0.05,0.50,n_phish),
        rng.binomial(1,0.70,n_phish).astype(float),
        rng.binomial(1,0.55,n_phish).astype(float),
        rng.binomial(1,0.65,n_phish).astype(float),
        rng.binomial(1,0.50,n_phish).astype(float),
    ])
    X_phish = np.hstack([phish_url, phish_email])

    legit_url = np.column_stack([
        rng.uniform(0.0,0.35,n_legit), rng.uniform(0.1,0.4,n_legit),
        rng.uniform(0.0,0.3,n_legit),  rng.uniform(0.0,0.2,n_legit),
        rng.uniform(0.0,0.2,n_legit),
        rng.binomial(1,0.02,n_legit).astype(float),
        rng.binomial(1,0.85,n_legit).astype(float),
        rng.binomial(1,0.01,n_legit).astype(float),
        rng.binomial(1,0.01,n_legit).astype(float),
        rng.binomial(1,0.05,n_legit).astype(float),
        rng.binomial(1,0.05,n_legit).astype(float),
        rng.binomial(1,0.10,n_legit).astype(float),
        rng.binomial(1,0.02,n_legit).astype(float),
        rng.binomial(1,0.02,n_legit).astype(float),
        rng.binomial(1,0.05,n_legit).astype(float),
        rng.binomial(1,0.02,n_legit).astype(float),
        rng.uniform(0.2,0.55,n_legit), rng.uniform(0.15,0.45,n_legit),
        rng.uniform(0.0,0.15,n_legit), rng.uniform(0.0,0.05,n_legit),
        rng.uniform(0.0,0.1,n_legit),
    ])
    legit_email = np.column_stack([
        rng.uniform(0.0,0.30,n_legit), rng.uniform(0.0,0.20,n_legit),
        rng.uniform(0.0,0.10,n_legit), rng.uniform(0.0,0.20,n_legit),
        rng.uniform(0.0,0.25,n_legit), rng.uniform(0.1,0.80,n_legit),
        rng.uniform(0.05,0.30,n_legit),
        rng.binomial(1,0.05,n_legit).astype(float),
        rng.binomial(1,0.03,n_legit).astype(float),
        rng.binomial(1,0.05,n_legit).astype(float),
        rng.binomial(1,0.10,n_legit).astype(float),
    ])
    X_legit = np.hstack([legit_url, legit_email])

    X_phish += rng.normal(0, 0.03, X_phish.shape)
    X_legit  += rng.normal(0, 0.03, X_legit.shape)
    X_phish  = np.clip(X_phish, 0.0, 1.0)
    X_legit  = np.clip(X_legit, 0.0, 1.0)

    X = np.vstack([X_phish, X_legit])
    y = np.hstack([np.ones(n_phish), np.zeros(n_legit)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


@dataclass
class ClassifierResult:
    phishing_probability: float
    confidence: float
    rf_probability: float
    xgb_probability: float
    feature_importances: dict
    trained_on_real_data: bool = False


class EnsembleClassifier:
    FEATURE_NAMES = [
        "url_length","domain_length","path_length","subdomain_count",
        "query_param_count","has_ip_host","is_https","has_at_symbol",
        "has_double_slash","has_redirect_param","has_suspicious_tld",
        "has_suspicious_keyword","has_brand_in_subdomain","has_unicode",
        "has_excessive_dots","has_port","url_entropy","domain_entropy",
        "typosquatting_score","special_char_ratio","digit_ratio_domain",
        "heuristic_score","urgency_score","spoofing_score","content_score",
        "link_count_norm","body_length_norm","subject_length_norm",
        "urgency_flag","spoofing_flag","content_flag","link_count_flag",
    ]

    def __init__(self):
        self._rf  = None
        self._xgb = None
        self._trained = False
        self._real_data = False
        self._lock = threading.Lock()

    def ensure_trained(self):
        if self._trained:
            return
        with self._lock:
            if self._trained:
                return
            # Try loading real trained models first
            if self._load_from_disk():
                return
            # Fall back to synthetic training
            self._train_synthetic()

    def _load_from_disk(self) -> bool:
        """Load RF + XGBoost trained by train.py on real data."""
        rf_path  = MODELS_DIR / "url_rf.joblib"
        xgb_path = MODELS_DIR / "url_xgb.joblib"
        if not rf_path.exists() or not xgb_path.exists():
            return False
        try:
            import joblib
            self._rf  = joblib.load(rf_path)
            self._xgb = joblib.load(xgb_path)
            self._trained    = True
            self._real_data  = True
            log.info(f"ML classifier loaded from disk (trained on real dataset)")
            return True
        except Exception as e:
            log.warning(f"Could not load saved models: {e}")
            return False

    def _train_synthetic(self):
        log.info("Training ML ensemble on synthetic data (run train.py for real-data models)...")
        t0 = time.time()
        X, y = _generate_synthetic_data(800, 800)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        rf_base = RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
        self._rf = CalibratedClassifierCV(rf_base, cv=3, method="isotonic")
        self._rf.fit(X, y)
        try:
            from xgboost import XGBClassifier
            self._xgb = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0,
                random_state=42, n_jobs=-1,
            )
            self._xgb.fit(X, y)
        except Exception:
            self._xgb = None
        self._trained   = True
        self._real_data = False
        log.info(f"Synthetic ML ensemble trained in {time.time()-t0:.2f}s")

    def predict(self, feature_vector: np.ndarray) -> ClassifierResult:
        self.ensure_trained()
        X = feature_vector.reshape(1, -1)

        # Real-data models only have URL features (21-dim) — use first 21 dims
        if self._real_data:
            X_url = X[:, :21]
            rf_prob  = float(self._rf.predict_proba(X_url)[0, 1])
            xgb_prob = float(self._xgb.predict_proba(X_url)[0, 1]) if self._xgb else rf_prob
        else:
            rf_prob  = float(self._rf.predict_proba(X)[0, 1])
            xgb_prob = float(self._xgb.predict_proba(X)[0, 1]) if self._xgb else rf_prob

        final = 0.55 * rf_prob + 0.45 * xgb_prob

        try:
            if self._real_data:
                importances = {}
            else:
                base_rf = self._rf.calibrated_classifiers_[0].estimator
                importances = dict(zip(
                    self.FEATURE_NAMES,
                    [round(float(v), 4) for v in base_rf.feature_importances_]
                ))
        except Exception:
            importances = {}

        confidence = min(0.95, 0.55 + abs(final - 0.5) * 0.7)

        return ClassifierResult(
            phishing_probability=round(final, 4),
            confidence=round(confidence, 4),
            rf_probability=round(rf_prob, 4),
            xgb_probability=round(xgb_prob, 4),
            feature_importances=importances,
            trained_on_real_data=self._real_data,
        )


_classifier = EnsembleClassifier()

def get_classifier() -> EnsembleClassifier:
    return _classifier

def preload_classifier():
    t = threading.Thread(target=_classifier.ensure_trained, daemon=True, name="ClassifierLoader")
    t.start()
    return t
