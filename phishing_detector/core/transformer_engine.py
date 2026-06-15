"""
PhishGuard AI - Transformer Engine
Uses fine-tuned DistilBERT if available (from train.py).
Falls back to zero-shot + embedding similarity if not.
"""
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from config.settings import (
    EMBEDDING_MODEL, ZERO_SHOT_MODEL, MODEL_CACHE_DIR,
    MAX_TOKEN_LENGTH, ZS_CANDIDATE_LABELS,
    PHISHING_REFERENCE_TEXTS, LEGITIMATE_REFERENCE_TEXTS,
)
from utils.logger import log

MODELS_DIR       = Path(__file__).resolve().parent.parent / "models"
FINETUNED_PATH   = MODELS_DIR / "distilbert_email"


@dataclass
class TransformerResult:
    phishing_probability: float
    confidence: float
    embedding_score: float
    zero_shot_score: float
    model_used: str
    inference_time_ms: float
    top_tokens: list
    explanation: str
    fine_tuned: bool = False


class _ModelManager:
    def __init__(self):
        self._lock              = threading.Lock()
        self._finetuned_model   = None
        self._finetuned_tok     = None
        self._embedding_model   = None
        self._zs_pipeline       = None
        self._phishing_centroids = None
        self._legit_centroids    = None
        self._loaded             = False
        self._use_finetuned      = False

    def ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_all()

    def _load_all(self):
        log.info("Loading transformer models...")
        t0 = time.time()

        # ── Priority 1: Fine-tuned DistilBERT from train.py ───────────────
        if FINETUNED_PATH.exists():
            try:
                from transformers import (DistilBertForSequenceClassification,
                                          DistilBertTokenizerFast)
                log.info(f"  Loading fine-tuned DistilBERT from {FINETUNED_PATH}")
                self._finetuned_tok   = DistilBertTokenizerFast.from_pretrained(str(FINETUNED_PATH))
                self._finetuned_model = DistilBertForSequenceClassification.from_pretrained(
                    str(FINETUNED_PATH)
                )
                self._finetuned_model.eval()
                self._use_finetuned = True
                log.info("  Fine-tuned DistilBERT ready (trained on real phishing data)")
            except Exception as e:
                log.warning(f"  Could not load fine-tuned model: {e}")
                self._use_finetuned = False

        # ── Priority 2: Sentence embeddings (always load as backup) ───────
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"  Loading embedding model: {EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(
                EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR
            )
            phish_embs = self._embedding_model.encode(
                PHISHING_REFERENCE_TEXTS, batch_size=12,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            legit_embs = self._embedding_model.encode(
                LEGITIMATE_REFERENCE_TEXTS, batch_size=12,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            self._phishing_centroids = phish_embs
            self._legit_centroids    = legit_embs
            log.info("  Embedding model ready")
        except Exception as e:
            log.warning(f"  Embedding model failed: {e}")

        # ── Priority 3: Zero-shot NLI (only if no fine-tuned model) ───────
        if not self._use_finetuned:
            try:
                from transformers import pipeline
                log.info(f"  Loading zero-shot model: {ZERO_SHOT_MODEL}")
                self._zs_pipeline = pipeline(
                    "zero-shot-classification",
                    model=ZERO_SHOT_MODEL,
                    device=-1,
                    model_kwargs={"cache_dir": MODEL_CACHE_DIR},
                )
                self._zs_pipeline("warm up", candidate_labels=["phishing", "legitimate"])
                log.info("  Zero-shot model ready")
            except Exception as e:
                log.warning(f"  Zero-shot model unavailable: {e}")

        elapsed = time.time() - t0
        mode = "fine-tuned DistilBERT" if self._use_finetuned else "zero-shot + embeddings"
        log.info(f"Models loaded in {elapsed:.1f}s — mode: {mode}")
        self._loaded = True

    @property
    def use_finetuned(self): return self._use_finetuned
    @property
    def finetuned_model(self): self.ensure_loaded(); return self._finetuned_model
    @property
    def finetuned_tok(self):   self.ensure_loaded(); return self._finetuned_tok
    @property
    def embedding_model(self): self.ensure_loaded(); return self._embedding_model
    @property
    def zs_pipeline(self):     self.ensure_loaded(); return self._zs_pipeline
    @property
    def phishing_centroids(self): self.ensure_loaded(); return self._phishing_centroids
    @property
    def legit_centroids(self):    self.ensure_loaded(); return self._legit_centroids


_manager = _ModelManager()


def preload_models():
    t = threading.Thread(target=_manager.ensure_loaded, daemon=True, name="ModelLoader")
    t.start()
    return t


class TransformerEngine:
    def __init__(self):
        self._m = _manager

    def analyze(self, text: str, subject: str = "") -> TransformerResult:
        t0       = time.perf_counter()
        combined = self._prepare_input(subject, text)

        if self._m.use_finetuned and self._m.finetuned_model is not None:
            score, confidence = self._finetuned_score(combined)
            emb_score = self._embedding_score(combined) if self._m.embedding_model else score
            # Weighted: 70% fine-tuned, 30% embeddings
            final = 0.70 * score + 0.30 * emb_score
            model_used  = "DistilBERT (fine-tuned on real data) + MiniLM embeddings"
            fine_tuned  = True
        else:
            emb_score = self._embedding_score(combined) if self._m.embedding_model else 0.5
            zs_score  = self._zero_shot_score(combined) if self._m.zs_pipeline else 0.5
            final       = 0.40 * emb_score + 0.60 * zs_score
            confidence  = min(0.90, 0.50 + abs(final - 0.5) * 0.40)
            model_used  = f"{EMBEDDING_MODEL} + {ZERO_SHOT_MODEL} (zero-shot)"
            fine_tuned  = False
            emb_score   = emb_score
            zs_score    = zs_score

        final = float(np.clip(final, 0.0, 1.0))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return TransformerResult(
            phishing_probability=round(final, 4),
            confidence=round(min(0.97, 0.55 + abs(final-0.5)*0.80), 4),
            embedding_score=round(emb_score if not self._m.use_finetuned else score, 4),
            zero_shot_score=round(zs_score if not self._m.use_finetuned else score, 4),
            model_used=model_used,
            inference_time_ms=round(elapsed_ms, 1),
            top_tokens=self._extract_suspicious_tokens(combined),
            explanation=self._explanation(final, model_used),
            fine_tuned=fine_tuned,
        )

    def _finetuned_score(self, text: str):
        """Run fine-tuned DistilBERT classifier."""
        import torch
        tok   = self._m.finetuned_tok
        model = self._m.finetuned_model
        max_len = MAX_TOKEN_LENGTH

        inputs = tok(text, truncation=True, padding=True,
                     max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=1)[0]
            phish_prob = float(probs[1])
            confidence = float(probs.max())
        return phish_prob, confidence

    def _embedding_score(self, text: str) -> float:
        try:
            model = self._m.embedding_model
            query = model.encode([text], convert_to_numpy=True,
                                 normalize_embeddings=True, batch_size=1)[0]
            phish_sims = self._m.phishing_centroids @ query
            legit_sims  = self._m.legit_centroids   @ query
            ps = 0.7*float(np.mean(phish_sims)) + 0.3*float(np.max(phish_sims))
            ls = 0.7*float(np.mean(legit_sims))  + 0.3*float(np.max(legit_sims))
            ep, el = np.exp(ps*5), np.exp(ls*5)
            return float(np.clip(ep/(ep+el), 0, 1))
        except Exception:
            return 0.5

    def _zero_shot_score(self, text: str) -> float:
        try:
            result = self._m.zs_pipeline(text, candidate_labels=ZS_CANDIDATE_LABELS,
                                          hypothesis_template="{}")
            return float(dict(zip(result["labels"], result["scores"])).get("phishing", 0.5))
        except Exception:
            return 0.5

    @staticmethod
    def _prepare_input(subject: str, body: str) -> str:
        combined = (f"Subject: {subject}\n\n" if subject else "") + body
        return combined[:MAX_TOKEN_LENGTH * 4].strip()

    @staticmethod
    def _extract_suspicious_tokens(text: str) -> list:
        import re
        patterns = [
            r"\b(verify|confirm|update|validate)\s+your\s+\w+",
            r"\b(account|password)\s+(has\s+been|will\s+be)\s+\w+",
            r"\b(click\s+here|click\s+below|click\s+the\s+link)\b",
            r"\b(urgent|immediate|warning|alert|action\s+required)\b",
            r"\b(suspended|locked|closed|terminated|compromised)\b",
            r"https?://[^\s]{10,50}",
        ]
        found = []
        for p in patterns:
            for m in re.findall(p, text, re.IGNORECASE):
                found.append(m if isinstance(m, str) else m[0])
        return list(dict.fromkeys(found))[:10]

    @staticmethod
    def _explanation(score: float, model: str) -> str:
        level = ("HIGH RISK" if score > 0.75 else "SUSPICIOUS" if score > 0.45
                 else "LOW RISK" if score > 0.20 else "SAFE")
        return f"[{level}] Phishing probability: {score:.1%} | Model: {model}"
