"""
PhishGuard AI - Ensemble Coordinator
Runs URL, heuristic, transformer, and ML analyses concurrently
using a thread pool, then combines results with weighted voting.
"""
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import (
    WEIGHT_TRANSFORMER, WEIGHT_HEURISTICS, WEIGHT_URL, WEIGHT_CLASSIFIER,
    THRESHOLD_HIGH, THRESHOLD_MEDIUM, THRESHOLD_LOW, MAX_WORKERS,
)
from core.email_parser import EmailParser, ParsedEmail
from core.heuristics_engine import HeuristicsEngine, HeuristicResult
from core.transformer_engine import TransformerEngine, TransformerResult
from core.url_analyzer import URLAnalyzer, URLAnalysisResult
from ml.classifier import get_classifier, build_combined_features
from utils.logger import log
from utils.cache import get_cache


# ── Result Schema ─────────────────────────────────────────────────────────────

@dataclass
class PhishingAnalysisReport:
    # Input
    input_type: str         # "email" | "url"
    raw_input: str

    # Overall verdict
    phishing_probability: float    # 0.0 → 1.0
    risk_label: str                # "Safe" | "Low Risk" | "Suspicious" | "High Risk"
    confidence: float

    # Per-module scores
    transformer_score:  float = 0.0
    heuristic_score:    float = 0.0
    url_score:          float = 0.0
    classifier_score:   float = 0.0

    # Detailed results
    heuristic_result:    Optional[HeuristicResult]    = None
    transformer_result:  Optional[TransformerResult]  = None
    url_results:         list[URLAnalysisResult]      = field(default_factory=list)
    parsed_email:        Optional[ParsedEmail]         = None

    # Explainability
    all_risk_factors:    list[str] = field(default_factory=list)
    all_safe_factors:    list[str] = field(default_factory=list)
    top_suspicious_urls: list[str] = field(default_factory=list)
    highlighted_body:    str = ""
    analysis_time_ms:    float = 0.0

    # Component weights actually used
    weights_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable summary for logging / caching."""
        return {
            "input_type":            self.input_type,
            "phishing_probability":  self.phishing_probability,
            "risk_label":            self.risk_label,
            "confidence":            self.confidence,
            "transformer_score":     self.transformer_score,
            "heuristic_score":       self.heuristic_score,
            "url_score":             self.url_score,
            "classifier_score":      self.classifier_score,
            "risk_factors":          self.all_risk_factors,
            "safe_factors":          self.all_safe_factors,
            "analysis_time_ms":      self.analysis_time_ms,
        }


# ── Ensemble Engine ────────────────────────────────────────────────────────────

class EnsembleDetector:
    """
    Main entry point for phishing detection.
    Orchestrates all analysis modules with concurrent execution.
    """

    def __init__(self):
        self._parser      = EmailParser()
        self._heuristics  = HeuristicsEngine()
        self._transformer = TransformerEngine()
        self._url_analyzer = URLAnalyzer()
        self._classifier  = get_classifier()
        self._cache       = get_cache()

        log.info("EnsembleDetector initialized")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze_email(self, raw_email: str) -> PhishingAnalysisReport:
        """
        Full email analysis pipeline.
        Runs heuristics, transformer, URL, and ML in parallel.
        """
        t0 = time.perf_counter()

        # Cache check
        cache_key = self._cache.make_key(raw_email, "email")
        cached = self._cache.get(cache_key)
        if cached:
            log.debug("Cache hit for email analysis")
            return cached

        # Parse email
        parsed = self._parser.parse(raw_email)
        body   = parsed.body_plain or parsed.body_html or raw_email
        subject = parsed.subject

        # ── Concurrent analysis ────────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

            # Submit all tasks
            fut_heuristic = pool.submit(
                self._heuristics.analyze,
                subject, body, parsed.sender,
                parsed.headers_raw, parsed.body_html,
            )
            fut_transformer = pool.submit(
                self._transformer.analyze, body, subject
            )
            fut_urls = pool.submit(
                self._analyze_urls, parsed.urls[:10]   # Limit to top 10 URLs
            )

            # Gather results (with timeouts)
            try:
                h_result: HeuristicResult = fut_heuristic.result(timeout=30)
            except Exception as e:
                log.warning(f"Heuristics failed: {e}")
                from core.heuristics_engine import HeuristicResult
                h_result = HeuristicResult(score=0.5, confidence=0.5)

            try:
                t_result: TransformerResult = fut_transformer.result(timeout=60)
            except Exception as e:
                log.warning(f"Transformer failed: {e}")
                from core.transformer_engine import TransformerResult
                t_result = TransformerResult(
                    phishing_probability=0.5, confidence=0.5,
                    embedding_score=0.5, zero_shot_score=0.5,
                    model_used="unavailable", inference_time_ms=0,
                    top_tokens=[], explanation="Model unavailable",
                )

            try:
                url_results: list[URLAnalysisResult] = fut_urls.result(timeout=15)
            except Exception as e:
                log.warning(f"URL analysis failed: {e}")
                url_results = []

        # ── ML Classifier ──────────────────────────────────────────────────
        url_feature_vec = self._aggregate_url_features(url_results)
        combined_features = build_combined_features(
            url_features_vector=url_feature_vec,
            heuristic_score=h_result.score,
            urgency_score=h_result.urgency_score,
            spoofing_score=h_result.spoofing_score,
            content_score=h_result.content_score,
            link_count=len(parsed.urls),
            body_length=parsed.char_count,
            subject_length=len(parsed.subject),
        )

        try:
            clf_result = self._classifier.predict(combined_features)
            clf_score  = clf_result.phishing_probability
        except Exception as e:
            log.warning(f"Classifier failed: {e}")
            clf_score = (h_result.score + t_result.phishing_probability) / 2

        # ── Ensemble Combination ──────────────────────────────────────────
        url_score = np.mean([r.risk_score for r in url_results]) if url_results else 0.0

        final_score = (
            WEIGHT_TRANSFORMER * t_result.phishing_probability +
            WEIGHT_HEURISTICS  * h_result.score +
            WEIGHT_URL         * float(url_score) +
            WEIGHT_CLASSIFIER  * clf_score
        )
        final_score = float(np.clip(final_score, 0.0, 1.0))

        # Confidence from agreement across modules
        scores = [t_result.phishing_probability, h_result.score, float(url_score), clf_score]
        spread = float(np.std(scores))
        final_confidence = float(np.clip(0.75 + abs(final_score - 0.5) * 0.4 - spread * 0.3, 0.40, 0.97))

        # ── Collate risk factors ──────────────────────────────────────────
        risk_factors = []
        safe_factors = []

        if t_result.phishing_probability > 0.5:
            risk_factors.append(
                f"🤖 AI Model: {t_result.phishing_probability:.0%} phishing probability "
                f"(embedding similarity + zero-shot NLI)"
            )
        else:
            safe_factors.append(
                f"🤖 AI Model: {t_result.phishing_probability:.0%} phishing probability (leaning legitimate)"
            )

        risk_factors.extend([f"⚠ {f}" for f in h_result.flags])
        safe_factors.extend([f"✓ {s}" for s in h_result.safe_signals])

        for r in url_results:
            risk_factors.extend([f"🔗 URL: {rf}" for rf in r.risk_factors])
            safe_factors.extend([f"🔗 URL: {sf}" for sf in r.safe_factors])

        # Top suspicious URLs sorted by risk score
        sorted_urls = sorted(url_results, key=lambda r: r.risk_score, reverse=True)
        top_urls = [r.url for r in sorted_urls[:3] if r.risk_score > 0.3]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        report = PhishingAnalysisReport(
            input_type="email",
            raw_input=raw_email[:500],      # Truncate for storage
            phishing_probability=round(final_score, 4),
            risk_label=self._risk_label(final_score),
            confidence=round(final_confidence, 4),
            transformer_score=round(t_result.phishing_probability, 4),
            heuristic_score=round(h_result.score, 4),
            url_score=round(float(url_score), 4),
            classifier_score=round(clf_score, 4),
            heuristic_result=h_result,
            transformer_result=t_result,
            url_results=url_results,
            parsed_email=parsed,
            all_risk_factors=risk_factors[:20],
            all_safe_factors=safe_factors[:10],
            top_suspicious_urls=top_urls,
            highlighted_body=h_result.highlighted_text,
            analysis_time_ms=round(elapsed_ms, 1),
            weights_used={
                "transformer": WEIGHT_TRANSFORMER,
                "heuristics":  WEIGHT_HEURISTICS,
                "url":         WEIGHT_URL,
                "classifier":  WEIGHT_CLASSIFIER,
            },
        )

        # Cache result
        self._cache.set(cache_key, report)

        log.info(
            f"Email analyzed in {elapsed_ms:.0f}ms — "
            f"risk={final_score:.2%} [{report.risk_label}] "
            f"(T={t_result.phishing_probability:.2%} H={h_result.score:.2%} "
            f"U={url_score:.2%} C={clf_score:.2%})"
        )

        return report

    def analyze_url(self, url: str) -> PhishingAnalysisReport:
        """
        URL-only analysis (no email body needed).
        """
        t0 = time.perf_counter()

        cache_key = self._cache.make_key(url, "url")
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        url_result = self._url_analyzer.analyze(url)
        url_score  = url_result.risk_score

        # Transformer on URL text (limited signal but adds value)
        try:
            t_result = self._transformer.analyze(url)
            t_score  = t_result.phishing_probability * 0.4  # Lower weight for URL-only
        except Exception:
            t_result = None
            t_score  = 0.0

        final_score = 0.75 * url_score + 0.25 * t_score
        final_score = float(np.clip(final_score, 0.0, 1.0))
        confidence  = float(np.clip(0.60 + abs(final_score - 0.5) * 0.5, 0.40, 0.95))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        report = PhishingAnalysisReport(
            input_type="url",
            raw_input=url,
            phishing_probability=round(final_score, 4),
            risk_label=self._risk_label(final_score),
            confidence=round(confidence, 4),
            transformer_score=round(t_score, 4),
            heuristic_score=0.0,
            url_score=round(url_score, 4),
            classifier_score=0.0,
            transformer_result=t_result,
            url_results=[url_result],
            all_risk_factors=[f"🔗 {rf}" for rf in url_result.risk_factors],
            all_safe_factors=[f"✓ {sf}" for sf in url_result.safe_factors],
            top_suspicious_urls=[url] if url_score > 0.3 else [],
            analysis_time_ms=round(elapsed_ms, 1),
        )

        self._cache.set(cache_key, report)
        log.info(f"URL analyzed in {elapsed_ms:.0f}ms — risk={final_score:.2%} [{report.risk_label}]")
        return report

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _analyze_urls(self, urls: list[str]) -> list[URLAnalysisResult]:
        """Analyze multiple URLs concurrently."""
        if not urls:
            return []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(urls))) as pool:
            futures = {pool.submit(self._url_analyzer.analyze, url): url for url in urls}
            for fut in concurrent.futures.as_completed(futures, timeout=10):
                try:
                    results.append(fut.result())
                except Exception as e:
                    log.debug(f"URL analysis error: {e}")
        return results

    @staticmethod
    def _aggregate_url_features(url_results: list[URLAnalysisResult]) -> list[float]:
        """Aggregate URL feature vectors (mean across all URLs found)."""
        if not url_results:
            return [0.0] * 21   # Zero vector — 21 URL features

        vectors = [r.features.to_feature_vector() for r in url_results]
        arr = np.array(vectors)

        # Use max (worst-case) for risk features, mean for continuous
        result = np.mean(arr, axis=0).tolist()
        # Override with max for binary risk flags (indices 5-15)
        for i in range(5, 16):
            result[i] = float(np.max(arr[:, i]))

        return result

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= THRESHOLD_HIGH:
            return "High Risk"
        if score >= THRESHOLD_MEDIUM:
            return "Suspicious"
        if score >= THRESHOLD_LOW:
            return "Low Risk"
        return "Safe"


# ── Module-level singleton ─────────────────────────────────────────────────────
_detector: Optional[EnsembleDetector] = None


def get_detector() -> EnsembleDetector:
    global _detector
    if _detector is None:
        _detector = EnsembleDetector()
    return _detector
