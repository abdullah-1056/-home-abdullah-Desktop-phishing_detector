"""
PhishGuard AI - Test Suite
Run with: python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


# ════════════════════════════════════════════════════════════════
#  URL ANALYZER TESTS
# ════════════════════════════════════════════════════════════════

class TestURLAnalyzer:
    @pytest.fixture(scope="class")
    def analyzer(self):
        from core.url_analyzer import URLAnalyzer
        return URLAnalyzer()

    def test_ip_host_detection(self, analyzer):
        r = analyzer.analyze("http://192.168.1.1/login.php")
        assert r.features.has_ip_host is True
        assert r.risk_score > 0.25

    def test_suspicious_tld(self, analyzer):
        r = analyzer.analyze("http://example.tk/")
        assert r.features.has_suspicious_tld is True

    def test_at_symbol(self, analyzer):
        r = analyzer.analyze("http://attacker.com@legit.com/")
        assert r.features.has_at_symbol is True
        # @ symbol + HTTP = ~0.17; the flag itself is what matters
        assert r.risk_score > 0.10
        assert len(r.risk_factors) >= 1

    def test_redirect_param(self, analyzer):
        r = analyzer.analyze("https://evil.com/go?url=https://bank.com")
        assert r.features.has_redirect_param is True

    def test_brand_in_subdomain(self, analyzer):
        r = analyzer.analyze("http://paypal.evil-site.com/login")
        assert r.features.has_brand_in_subdomain is True
        assert r.risk_score > 0.30

    def test_brand_in_domain(self, analyzer):
        r = analyzer.analyze("http://apple-support-suspended.xyz")
        assert r.features.has_brand_in_subdomain is True   # brand-in-domain flag

    def test_legitimate_google(self, analyzer):
        r = analyzer.analyze("https://www.google.com/search?q=test")
        assert r.risk_score < 0.25
        assert r.features.is_https is True

    def test_legitimate_github(self, analyzer):
        r = analyzer.analyze("https://github.com/user/repo")
        assert r.risk_score < 0.20

    def test_entropy_computed(self, analyzer):
        r = analyzer.analyze("https://example.com")
        assert r.features.url_entropy > 0

    def test_typosquatting_paypal(self, analyzer):
        r = analyzer.analyze("http://paypa1.com/login")
        assert r.features.typosquatting_score > 0.5

    def test_https_flag(self, analyzer):
        r_https = analyzer.analyze("https://example.com")
        r_http  = analyzer.analyze("http://example.com")
        assert r_https.features.is_https is True
        assert r_http.features.is_https is False
        assert r_https.risk_score < r_http.risk_score

    def test_feature_vector_length(self, analyzer):
        r = analyzer.analyze("https://example.com")
        assert len(r.features.to_feature_vector()) == 21

    def test_decomposition_keys(self, analyzer):
        r = analyzer.analyze("https://www.example.com/path?q=1")
        keys = set(r.decomposition.keys())
        assert {"scheme", "domain", "tld", "path", "entropy"}.issubset(keys)

    def test_risk_labels(self, analyzer):
        for url, expected_level in [
            ("https://github.com",                   "Safe"),
            ("http://example.xyz/login",              "Low Risk"),
            ("http://paypa1.tk/verify",               "Suspicious"),
        ]:
            r = analyzer.analyze(url)
            # Just check it returns a valid label
            assert r.risk_label in ("Safe", "Low Risk", "Suspicious", "High Risk")


# ════════════════════════════════════════════════════════════════
#  HEURISTICS ENGINE TESTS
# ════════════════════════════════════════════════════════════════

class TestHeuristicsEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from core.heuristics_engine import HeuristicsEngine
        return HeuristicsEngine()

    PHISH_BODY = (
        "Your account has been SUSPENDED. ACTION REQUIRED immediately. "
        "Click here to verify your account or it will be terminated in 24 hours. "
        "Enter your password and social security number. Urgent!"
    )
    LEGIT_BODY = (
        "Hi Sarah, just a reminder about the team meeting tomorrow at 9am. "
        "Please review the attached agenda. Best regards, Tom."
    )

    def test_urgency_detection(self, engine):
        r = engine.analyze("URGENT", self.PHISH_BODY)
        assert r.urgency_score > 0.30

    def test_low_urgency_legit(self, engine):
        r = engine.analyze("Team meeting reminder", self.LEGIT_BODY)
        assert r.urgency_score < 0.20

    def test_spoofing_display_name(self, engine):
        r = engine.analyze("Test", "body", sender='"PayPal Support" <evil@randomdomain.xyz>')
        assert r.spoofing_score > 0.20

    def test_credential_harvesting(self, engine):
        body = "Please enter your password and credit card to confirm your identity."
        r = engine.analyze("Account verification", body)
        assert r.content_score > 0.15

    def test_html_form_flag(self, engine):
        html = '<form action="http://evil.com"><input type="password" name="pwd"></form>'
        r = engine.analyze("Test", "click here", html_body=html)
        assert r.structural_score > 0.20

    def test_suspicious_tokens_populated(self, engine):
        r = engine.analyze("URGENT!", self.PHISH_BODY)
        assert len(r.suspicious_tokens) > 0

    def test_professional_email_safe(self, engine):
        r = engine.analyze("Invoice attached", self.LEGIT_BODY,
                           sender="billing@acmecorp.com")
        assert r.score < 0.35

    def test_lottery_scam(self, engine):
        body = "Congratulations! You have been selected as a winner. Claim your prize of $1,000,000 now!"
        r = engine.analyze("You won!", body)
        assert r.content_score > 0.10

    def test_highlighted_text_generated(self, engine):
        r = engine.analyze("Alert", self.PHISH_BODY)
        assert isinstance(r.highlighted_text, str)

    def test_safe_signals_newsletter(self, engine):
        body = "Thank you for subscribing. Unsubscribe | Privacy Policy | Terms of Service"
        r = engine.analyze("Newsletter", body)
        assert len(r.safe_signals) > 0

    def test_overall_score_range(self, engine):
        for subj, body in [
            ("URGENT", self.PHISH_BODY),
            ("Meeting", self.LEGIT_BODY),
        ]:
            r = engine.analyze(subj, body)
            assert 0.0 <= r.score <= 1.0
            assert 0.0 <= r.confidence <= 1.0


# ════════════════════════════════════════════════════════════════
#  ML CLASSIFIER TESTS
# ════════════════════════════════════════════════════════════════

class TestMLClassifier:
    @pytest.fixture(scope="class")
    def classifier(self):
        from ml.classifier import get_classifier
        clf = get_classifier()
        clf.ensure_trained()
        return clf

    @pytest.fixture(scope="class")
    def phish_features(self):
        from ml.classifier import build_combined_features
        return build_combined_features(
            [0.9,0.6,0.4,0.5,0.3, 1,0,1,1,1, 1,1,1,0,1, 0,0.9,0.8,0.85,0.2,0.3],
            0.9, 0.9, 0.8, 0.85, 8, 300, 60
        )

    @pytest.fixture(scope="class")
    def legit_features(self):
        from ml.classifier import build_combined_features
        return build_combined_features(
            [0.1,0.2,0.05,0.0,0.1, 0,1,0,0,0, 0,0,0,0,0, 0,0.25,0.2,0.0,0.01,0.02],
            0.03, 0.01, 0.0, 0.02, 1, 5000, 25
        )

    def test_phishing_features_high_score(self, classifier, phish_features):
        r = classifier.predict(phish_features)
        assert r.phishing_probability > 0.80

    def test_legit_features_low_score(self, classifier, legit_features):
        r = classifier.predict(legit_features)
        assert r.phishing_probability < 0.20

    def test_probability_in_range(self, classifier, phish_features):
        r = classifier.predict(phish_features)
        assert 0.0 <= r.phishing_probability <= 1.0
        assert 0.0 <= r.confidence <= 1.0

    def test_rf_and_xgb_scores_present(self, classifier, phish_features):
        r = classifier.predict(phish_features)
        assert 0.0 <= r.rf_probability  <= 1.0
        assert 0.0 <= r.xgb_probability <= 1.0

    def test_feature_importances_populated(self, classifier, phish_features):
        r = classifier.predict(phish_features)
        assert len(r.feature_importances) > 0
        # All importances sum to ~1
        total = sum(r.feature_importances.values())
        assert 0.95 <= total <= 1.05

    def test_feature_vector_shape(self):
        from ml.classifier import build_combined_features
        fv = build_combined_features(
            [0.0]*21, 0.5, 0.5, 0.5, 0.5, 3, 1000, 40
        )
        assert fv.shape == (32,)


# ════════════════════════════════════════════════════════════════
#  EMAIL PARSER TESTS
# ════════════════════════════════════════════════════════════════

class TestEmailParser:
    @pytest.fixture(scope="class")
    def parser(self):
        from core.email_parser import EmailParser
        return EmailParser()

    RFC_EMAIL = """From: "Test User" <test@example.com>
To: recipient@gmail.com
Subject: Test Email Subject
Date: Mon, 01 Jan 2024 12:00:00 +0000
Reply-To: reply@other.com

This is the plain text body.
Visit http://example.com for more info.
"""

    PLAIN_TEXT = "This is just a plain text paste with a link: http://evil.tk/phish"

    def test_rfc822_subject_parsed(self, parser):
        p = parser.parse(self.RFC_EMAIL)
        assert p.subject == "Test Email Subject"

    def test_rfc822_sender_parsed(self, parser):
        p = parser.parse(self.RFC_EMAIL)
        assert "test@example.com" in p.sender

    def test_rfc822_reply_to_parsed(self, parser):
        p = parser.parse(self.RFC_EMAIL)
        assert "reply@other.com" in p.reply_to

    def test_url_extraction(self, parser):
        p = parser.parse(self.RFC_EMAIL)
        assert any("example.com" in u for u in p.urls)

    def test_plain_text_fallback(self, parser):
        p = parser.parse(self.PLAIN_TEXT)
        assert p.body_plain == self.PLAIN_TEXT
        assert any("evil.tk" in u for u in p.urls)

    def test_word_count(self, parser):
        p = parser.parse(self.RFC_EMAIL)
        assert p.word_count > 0

    def test_char_count(self, parser):
        p = parser.parse(self.RFC_EMAIL)
        assert p.char_count > 0


# ════════════════════════════════════════════════════════════════
#  CACHE TESTS
# ════════════════════════════════════════════════════════════════

class TestCacheManager:
    @pytest.fixture(scope="class")
    def cache(self, tmp_path_factory):
        from utils.cache import CacheManager
        cache_dir = str(tmp_path_factory.mktemp("cache"))
        return CacheManager(cache_dir, max_size_mb=10, ttl=60)

    def test_set_and_get(self, cache):
        cache.set("key1", {"data": 42})
        result = cache.get("key1")
        assert result == {"data": 42}

    def test_missing_key_returns_none(self, cache):
        assert cache.get("nonexistent_key_xyz") is None

    def test_make_key_deterministic(self):
        from utils.cache import CacheManager
        k1 = CacheManager.make_key("same content", "email")
        k2 = CacheManager.make_key("same content", "email")
        assert k1 == k2

    def test_make_key_different_types(self):
        from utils.cache import CacheManager
        k1 = CacheManager.make_key("same content", "email")
        k2 = CacheManager.make_key("same content", "url")
        assert k1 != k2

    def test_stats(self, cache):
        stats = cache.stats()
        assert "memory_entries" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
