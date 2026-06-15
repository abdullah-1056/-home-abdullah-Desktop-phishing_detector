"""
PhishGuard AI - URL Analyzer
Deep structural analysis: entropy, typosquatting, redirect patterns,
subdomain abuse, unicode obfuscation, IP hosting, and more.
"""
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, parse_qs, unquote

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

try:
    from Levenshtein import distance as levenshtein_distance
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

from config.settings import (
    SUSPICIOUS_TLDS, SUSPICIOUS_URL_KEYWORDS, POPULAR_DOMAINS,
    REDIRECT_PARAMS, MAX_LEGITIMATE_URL_LENGTH, MAX_SUBDOMAIN_COUNT,
    MAX_URL_ENTROPY,
)
from utils.logger import log


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class URLFeatures:
    """Numerical feature vector extracted from a URL."""
    raw_url: str = ""
    scheme: str = ""
    domain: str = ""
    subdomain: str = ""
    tld: str = ""
    path: str = ""
    query: str = ""

    # Length features
    url_length: int = 0
    domain_length: int = 0
    path_length: int = 0
    subdomain_count: int = 0
    query_param_count: int = 0

    # Risk flags
    has_ip_host: bool = False
    is_https: bool = False
    has_at_symbol: bool = False
    has_double_slash: bool = False
    has_redirect_param: bool = False
    has_suspicious_tld: bool = False
    has_suspicious_keyword: bool = False
    has_brand_in_subdomain: bool = False
    has_unicode_obfuscation: bool = False
    has_excessive_dots: bool = False
    has_port: bool = False

    # Computed scores
    url_entropy: float = 0.0
    domain_entropy: float = 0.0
    typosquatting_score: float = 0.0   # 0 = no match, 1 = exact typosquat
    special_char_ratio: float = 0.0
    digit_ratio_in_domain: float = 0.0
    heuristic_score: float = 0.0        # Final 0-1 URL risk

    # Explanations
    risk_factors: list = field(default_factory=list)
    safe_factors: list = field(default_factory=list)

    def to_feature_vector(self) -> list[float]:
        """Return numerical features for ML classifier."""
        return [
            self.url_length / 300,
            self.domain_length / 50,
            self.path_length / 200,
            float(self.subdomain_count) / 5,
            float(self.query_param_count) / 10,
            float(self.has_ip_host),
            float(self.is_https),
            float(self.has_at_symbol),
            float(self.has_double_slash),
            float(self.has_redirect_param),
            float(self.has_suspicious_tld),
            float(self.has_suspicious_keyword),
            float(self.has_brand_in_subdomain),
            float(self.has_unicode_obfuscation),
            float(self.has_excessive_dots),
            float(self.has_port),
            self.url_entropy / 6.0,
            self.domain_entropy / 6.0,
            self.typosquatting_score,
            self.special_char_ratio,
            self.digit_ratio_in_domain,
        ]


@dataclass
class URLAnalysisResult:
    url: str
    features: URLFeatures
    risk_score: float           # 0.0 → 1.0
    risk_label: str             # "Safe" | "Suspicious" | "High Risk"
    confidence: float
    risk_factors: list[str]
    safe_factors: list[str]
    decomposition: dict         # For UI visualization


# ── Core Analyzer ──────────────────────────────────────────────────────────────

class URLAnalyzer:
    """
    Extracts 20+ structural and statistical features from URLs
    and computes a heuristic risk score without any external API calls.
    """

    IP_REGEX = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$"
    )
    PUNYCODE_REGEX = re.compile(r"xn--[a-z0-9]+", re.IGNORECASE)
    HEX_ENCODE_REGEX = re.compile(r"%[0-9A-Fa-f]{2}")

    # Brand names often impersonated in subdomains
    BRAND_NAMES = {
        "paypal", "apple", "google", "amazon", "microsoft", "facebook",
        "netflix", "ebay", "instagram", "twitter", "chase", "wellsfargo",
        "citibank", "bankofamerica", "dropbox", "icloud", "outlook",
    }

    def __init__(self):
        log.debug("URLAnalyzer initialized")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, url: str) -> URLAnalysisResult:
        """Full URL analysis. Returns structured result with score."""
        url = url.strip()
        if not url.startswith(("http://", "https://", "ftp://")):
            url = "http://" + url

        features = self._extract_features(url)
        risk_score, risk_factors, safe_factors = self._compute_risk(features)
        features.heuristic_score = risk_score
        features.risk_factors = risk_factors
        features.safe_factors = safe_factors

        label = self._risk_label(risk_score)
        decomposition = self._decompose(features)

        return URLAnalysisResult(
            url=url,
            features=features,
            risk_score=risk_score,
            risk_label=label,
            confidence=min(0.95, 0.6 + abs(risk_score - 0.5)),
            risk_factors=risk_factors,
            safe_factors=safe_factors,
            decomposition=decomposition,
        )

    # ── Feature Extraction ─────────────────────────────────────────────────────

    def _extract_features(self, url: str) -> URLFeatures:
        f = URLFeatures(raw_url=url)
        parsed = urlparse(url)

        f.scheme  = parsed.scheme
        f.path    = parsed.path
        f.query   = parsed.query
        f.is_https = parsed.scheme == "https"

        # Domain extraction
        netloc = parsed.netloc
        if "@" in netloc:
            f.has_at_symbol = True
            netloc = netloc.split("@")[-1]

        # Port detection
        if ":" in netloc:
            f.has_port = True
            netloc = netloc.split(":")[0]

        # TLD extraction
        if HAS_TLDEXTRACT:
            ext = tldextract.extract(url)
            f.domain    = ext.registered_domain
            f.subdomain = ext.subdomain
            f.tld       = "." + ext.suffix if ext.suffix else ""
        else:
            parts = netloc.split(".")
            f.domain    = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
            f.subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
            f.tld       = "." + parts[-1] if parts else ""

        # Length features
        f.url_length    = len(url)
        f.domain_length = len(f.domain)
        f.path_length   = len(f.path)
        f.query_param_count = len(parse_qs(f.query))

        # Subdomain depth
        f.subdomain_count = len(f.subdomain.split(".")) if f.subdomain else 0

        # IP address as host
        f.has_ip_host = bool(self.IP_REGEX.match(netloc))

        # Double slash in path
        f.has_double_slash = "//" in parsed.path

        # Redirect parameters
        query_lower = f.query.lower()
        f.has_redirect_param = any(p in query_lower for p in REDIRECT_PARAMS)

        # Suspicious TLD
        f.has_suspicious_tld = f.tld.lower() in SUSPICIOUS_TLDS

        # Suspicious keywords in full URL (lower)
        url_lower = url.lower()
        f.has_suspicious_keyword = any(kw in url_lower for kw in SUSPICIOUS_URL_KEYWORDS)

        # Brand impersonation in subdomain OR domain root (e.g. apple-support-suspended.xyz)
        sub_lower    = f.subdomain.lower()
        domain_lower = f.domain.lower().split(".")[0]   # registered domain root only
        f.has_brand_in_subdomain = (
            any(brand in sub_lower    for brand in self.BRAND_NAMES) or
            # Brand in domain root but domain is NOT the brand's own site
            any(
                brand in domain_lower and domain_lower != brand
                for brand in self.BRAND_NAMES
            )
        )

        # Unicode / punycode obfuscation
        f.has_unicode_obfuscation = (
            bool(self.PUNYCODE_REGEX.search(url)) or
            any(ord(c) > 127 for c in url) or
            bool(self.HEX_ENCODE_REGEX.search(url))
        )

        # Excessive dots
        dot_count = url.count(".")
        f.has_excessive_dots = dot_count > 5

        # Entropy calculations
        f.url_entropy    = self._entropy(url)
        f.domain_entropy = self._entropy(f.domain)

        # Special character ratio (excluding alphanumeric and ://)
        special = sum(1 for c in url if not c.isalnum() and c not in "://.-_?=&%")
        f.special_char_ratio = special / max(len(url), 1)

        # Digit ratio in domain
        digits = sum(1 for c in f.domain if c.isdigit())
        f.digit_ratio_in_domain = digits / max(len(f.domain), 1)

        # Typosquatting detection
        f.typosquatting_score = self._typosquatting_score(f.domain)

        return f

    # ── Risk Scoring ──────────────────────────────────────────────────────────

    def _compute_risk(self, f: URLFeatures) -> tuple[float, list[str], list[str]]:
        """Weighted heuristic risk score from features."""
        score = 0.0
        risk  = []
        safe  = []
        weights_used = 0.0

        def add(condition, weight, risk_msg=None, safe_msg=None):
            nonlocal score, weights_used
            weights_used += abs(weight)
            if condition:
                score += weight
                if risk_msg:
                    risk.append(risk_msg)
            else:
                if safe_msg:
                    safe.append(safe_msg)

        # --- High-signal flags ---
        add(f.has_ip_host,            0.20, "IP address used as hostname (hides true domain)")
        add(f.has_at_symbol,          0.12, "'@' symbol in URL (credential harvesting trick)")
        add(f.has_unicode_obfuscation, 0.15, "Unicode/punycode encoding detected (visual spoofing)")
        add(f.has_suspicious_tld,     0.12, f"High-risk TLD '{f.tld}' commonly used in phishing")
        add(f.has_brand_in_subdomain, 0.18, f"Brand name found in subdomain (e.g. paypal.evil.com)")
        add(f.typosquatting_score > 0.6, 0.20, f"Typosquatting detected (domain resembles trusted site)")

        # --- Medium-signal flags ---
        add(f.has_redirect_param,     0.10, "URL redirect parameter detected")
        add(f.has_suspicious_keyword, 0.08, "Phishing keyword found in URL path/query")
        add(f.has_double_slash,       0.06, "Double slash in URL path (obfuscation)")
        add(f.has_port,               0.05, "Non-standard port in URL")
        add(f.has_excessive_dots,     0.06, f"Excessive dots ({f.raw_url.count('.')} — suspicious subdomain nesting)")

        # --- Entropy ---
        if f.url_entropy > MAX_URL_ENTROPY:
            score += 0.08
            weights_used += 0.08
            risk.append(f"High URL entropy ({f.url_entropy:.2f} bits — random-looking, machine-generated)")
        else:
            safe.append(f"Normal URL entropy ({f.url_entropy:.2f} bits)")

        # --- Length ---
        if f.url_length > MAX_LEGITIMATE_URL_LENGTH:
            score += 0.06
            weights_used += 0.06
            risk.append(f"Unusually long URL ({f.url_length} chars — hides true destination)")
        else:
            safe.append(f"URL length is normal ({f.url_length} chars)")

        # --- Subdomain depth ---
        if f.subdomain_count > MAX_SUBDOMAIN_COUNT:
            score += 0.07
            weights_used += 0.07
            risk.append(f"Deep subdomain nesting ({f.subdomain_count} levels)")

        # --- Safe signals ---
        if f.is_https:
            safe.append("HTTPS scheme (encrypted connection)")
        else:
            score += 0.05
            weights_used += 0.05
            risk.append("HTTP only — no encryption, credentials sent in plaintext")

        # --- Domain digit ratio ---
        if f.digit_ratio_in_domain > 0.3:
            score += 0.05
            weights_used += 0.05
            risk.append(f"High digit ratio in domain ({f.digit_ratio_in_domain:.0%})")

        # --- Special char ratio ---
        if f.special_char_ratio > 0.1:
            score += 0.05
            weights_used += 0.05
            risk.append("Excessive special characters in URL")

        # Normalize
        final = min(1.0, max(0.0, score))
        return final, risk, safe

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())

    def _typosquatting_score(self, domain: str) -> float:
        """
        Compute similarity to popular domains.
        Returns 0 if domain IS a popular domain, 1 if one edit away.
        """
        if not domain or not HAS_LEVENSHTEIN:
            return 0.0

        domain_root = domain.split(".")[0].lower() if "." in domain else domain.lower()
        min_score = 0.0

        for popular in POPULAR_DOMAINS:
            popular_root = popular.split(".")[0].lower()

            # Skip if it's the same
            if domain_root == popular_root:
                return 0.0

            dist = levenshtein_distance(domain_root, popular_root)
            # Normalize by max length
            max_len = max(len(domain_root), len(popular_root))
            similarity = 1.0 - (dist / max_len)

            # Only flag if very close (0.6–0.95 similarity = suspicious)
            if 0.60 <= similarity < 0.98:
                min_score = max(min_score, similarity)

        return min_score

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 0.75:
            return "High Risk"
        if score >= 0.45:
            return "Suspicious"
        if score >= 0.20:
            return "Low Risk"
        return "Safe"

    def _decompose(self, f: URLFeatures) -> dict:
        """URL decomposition for UI visualization."""
        return {
            "scheme":    {"value": f.scheme, "risk": not f.is_https},
            "subdomain": {"value": f.subdomain or "(none)", "risk": f.has_brand_in_subdomain or f.subdomain_count > MAX_SUBDOMAIN_COUNT},
            "domain":    {"value": f.domain, "risk": f.typosquatting_score > 0.6 or f.has_ip_host},
            "tld":       {"value": f.tld, "risk": f.has_suspicious_tld},
            "path":      {"value": f.path or "/", "risk": f.has_double_slash},
            "query":     {"value": f.query or "(none)", "risk": f.has_redirect_param or f.has_suspicious_keyword},
            "entropy":   {"value": f"{f.url_entropy:.2f} bits", "risk": f.url_entropy > MAX_URL_ENTROPY},
            "length":    {"value": f"{f.url_length} chars", "risk": f.url_length > MAX_LEGITIMATE_URL_LENGTH},
        }
