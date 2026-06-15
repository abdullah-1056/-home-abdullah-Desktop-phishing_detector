"""
PhishGuard AI - Heuristics Engine
Rule-based phishing detection: urgency language, header anomalies,
HTML forms, suspicious patterns, sender analysis, and content flags.
"""
import re
import math
from dataclasses import dataclass, field
from typing import Optional
from email import message_from_string
from email.header import decode_header

from config.settings import (
    URGENCY_KEYWORDS, PHISHING_CONTENT_PATTERNS, SAFE_SENDER_DOMAINS,
    SUSPICIOUS_URL_KEYWORDS,
)
from utils.logger import log


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class HeuristicResult:
    score: float                          # 0.0 → 1.0
    confidence: float
    flags: list[str] = field(default_factory=list)       # Triggered rules
    safe_signals: list[str] = field(default_factory=list)
    urgency_score: float = 0.0
    spoofing_score: float = 0.0
    content_score: float = 0.0
    structural_score: float = 0.0
    suspicious_tokens: list[str] = field(default_factory=list)
    highlighted_text: str = ""            # HTML with <mark> tags


# ── Engine ────────────────────────────────────────────────────────────────────

class HeuristicsEngine:
    """
    40+ hand-crafted phishing heuristic rules across 4 dimensions:
      1. Urgency & psychological manipulation
      2. Sender/header spoofing
      3. Content & structural patterns
      4. Link & attachment analysis
    """

    # Precompile patterns
    _CONTENT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PHISHING_CONTENT_PATTERNS]

    _HTML_FORM_RE       = re.compile(r"<form\b[^>]*>", re.IGNORECASE)
    _INPUT_RE           = re.compile(r"<input\b[^>]*type=['\"]?(password|text|email)['\"]?", re.IGNORECASE)
    _URL_RE             = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
    _IP_URL_RE          = re.compile(r"https?://(\d{1,3}\.){3}\d{1,3}", re.IGNORECASE)
    _EXECUTABLE_EXT_RE  = re.compile(r"\.(exe|bat|cmd|vbs|js|jar|zip|rar|7z|scr|pif)\b", re.IGNORECASE)
    _CAPS_RATIO_RE      = re.compile(r"[A-Z]")
    _PUNYCODE_RE        = re.compile(r"xn--[a-z0-9]+", re.IGNORECASE)

    _GRAMMAR_ERRORS = [
        re.compile(r"\byour (account|informations?|details?)\s+(is|are|has been)\s+(compromised|hacked|suspended|locked)\b", re.IGNORECASE),
        re.compile(r"\bkindly\s+(click|verify|provide|update)\b", re.IGNORECASE),
        re.compile(r"\bdear\s+(customer|user|member|client|valued)\b", re.IGNORECASE),
        re.compile(r"\bwe\s+(has|have)\s+detected\b", re.IGNORECASE),
        re.compile(r"\bplease\s+to\s+\w+\b", re.IGNORECASE),
    ]

    _SPOOFING_PATTERNS = [
        re.compile(r"noreply@(?!.*\.(com|org|net|edu|gov)$)", re.IGNORECASE),
        re.compile(r"@(?:\d{1,3}\.){3}\d{1,3}", re.IGNORECASE),  # IP-based sender
        re.compile(r"paypal@[\w-]+\.[a-z]{2,}", re.IGNORECASE),    # paypal@notpaypal
    ]

    def __init__(self):
        log.debug("HeuristicsEngine initialized with 40+ rules")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, subject: str, body: str, sender: str = "",
                headers: str = "", html_body: str = "") -> HeuristicResult:
        """Run all heuristic rules and return scored result."""

        result = HeuristicResult(score=0.0, confidence=0.0)
        full_text = f"{subject}\n{body}"
        full_text_lower = full_text.lower()

        # Run all rule groups
        urgency_score   = self._check_urgency(full_text_lower, result)
        spoofing_score  = self._check_spoofing(sender, headers, result)
        content_score   = self._check_content_patterns(body, html_body, result)
        structural_score = self._check_structural(body, html_body, result)
        link_score      = self._check_links(body, html_body, result)
        grammar_score   = self._check_grammar(body, result)

        # Safe signals
        self._check_safe_signals(sender, body, result)

        # Weighted combination
        raw = (
            urgency_score    * 0.30 +
            spoofing_score   * 0.25 +
            content_score    * 0.20 +
            structural_score * 0.10 +
            link_score       * 0.10 +
            grammar_score    * 0.05
        )

        result.score          = min(1.0, max(0.0, raw))
        result.urgency_score  = urgency_score
        result.spoofing_score = spoofing_score
        result.content_score  = content_score
        result.structural_score = structural_score
        result.confidence     = min(0.95, 0.50 + len(result.flags) * 0.05)

        # Generate highlighted HTML
        result.highlighted_text = self._highlight_suspicious(full_text, result.suspicious_tokens)

        return result

    # ── Rule Groups ───────────────────────────────────────────────────────────

    def _check_urgency(self, text_lower: str, result: HeuristicResult) -> float:
        """Detect urgency and psychological manipulation tactics."""
        triggered = []
        for kw in URGENCY_KEYWORDS:
            if kw in text_lower:
                triggered.append(kw)

        score = 0.0
        if triggered:
            # Diminishing returns — more keywords → higher score but capped
            score = min(1.0, 0.15 + len(triggered) * 0.08)
            result.suspicious_tokens.extend(triggered[:5])

            if len(triggered) >= 5:
                result.flags.append(f"Extreme urgency manipulation: {len(triggered)} pressure keywords detected")
            elif len(triggered) >= 3:
                result.flags.append(f"Multiple urgency phrases: '{', '.join(triggered[:3])}'")
            elif triggered:
                result.flags.append(f"Urgency language: '{triggered[0]}'")

        # Exclamation mark abuse
        excl_count = text_lower.count("!")
        if excl_count > 3:
            score = min(1.0, score + 0.1)
            result.flags.append(f"Excessive exclamation marks ({excl_count}) — pressure tactic")

        # ALL CAPS words
        caps_words = [w for w in text_lower.upper().split() if w.isupper() and len(w) > 3]
        if len(caps_words) > 4:
            score = min(1.0, score + 0.08)
            result.flags.append(f"Excessive capitalization: {len(caps_words)} all-caps words")

        return score

    def _check_spoofing(self, sender: str, headers: str, result: HeuristicResult) -> float:
        """Detect sender spoofing, header anomalies, and display name tricks."""
        score = 0.0
        sender_lower = sender.lower()

        # Display name ≠ email domain
        display_match = re.match(r'"?([^"<]+)"?\s*<([^>]+)>', sender)
        if display_match:
            display_name = display_match.group(1).strip().lower()
            email_addr   = display_match.group(2).strip().lower()
            email_domain = email_addr.split("@")[-1] if "@" in email_addr else ""

            # Display name contains a trusted brand but email domain doesn't match
            for brand in ["paypal", "apple", "amazon", "google", "microsoft",
                          "facebook", "netflix", "bank", "chase"]:
                if brand in display_name and brand not in email_domain:
                    score += 0.4
                    result.flags.append(
                        f"Display name spoofing: '{display_match.group(1).strip()}' but email from '{email_domain}'"
                    )
                    result.suspicious_tokens.append(display_match.group(1).strip())
                    break

        # IP-based sender
        if re.search(r"@(\d{1,3}\.){3}\d{1,3}", sender):
            score += 0.3
            result.flags.append("Sender uses IP address instead of domain name")

        # Numeric-heavy or random-looking sender
        local = sender.split("@")[0].replace('"', '').replace('<', '')
        digits = sum(c.isdigit() for c in local)
        if len(local) > 0 and digits / len(local) > 0.5:
            score += 0.15
            result.flags.append(f"Sender local-part is mostly digits ({local})")

        # Headers: Reply-To ≠ From
        if headers:
            from_match  = re.search(r"^From:\s*(.+)$", headers, re.MULTILINE | re.IGNORECASE)
            reply_match = re.search(r"^Reply-To:\s*(.+)$", headers, re.MULTILINE | re.IGNORECASE)
            if from_match and reply_match:
                from_domain  = self._extract_domain(from_match.group(1))
                reply_domain = self._extract_domain(reply_match.group(1))
                if from_domain and reply_domain and from_domain != reply_domain:
                    score += 0.25
                    result.flags.append(
                        f"Reply-To domain ({reply_domain}) differs from From domain ({from_domain})"
                    )

            # Missing authentication markers
            if "dkim-signature" not in headers.lower():
                score += 0.05
                result.flags.append("No DKIM signature in headers (email not cryptographically authenticated)")
            if "received-spf: pass" not in headers.lower() and "spf=pass" not in headers.lower():
                score += 0.05
                result.flags.append("SPF check not confirmed (possible spoofed sender domain)")

        return min(1.0, score)

    def _check_content_patterns(self, body: str, html_body: str, result: HeuristicResult) -> float:
        """Match against known phishing content patterns."""
        score = 0.0
        text = body + " " + html_body

        for pattern in self._CONTENT_PATTERNS:
            match = pattern.search(text)
            if match:
                score += 0.15
                result.flags.append(f"Phishing pattern: '{match.group()[:60]}…'")
                result.suspicious_tokens.append(match.group()[:40])

        # Check for credential-harvesting phrases
        cred_patterns = [
            r"(enter|provide|confirm|verify)\s+(your\s+)?(password|pin|ssn|credit card)",
            r"social\s+security\s+number",
            r"mother'?s?\s+maiden\s+name",
            r"date\s+of\s+birth\s+(and|or)\s+(address|ssn|password)",
        ]
        for pat in cred_patterns:
            if re.search(pat, text, re.IGNORECASE):
                score += 0.20
                result.flags.append("Credential harvesting: requesting sensitive personal data")
                break

        # Lottery / prize scams
        if re.search(r"(won|winner|selected|chosen).{0,30}(prize|reward|gift|lottery|lucky)", text, re.IGNORECASE):
            score += 0.15
            result.flags.append("Prize/lottery scam language detected")

        # Advance fee fraud
        if re.search(r"(million|billion)\s+(dollars?|usd|euros?|pounds?).{0,50}(transfer|fund|share)", text, re.IGNORECASE):
            score += 0.20
            result.flags.append("Advance-fee fraud pattern (Nigerian 419 style)")

        return min(1.0, score)

    def _check_structural(self, body: str, html_body: str, result: HeuristicResult) -> float:
        """Detect structural phishing indicators: forms, obfuscation, layout."""
        score = 0.0

        # HTML form with password/text fields (classic phishing page embedded in email)
        if html_body:
            if self._HTML_FORM_RE.search(html_body):
                score += 0.25
                result.flags.append("HTML form found in email body (credential harvesting form)")

            if self._INPUT_RE.search(html_body):
                score += 0.20
                result.flags.append("Password/text input field in email HTML")

            # Hidden elements (obfuscation)
            if re.search(r'style=["\'][^"\']*display\s*:\s*none', html_body, re.IGNORECASE):
                score += 0.10
                result.flags.append("Hidden HTML elements detected (content obfuscation)")

            # Tiny font (hidden text to fool filters)
            if re.search(r'font-size\s*:\s*[01]px', html_body, re.IGNORECASE):
                score += 0.10
                result.flags.append("Tiny/invisible text detected (spam filter evasion)")

            # Punycode in HTML links
            if self._PUNYCODE_RE.search(html_body):
                score += 0.15
                result.flags.append("Punycode domain in HTML (visual spoofing via Unicode characters)")

        # No body content (empty bait email)
        if len(body.strip()) < 20 and not html_body:
            score += 0.05
            result.flags.append("Near-empty email body (possible reconnaissance or bait)")

        return min(1.0, score)

    def _check_links(self, body: str, html_body: str, result: HeuristicResult) -> float:
        """Analyze all URLs found in the email."""
        score = 0.0
        text = body + " " + html_body

        urls = self._URL_RE.findall(text)
        if not urls:
            result.safe_signals.append("No external URLs detected in email body")
            return 0.0

        # Count suspicious URLs
        suspicious_url_count = 0
        for url in urls[:20]:  # Limit to 20 URLs
            url_lower = url.lower()
            if self._IP_URL_RE.match(url):
                suspicious_url_count += 1
                result.flags.append(f"URL uses IP address: {url[:60]}")
            elif any(kw in url_lower for kw in SUSPICIOUS_URL_KEYWORDS):
                suspicious_url_count += 1

        if suspicious_url_count > 0:
            score += min(0.5, suspicious_url_count * 0.15)
            if suspicious_url_count > 1:
                result.flags.append(f"{suspicious_url_count} suspicious URLs detected in email body")

        # URL count anomaly (too many = spam/phishing)
        if len(urls) > 10:
            score += 0.10
            result.flags.append(f"Unusually high number of URLs ({len(urls)}) — common in phishing")

        # Mismatched anchor text (displayed text ≠ actual URL)
        mismatches = re.findall(
            r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
            html_body, re.IGNORECASE
        )
        for href, display_text in mismatches[:10]:
            display_lower = display_text.strip().lower()
            # If display text looks like a URL but differs from actual href
            if "http" in display_lower:
                display_domain = re.search(r"https?://([^/\s]+)", display_lower)
                href_domain    = re.search(r"https?://([^/\s]+)", href.lower())
                if display_domain and href_domain:
                    if display_domain.group(1) != href_domain.group(1):
                        score += 0.20
                        result.flags.append(
                            f"Anchor text mismatch: displays '{display_domain.group(1)}' links to '{href_domain.group(1)}'"
                        )

        # Executable attachments or downloads in links
        if self._EXECUTABLE_EXT_RE.search(text):
            score += 0.20
            result.flags.append("Link to executable/archive file (.exe, .bat, .zip, etc.)")

        return min(1.0, score)

    def _check_grammar(self, body: str, result: HeuristicResult) -> float:
        """Detect poor grammar patterns common in phishing."""
        score = 0.0
        matches_found = 0

        for pattern in self._GRAMMAR_ERRORS:
            if pattern.search(body):
                matches_found += 1
                result.flags.append(f"Grammar anomaly: '{pattern.pattern[:50]}'")

        if matches_found > 0:
            score = min(0.5, matches_found * 0.15)

        return score

    def _check_safe_signals(self, sender: str, body: str, result: HeuristicResult) -> None:
        """Identify signals that suggest legitimacy."""
        sender_domain = self._extract_domain(sender)

        if sender_domain in SAFE_SENDER_DOMAINS:
            result.safe_signals.append(f"Sender domain '{sender_domain}' is a known email provider")

        if len(body) > 500:
            result.safe_signals.append("Email has substantial content (not a minimal bait)")

        # Professional sign-off patterns
        if re.search(r"(regards|sincerely|best wishes|thank you for|yours truly)", body, re.IGNORECASE):
            result.safe_signals.append("Professional email sign-off detected")

        # Company legal footer
        if re.search(r"(unsubscribe|privacy policy|terms of service|all rights reserved)", body, re.IGNORECASE):
            result.safe_signals.append("Legal/unsubscribe footer present (consistent with legitimate marketing email)")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(email_or_header: str) -> str:
        match = re.search(r"@([a-zA-Z0-9.-]+\.[a-z]{2,})", email_or_header)
        return match.group(1).lower() if match else ""

    @staticmethod
    def _highlight_suspicious(text: str, tokens: list[str]) -> str:
        """Wrap suspicious tokens in <mark> tags for UI rendering."""
        if not tokens:
            return text

        # Sort by length desc to avoid partial replacements
        sorted_tokens = sorted(set(tokens), key=len, reverse=True)
        result = text

        for token in sorted_tokens[:15]:
            if len(token) < 3:
                continue
            escaped = re.escape(token)
            result = re.sub(
                f"({escaped})",
                r'<mark class="phish-highlight">\1</mark>',
                result,
                flags=re.IGNORECASE,
                count=3,
            )

        return result
