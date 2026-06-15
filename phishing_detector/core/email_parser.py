"""
PhishGuard AI - Email Parser
Parses raw email text (plain text, RFC-822, EML format) and
extracts structured fields for analysis.
"""
import re
import html
from dataclasses import dataclass, field
from email import message_from_string
from email.header import decode_header as _decode_header
from typing import Optional

from utils.logger import log


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ParsedEmail:
    raw: str
    subject: str = ""
    sender: str = ""
    reply_to: str = ""
    recipients: list[str] = field(default_factory=list)
    date: str = ""
    message_id: str = ""
    headers_raw: str = ""
    body_plain: str = ""
    body_html: str = ""
    urls: list[str] = field(default_factory=list)
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)
    is_multipart: bool = False
    char_count: int = 0
    word_count: int = 0


# ── Parser ─────────────────────────────────────────────────────────────────────

class EmailParser:
    """
    Parses RFC-822 / plain-text email input.
    Falls back gracefully when input is plain text only (no headers).
    """

    _URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
    _ANCHOR_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

    def parse(self, raw_input: str) -> ParsedEmail:
        """
        Parse email from raw string.
        Accepts:
          - Full RFC-822 format (with From:, To:, Subject:, headers)
          - Plain text / paste (treated as body only)
        """
        parsed = ParsedEmail(raw=raw_input)

        if self._looks_like_email(raw_input):
            self._parse_rfc822(raw_input, parsed)
        else:
            # Plain text paste — treat entire input as body
            parsed.body_plain = raw_input
            log.debug("Input treated as plain text (no RFC-822 structure detected)")

        # Post-process
        parsed.urls      = self._extract_urls(parsed.body_plain, parsed.body_html)
        parsed.char_count = len(parsed.body_plain)
        parsed.word_count = len(parsed.body_plain.split())

        return parsed

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _looks_like_email(text: str) -> bool:
        """Heuristic: does this look like a properly formatted email?"""
        lines = text.strip().split("\n")[:10]
        header_patterns = ["from:", "to:", "subject:", "date:", "message-id:", "mime-version:"]
        matches = sum(1 for line in lines if any(line.lower().startswith(p) for p in header_patterns))
        return matches >= 2

    def _parse_rfc822(self, raw: str, parsed: ParsedEmail):
        try:
            msg = message_from_string(raw)
        except Exception as e:
            log.warning(f"RFC-822 parse failed: {e}")
            parsed.body_plain = raw
            return

        parsed.subject    = self._decode_header_value(msg.get("Subject", ""))
        parsed.sender     = self._decode_header_value(msg.get("From", ""))
        parsed.reply_to   = self._decode_header_value(msg.get("Reply-To", ""))
        parsed.date       = msg.get("Date", "")
        parsed.message_id = msg.get("Message-ID", "")

        to_header = msg.get("To", "")
        parsed.recipients = [r.strip() for r in to_header.split(",") if r.strip()]

        # Reconstruct headers string
        header_lines = []
        for key, val in msg.items():
            header_lines.append(f"{key}: {val}")
        parsed.headers_raw = "\n".join(header_lines)

        # Walk MIME parts
        parsed.is_multipart = msg.is_multipart()

        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                disp = str(part.get("Content-Disposition", ""))

                if "attachment" in disp.lower():
                    parsed.has_attachments = True
                    fname = part.get_filename()
                    if fname:
                        parsed.attachment_names.append(self._decode_header_value(fname))
                    continue

                if ct == "text/plain":
                    payload = self._safe_decode(part)
                    if payload:
                        parsed.body_plain += payload + "\n"
                elif ct == "text/html":
                    payload = self._safe_decode(part)
                    if payload:
                        parsed.body_html += payload + "\n"
        else:
            ct = msg.get_content_type()
            payload = self._safe_decode(msg)
            if ct == "text/html":
                parsed.body_html  = payload
                parsed.body_plain = self._html_to_text(payload)
            else:
                parsed.body_plain = payload

        # If we have HTML but no plain, extract from HTML
        if parsed.body_html and not parsed.body_plain:
            parsed.body_plain = self._html_to_text(parsed.body_html)

    def _extract_urls(self, plain: str, html_body: str) -> list[str]:
        """Extract all unique URLs from body and anchor hrefs."""
        urls = set()

        # From plain text
        for url in self._URL_RE.findall(plain):
            urls.add(url.rstrip(".,;:)\"'"))

        # From HTML body
        for url in self._URL_RE.findall(html_body):
            urls.add(url.rstrip(".,;:)\"'"))

        # From anchor hrefs
        for href in self._ANCHOR_RE.findall(html_body):
            if href.startswith("http"):
                urls.add(href)

        return sorted(urls)[:50]   # Cap at 50 URLs

    @staticmethod
    def _html_to_text(html_body: str) -> str:
        """Simple HTML → plain text conversion."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_body, "html.parser")
            # Remove script/style
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return re.sub(r"\s+", " ", text).strip()
        except ImportError:
            # Fallback: strip tags with regex
            text = re.sub(r"<[^>]+>", " ", html_body)
            return html.unescape(re.sub(r"\s+", " ", text)).strip()

    @staticmethod
    def _safe_decode(part) -> str:
        """Decode a MIME part payload to string."""
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return part.get_payload() or ""

            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except Exception:
                return payload.decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _decode_header_value(value: str) -> str:
        """Decode encoded email header (RFC-2047)."""
        if not value:
            return ""
        try:
            parts = _decode_header(value)
            decoded = []
            for text, charset in parts:
                if isinstance(text, bytes):
                    decoded.append(text.decode(charset or "utf-8", errors="replace"))
                else:
                    decoded.append(str(text))
            return " ".join(decoded)
        except Exception:
            return value
