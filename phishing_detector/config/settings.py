"""
PhishGuard AI - Global Configuration
All tunable parameters in one place.
"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "cache"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
for d in [MODELS_DIR, LOGS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model Configuration ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # 80MB, 22ms/email, excellent quality
ZERO_SHOT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"  # 90MB, fast NLI
EMBEDDING_DIM   = 384                          # MiniLM-L6 output size
MODEL_CACHE_DIR = str(MODELS_DIR)

# Zero-shot classification labels
PHISHING_HYPOTHESIS    = "This is a phishing, scam, or fraudulent email attempting to steal credentials or personal information."
LEGITIMATE_HYPOTHESIS  = "This is a legitimate, safe, and trustworthy email communication."
ZS_CANDIDATE_LABELS    = ["phishing", "legitimate"]

# ── Ensemble Weights ───────────────────────────────────────────────────────────
WEIGHT_TRANSFORMER  = 0.40   # Semantic transformer confidence
WEIGHT_HEURISTICS   = 0.35   # Rule-based heuristic score
WEIGHT_URL          = 0.15   # URL structural risk
WEIGHT_CLASSIFIER   = 0.10   # RandomForest/XGBoost

# ── Risk Thresholds ────────────────────────────────────────────────────────────
THRESHOLD_HIGH    = 0.75   # Red - High risk
THRESHOLD_MEDIUM  = 0.45   # Orange - Medium risk
THRESHOLD_LOW     = 0.20   # Yellow - Low risk
# Below LOW = Safe (Green)

# ── URL Analysis ──────────────────────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".work", ".date", ".racing", ".download", ".stream", ".bid",
    ".click", ".science", ".link", ".win", ".men", ".loan",
    ".review", ".trade", ".accountant", ".faith",
}

SUSPICIOUS_URL_KEYWORDS = {
    "login", "signin", "sign-in", "secure", "account", "verify",
    "update", "confirm", "banking", "password", "credential",
    "paypal", "ebay", "amazon", "apple", "microsoft", "google",
    "facebook", "netflix", "instagram", "twitter", "bank",
    "support", "helpdesk", "service", "wallet", "recovery",
    "suspended", "unusual", "activity", "urgent", "alert",
}

POPULAR_DOMAINS = [
    "google.com", "facebook.com", "microsoft.com", "apple.com",
    "amazon.com", "paypal.com", "ebay.com", "netflix.com",
    "instagram.com", "twitter.com", "linkedin.com", "youtube.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "citibank.com",
    "dropbox.com", "icloud.com", "outlook.com", "yahoo.com",
]

REDIRECT_PARAMS = {
    "url=", "redirect=", "next=", "return=", "goto=",
    "forward=", "dest=", "destination=", "redir=", "link=",
}

MAX_LEGITIMATE_URL_LENGTH = 100
MAX_SUBDOMAIN_COUNT = 3
MAX_URL_ENTROPY = 4.2   # bits — legitimate URLs rarely exceed this

# ── Email Heuristics ──────────────────────────────────────────────────────────
URGENCY_KEYWORDS = {
    "urgent", "immediate", "action required", "act now", "expires",
    "limited time", "warning", "alert", "suspended", "locked",
    "verify now", "click here", "immediately", "24 hours",
    "your account", "unauthorized", "compromised", "unusual activity",
    "confirm your", "update your", "validate", "reactivate",
    "failure to", "will be closed", "will be terminated",
}

PHISHING_CONTENT_PATTERNS = [
    r"click\s+here\s+to\s+(verify|confirm|update|secure)",
    r"your\s+account\s+(has\s+been|will\s+be)\s+(suspended|locked|closed|terminated)",
    r"enter\s+your\s+(password|credentials|details|information)",
    r"(win|won|winner|prize|reward|gift|congratulations)",
    r"(dear\s+customer|dear\s+user|dear\s+member|valued\s+customer)",
    r"(bank\s+details|credit\s+card|social\s+security|ssn)",
    r"(unusual\s+sign.in|suspicious\s+activity|security\s+alert)",
    r"(update\s+your\s+payment|billing\s+information\s+required)",
]

SAFE_SENDER_DOMAINS = {
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com",
    "icloud.com", "protonmail.com", "apple.com", "microsoft.com",
}

# ── Performance / Hardware ─────────────────────────────────────────────────────
MAX_WORKERS            = 4           # Thread pool size (i5 = 4 cores)
BATCH_SIZE             = 8           # Transformer batch size for CPU
MAX_TOKEN_LENGTH       = 256         # Truncate long emails (faster inference)
CACHE_TTL_SECONDS      = 3600        # 1 hour cache TTL
CACHE_MAX_SIZE_MB      = 256         # Max cache size
REQUEST_TIMEOUT        = 5           # External request timeout (seconds)
MODEL_WARM_UP          = True        # Pre-warm models on startup

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL    = "INFO"
LOG_FILE     = str(LOGS_DIR / "phishguard.log")
LOG_MAX_BYTES = 10 * 1024 * 1024    # 10MB
LOG_BACKUPS  = 3

# ── Reference Phishing Embeddings ─────────────────────────────────────────────
# These canonical phrases define the phishing semantic space for cosine similarity
PHISHING_REFERENCE_TEXTS = [
    "Your account has been suspended. Click here to verify your identity immediately.",
    "Urgent: Unusual sign-in activity detected on your account. Update your password now.",
    "Dear customer, your banking credentials need to be confirmed or your account will be closed.",
    "Congratulations! You have been selected to receive a prize. Enter your details to claim.",
    "Your PayPal account has been limited. Please verify your information to restore access.",
    "Security Alert: Your credit card was used in an unauthorized transaction. Verify now.",
    "Your Apple ID has been disabled. Click the link below to restore access to your account.",
    "Action required: Update your payment information to avoid service interruption.",
    "We detected suspicious login from unknown device. Confirm your account details.",
    "Your password will expire in 24 hours. Click here to update your password immediately.",
    "Dear valued member, your account is at risk. Please confirm your social security number.",
    "Netflix billing problem. Update payment method to continue enjoying your subscription.",
]

LEGITIMATE_REFERENCE_TEXTS = [
    "Thank you for your recent purchase. Your order has been shipped and is on its way.",
    "Here is your monthly newsletter with the latest news and updates from our team.",
    "Your meeting is scheduled for tomorrow at 3 PM. Please find the agenda attached.",
    "We wanted to follow up on our previous conversation and share the project updates.",
    "Please find attached the invoice for services rendered in the previous month.",
    "Thank you for signing up. We are excited to have you on board with our platform.",
    "The quarterly report is now available. Please review the attached document.",
    "Your support ticket has been resolved. Let us know if you need further assistance.",
    "Reminder: The team meeting is tomorrow. Here are the discussion points for the agenda.",
    "Your subscription renewal is coming up next month. No action required at this time.",
]
