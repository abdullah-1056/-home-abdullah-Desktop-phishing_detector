#!/bin/bash
# ============================================================
# PhishGuard AI - Linux Setup Script
# Ubuntu 24.04 LTS x86_64
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}[OK]${RESET}   $1"; }
info() { echo -e "${CYAN}[...]${RESET}  $1"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }
err()  { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }

echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
  ██████╗ ██╗  ██╗██╗███████╗██╗  ██╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
  ██╔══██╗██║  ██║██║██╔════╝██║  ██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
  ██████╔╝███████║██║███████╗███████║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
  ██╔═══╝ ██╔══██║██║╚════██║██╔══██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
  ██║     ██║  ██║██║███████║██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
BANNER
echo -e "${RESET}"
echo -e "${BOLD}  AI-Powered Phishing Detection | Ubuntu 24.04 LTS${RESET}"
echo    "  ======================================================="
echo

# 1. Python
echo -e "${BOLD}[1/7] Checking Python...${RESET}"
command -v python3 &>/dev/null || err "python3 not found. Run: sudo apt install python3"
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
PYMAJ=$(python3 -c 'import sys; print(sys.version_info.major)')
PYMIN=$(python3 -c 'import sys; print(sys.version_info.minor)')
{ [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 9 ]; }; } && err "Python 3.9+ required. Found $PYVER"
ok "Python $PYVER"

# 2. System packages
echo
echo -e "${BOLD}[2/7] Installing system dependencies...${RESET}"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip python3-dev build-essential 2>/dev/null || warn "Some system packages failed (non-fatal)"
ok "System dependencies ready"

# 3. Virtual environment
echo
echo -e "${BOLD}[3/7] Creating virtual environment...${RESET}"
if [ -d "venv" ]; then
    info "Already exists — skipping"
else
    python3 -m venv venv || err "Failed. Try: sudo apt install python3-venv python3-full"
    ok "venv created at ./venv/"
fi
source venv/bin/activate
ok "venv activated"

# 4. pip
echo
echo -e "${BOLD}[4/7] Upgrading pip...${RESET}"
pip install --upgrade pip --quiet
ok "pip $(pip --version | cut -d' ' -f2)"

# 5. PyTorch CPU (no version pin)
echo
echo -e "${BOLD}[5/7] Installing PyTorch CPU (latest, no CUDA)...${RESET}"
if pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet; then
    ok "PyTorch $(python3 -c 'import torch; print(torch.__version__)')"
else
    warn "CPU wheel index failed — trying PyPI..."
    pip install torch torchvision --quiet
    ok "PyTorch installed (PyPI fallback)"
fi

# 6. All other packages (no version pins)
echo
echo -e "${BOLD}[6/7] Installing all dependencies...${RESET}"
info "ML/NLP..."
pip install transformers sentence-transformers tokenizers scikit-learn xgboost joblib numpy pandas scipy --quiet
info "UI..."
pip install streamlit plotly --quiet
info "URL/email utilities..."
pip install tldextract dnspython beautifulsoup4 chardet html2text lxml requests urllib3 --quiet
info "Misc..."
pip install diskcache cachetools colorlog python-dotenv pydantic tqdm Levenshtein regex --quiet
ok "All packages installed"

# 7. Download models
echo
echo -e "${BOLD}[7/7] Pre-downloading AI models (~300 MB, one time only)...${RESET}"
python3 - << 'PYEOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print("[OK] Embedding model ready (all-MiniLM-L6-v2, ~80 MB)")
try:
    from transformers import pipeline
    pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768')
    print("[OK] Zero-shot NLI model ready (nli-MiniLM2-L6-H768, ~90 MB)")
except Exception as e:
    print(f"[WARN] Zero-shot model skipped (downloads on first scan): {e}")
PYEOF

# Verify
echo
echo "======================================================="
echo -e "${BOLD}  ENVIRONMENT VERIFICATION${RESET}"
echo "======================================================="
python3 - << 'PYEOF'
import sys
print(f"  Python           {sys.version.split()[0]}")
packages = [
    ("PyTorch",         "torch"),
    ("Transformers",    "transformers"),
    ("SentenceTransf.", "sentence_transformers"),
    ("Scikit-learn",    "sklearn"),
    ("XGBoost",         "xgboost"),
    ("Streamlit",       "streamlit"),
    ("Plotly",          "plotly"),
    ("tldextract",      "tldextract"),
    ("Levenshtein",     "Levenshtein"),
    ("diskcache",       "diskcache"),
]
all_ok = True
for name, mod in packages:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "ok")
        print(f"  {name:<20} {v:<15} ✓")
    except ImportError:
        print(f"  {name:<20} NOT INSTALLED    ✗")
        all_ok = False
print()
print("  ✅ Ready! Run:  bash launch.sh" if all_ok else "  ⚠  Re-run setup.sh to fix missing packages")
PYEOF

echo
echo "======================================================="
echo -e "${GREEN}${BOLD}  Setup complete!  Run:  bash launch.sh${RESET}"
echo "======================================================="
echo
