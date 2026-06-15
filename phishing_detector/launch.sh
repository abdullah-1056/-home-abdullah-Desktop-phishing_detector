#!/bin/bash
# ============================================================
# PhishGuard AI - Linux Launch Script
# Ubuntu 24.04 LTS x86_64
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}[ERROR]${RESET} Virtual environment not found. Run:  python3 start.py --setup-only"
    exit 1
fi

source venv/bin/activate

if ! command -v streamlit &>/dev/null; then
    echo -e "${RED}[ERROR]${RESET} Streamlit not found. Run:  python3 start.py --setup-only"
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TRANSFORMERS_VERBOSITY=error
export PYTHONPATH="$SCRIPT_DIR"

echo
echo -e "${CYAN}${BOLD}  PhishGuard AI — Starting...${RESET}"
echo    "  ================================================"
echo -e "  Browser:  ${GREEN}http://localhost:8501${RESET}"
echo    "  Stop:     Ctrl+C"
echo    "  ================================================"
echo

streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless true \
    --server.maxUploadSize 50 \
    --browser.gatherUsageStats false \
    --theme.base dark
