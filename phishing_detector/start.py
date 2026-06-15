#!/usr/bin/env python3
"""
PhishGuard AI - Cross-platform startup script.
Works on Linux (Ubuntu 24.04), macOS, and Windows.
Usage:
  python3 start.py              # setup + launch
  python3 start.py --setup-only
  python3 start.py --skip-models
  python3 start.py --launch-only
"""
import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
VENV_DIR   = BASE_DIR / "venv"
IS_WINDOWS = platform.system() == "Windows"


def _venv_python():
    return str(VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python"))

def _venv_pip():
    return str(VENV_DIR / ("Scripts/pip.exe" if IS_WINDOWS else "bin/pip"))

def _venv_streamlit():
    return str(VENV_DIR / ("Scripts/streamlit.exe" if IS_WINDOWS else "bin/streamlit"))

def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs).returncode


def check_python():
    v = sys.version_info
    if v < (3, 9):
        print(f"[ERROR] Python 3.9+ required. Found {v.major}.{v.minor}")
        sys.exit(1)
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")


def create_venv():
    if not VENV_DIR.exists():
        print("[...] Creating virtual environment...")
        rc = run([sys.executable, "-m", "venv", str(VENV_DIR)])
        if rc != 0:
            print("[ERROR] Failed to create venv.")
            print("        Ubuntu fix:  sudo apt install python3-venv python3-full")
            sys.exit(1)
        print(f"[OK] venv created at {VENV_DIR}")
    else:
        print("[OK] venv already exists")


def install_deps():
    pip = _venv_pip()

    print("[...] Upgrading pip...")
    run([pip, "install", "--upgrade", "pip", "-q"])

    # PyTorch CPU — NO version pin, picks latest compatible with running Python
    print("[...] Installing PyTorch CPU (latest, no CUDA)...")
    rc = run([
        pip, "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "-q",
    ])
    if rc != 0:
        print("[WARN] CPU wheel index failed — falling back to default PyPI torch...")
        run([pip, "install", "torch", "torchvision", "-q"])

    print("[...] Installing ML/NLP packages...")
    run([pip, "install",
         "transformers", "sentence-transformers", "tokenizers",
         "scikit-learn", "xgboost", "joblib",
         "numpy", "pandas", "scipy", "-q"])

    print("[...] Installing UI packages...")
    run([pip, "install", "streamlit", "plotly", "-q"])

    print("[...] Installing URL/email packages...")
    run([pip, "install",
         "tldextract", "dnspython",
         "beautifulsoup4", "chardet", "html2text", "lxml",
         "requests", "urllib3", "-q"])

    print("[...] Installing misc utilities...")
    run([pip, "install",
         "diskcache", "cachetools", "colorlog",
         "python-dotenv", "pydantic", "tqdm",
         "Levenshtein", "regex", "-q"])

    print("[OK] All packages installed.")


def predownload_models():
    python = _venv_python()
    print("[...] Pre-downloading transformer models (~300 MB, first time only)...")
    script = "\n".join([
        "from sentence_transformers import SentenceTransformer",
        "m = SentenceTransformer('all-MiniLM-L6-v2')",
        "print('[OK] Embedding model ready  (all-MiniLM-L6-v2, ~80 MB)')",
        "try:",
        "    from transformers import pipeline",
        "    p = pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768')",
        "    print('[OK] Zero-shot NLI model ready (nli-MiniLM2-L6-H768, ~90 MB)')",
        "except Exception as e:",
        "    print(f'[WARN] Zero-shot model skipped: {e}')",
    ])
    rc = run([python, "-c", script])
    if rc != 0:
        print("[WARN] Model download had issues — will retry on first scan.")


def verify():
    python = _venv_python()
    print()
    print("=" * 58)
    print("  ENVIRONMENT VERIFICATION")
    print("=" * 58)
    checks = [
        ("PyTorch",          "torch"),
        ("Transformers",     "transformers"),
        ("SentenceTransf.",  "sentence_transformers"),
        ("Scikit-learn",     "sklearn"),
        ("XGBoost",          "xgboost"),
        ("Streamlit",        "streamlit"),
        ("Plotly",           "plotly"),
        ("tldextract",       "tldextract"),
        ("Levenshtein",      "Levenshtein"),
        ("diskcache",        "diskcache"),
    ]
    script_lines = [
        "import sys",
        "print(f'  Python          {sys.version.split()[0]}')",
        "checks = " + repr(checks),
        "all_ok = True",
        "for name, mod in checks:",
        "    try:",
        "        m = __import__(mod)",
        "        v = getattr(m, '__version__', 'ok')",
        "        print(f'  {name:<20} {v:<15} ok')",
        "    except ImportError:",
        "        print(f'  {name:<20} NOT INSTALLED')",
        "        all_ok = False",
        "print()",
        "print('  All packages present — ready to launch!' if all_ok else '  Some packages missing — re-run start.py.')",
    ]
    run([python, "-c", "\n".join(script_lines)])


def launch():
    streamlit = _venv_streamlit()
    app_path  = BASE_DIR / "app" / "streamlit_app.py"
    env = os.environ.copy()
    env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS":        "4",
        "MKL_NUM_THREADS":        "4",
        "TRANSFORMERS_VERBOSITY": "error",
        "PYTHONPATH":             str(BASE_DIR),
    })
    print()
    print("=" * 58)
    print("  PhishGuard AI is starting...")
    print("  Browser: http://localhost:8501")
    print("  Stop:    Ctrl+C")
    print("=" * 58)
    print()
    subprocess.run([
        streamlit, "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "true",       # <-- true prevents the email prompt
        "--server.maxUploadSize", "50",
        "--browser.gatherUsageStats", "false",
    ], env=env)


def main():
    ap = argparse.ArgumentParser(description="PhishGuard AI launcher")
    ap.add_argument("--setup-only",  action="store_true", help="Install deps but don't launch")
    ap.add_argument("--skip-models", action="store_true", help="Skip transformer model pre-download")
    ap.add_argument("--launch-only", action="store_true", help="Skip install, just launch")
    args = ap.parse_args()

    print()
    print("  PhishGuard AI — Setup & Launch")
    print("=" * 58)

    check_python()

    if not args.launch_only:
        create_venv()
        install_deps()
        if not args.skip_models:
            predownload_models()
        verify()

    if not args.setup_only:
        launch()


if __name__ == "__main__":
    main()
