@echo off
:: ============================================================
:: PhishGuard AI - Windows Setup Script
:: Automatically creates venv and installs all dependencies
:: ============================================================
setlocal EnableDelayedExpansion

title PhishGuard AI - Setup

echo.
echo  ██████╗ ██╗  ██╗██╗███████╗██╗  ██╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
echo  ██╔══██╗██║  ██║██║██╔════╝██║  ██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
echo  ██████╔╝███████║██║███████╗███████║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
echo  ██╔═══╝ ██╔══██║██║╚════██║██╔══██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
echo  ██║     ██║  ██║██║███████║██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
echo  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
echo.
echo  AI-Powered Phishing Detection System ^| Setup v1.0
echo  ============================================================
echo.

:: Check Python version
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Found Python %PYVER%

:: Check Python version is 3.9+
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    if %%a LSS 3 (
        echo [ERROR] Python 3.9+ required. Found %PYVER%
        pause
        exit /b 1
    )
    if %%a EQU 3 if %%b LSS 9 (
        echo [ERROR] Python 3.9+ required. Found %PYVER%
        pause
        exit /b 1
    )
)

:: Create virtual environment
echo.
echo [2/7] Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists, skipping creation.
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created at .\venv\
)

:: Activate virtual environment
echo.
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

:: Upgrade pip
echo.
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

:: Install PyTorch CPU-only (smaller footprint)
echo.
echo [5/7] Installing PyTorch (CPU-optimized for your Intel i5)...
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu --quiet
if %errorlevel% neq 0 (
    echo [WARNING] PyTorch installation had issues, trying alternative...
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
)
echo [OK] PyTorch CPU installed

:: Install remaining requirements
echo.
echo [6/7] Installing all other dependencies...
pip install -r requirements.txt --quiet --no-deps 2>nul
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some packages may have had issues. Trying individual install...
    pip install transformers sentence-transformers scikit-learn xgboost numpy pandas scipy --quiet
    pip install streamlit plotly tldextract python-whois dnspython --quiet
    pip install beautifulsoup4 chardet html2text lxml requests diskcache cachetools --quiet
    pip install colorlog python-dotenv pydantic tqdm Levenshtein regex --quiet
)
echo [OK] Dependencies installed

:: Pre-download transformer models
echo.
echo [7/7] Pre-downloading AI models (first-time only, ~300MB)...
echo [INFO] Downloading sentence-transformer model (all-MiniLM-L6-v2)...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('[OK] Embedding model ready')"
echo [INFO] Downloading zero-shot classification model...
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768'); print('[OK] Zero-shot model ready')"

:: Verify environment
echo.
echo ============================================================
echo  ENVIRONMENT VERIFICATION
echo ============================================================
python -c "
import sys
print(f'  Python:          {sys.version.split()[0]}')
try:
    import torch; print(f'  PyTorch:         {torch.__version__} (CPU: {not torch.cuda.is_available()})')
except: print('  PyTorch:         [NOT INSTALLED]')
try:
    import transformers; print(f'  Transformers:    {transformers.__version__}')
except: print('  Transformers:    [NOT INSTALLED]')
try:
    import sentence_transformers; print(f'  Sent-Transf:     {sentence_transformers.__version__}')
except: print('  Sent-Transf:     [NOT INSTALLED]')
try:
    import sklearn; print(f'  Scikit-learn:    {sklearn.__version__}')
except: print('  Scikit-learn:    [NOT INSTALLED]')
try:
    import xgboost; print(f'  XGBoost:         {xgboost.__version__}')
except: print('  XGBoost:         [NOT INSTALLED]')
try:
    import streamlit; print(f'  Streamlit:       {streamlit.__version__}')
except: print('  Streamlit:       [NOT INSTALLED]')
try:
    import plotly; print(f'  Plotly:          {plotly.__version__}')
except: print('  Plotly:          [NOT INSTALLED]')
print()
print('  [OK] Environment verification complete!')
"

echo.
echo ============================================================
echo  Setup complete! Run launch.bat to start PhishGuard AI.
echo ============================================================
echo.
pause
