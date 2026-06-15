@echo off
:: ============================================================
:: PhishGuard AI - Launch Script
:: ============================================================
setlocal

title PhishGuard AI - Running

echo.
echo  Starting PhishGuard AI...
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop the server.
echo.

:: Check if venv exists
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate.bat

:: Set environment variables for CPU optimization
set TOKENIZERS_PARALLELISM=false
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set TRANSFORMERS_VERBOSITY=error

:: Launch Streamlit
streamlit run app\streamlit_app.py ^
    --server.port 8501 ^
    --server.headless false ^
    --server.maxUploadSize 50 ^
    --browser.gatherUsageStats false ^
    --theme.base dark

pause
