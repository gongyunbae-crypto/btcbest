@echo off
TITLE BTC Strategy Miner V3 Launcher
cd /d "%~dp0"
echo ==========================================
echo    BTC Strategy Miner V3 - Auto Launcher
echo ==========================================
echo.
echo [1/2] Checking Dependencies...
python -c "import streamlit, pandas, vectorbt, plotly" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Missing libraries. Installing now...
    pip install streamlit pandas numpy plotly vectorbt ccxt
)

echo [2/2] Starting Backend Engine...
echo.
echo * Dashboard will open in your browser shortly...
echo * Close this window to SHUTDOWN the miner.
echo.
python -m streamlit run app.py
pause
