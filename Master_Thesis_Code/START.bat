@echo off
echo ============================================================
echo   Multi-Agent Churn Mitigation System - Master's Thesis
echo ============================================================
echo.

if not exist venv (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Installing dependencies...
call venv\Scripts\activate
pip install -q -r requirements.txt

echo [3/4] Pre-flight LLM provider check + Pipeline execution...
echo.
python main.py

echo.
echo [4/4] Pipeline finished. Check outputs/ folder for results.
echo.
pause