@echo off
REM ============================================
REM TranslateBookWithLLM - Start Application
REM Quick Launch Script
REM ============================================

setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cls

REM ========================================
REM BANNER
REM ========================================
echo.
echo TranslateBook with LLMs
echo ────────────────────────
echo.

REM ========================================
REM Check if setup was run
REM ========================================
if not exist "venv" (
    echo [X] Virtual environment not found!
    echo     Please run setup-and-update.bat first to install.
    echo.
    pause
    exit /b 1
)

REM ========================================
REM Activate Virtual Environment
REM ========================================
echo Initializing environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [X] Failed to activate virtual environment
    echo     Try running setup-and-update.bat to fix the installation
    pause
    exit /b 1
)
echo [OK] Environment ready
echo.

REM ========================================
REM LAUNCH APPLICATION
REM ========================================
echo Launching server...
echo.
echo Web interface:  http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask application (browser auto-opens from Python code)
python translation_api.py

REM If server stops
echo.
echo ────────────────────────
echo Server stopped.
echo.
pause
