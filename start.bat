@echo off
REM ============================================
REM TranslateBookWithLLM - Start Application
REM Quick Launch Script (with auto-update + restart loop)
REM ============================================

setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cls

REM ========================================
REM BANNER
REM ========================================
echo.
echo TranslateBook with LLMs
echo --------------------------
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

REM ========================================
REM Optional: pull latest code on startup
REM (mirrors start.sh behavior; safe no-op if git absent or no changes)
REM ========================================
git --version >nul 2>&1
if not errorlevel 1 (
    echo Checking for code updates from Git...
    git fetch >nul 2>&1
    for /f %%i in ('git rev-parse HEAD 2^>nul') do set LOCAL_COMMIT=%%i
    for /f %%i in ('git rev-parse @{u} 2^>nul') do set REMOTE_COMMIT=%%i
    if not "!LOCAL_COMMIT!"=="!REMOTE_COMMIT!" (
        if not "!REMOTE_COMMIT!"=="" (
            echo Updates available, pulling latest changes...
            git pull --ff-only
        )
    ) else (
        echo [OK] Code is up to date
    )
) else (
    echo [INFO] Git not available, skipping code update check
)
echo.

REM ========================================
REM LAUNCH APPLICATION (restart loop)
REM ----------------------------------------
REM The Python process can request a restart by exiting with code 42 (used by
REM the in-app auto-update flow). Any other exit code stops the loop.
REM ========================================
:run_server
echo Launching server...
echo.
echo The web interface URL is shown in the server log below.
echo Press Ctrl+C to stop the server
echo.

python translation_api.py
set EXITCODE=!ERRORLEVEL!

if "!EXITCODE!"=="42" (
    echo.
    echo --------------------------
    echo Restart requested by in-app updater.
    echo Re-installing dependencies if requirements.txt changed...
    if exist "requirements.txt" (
        pip install -r requirements.txt --upgrade --quiet
    )
    echo Relaunching...
    echo.
    goto run_server
)

REM If server stops normally
echo.
echo --------------------------
echo Server stopped (exit code !EXITCODE!).
echo.
pause
