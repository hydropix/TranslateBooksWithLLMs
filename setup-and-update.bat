@echo off
REM ============================================
REM TranslateBookWithLLM - Setup & Update
REM Installation and Update Script
REM ============================================

setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cls

REM ========================================
REM BANNER
REM ========================================
echo.
echo TranslateBook with LLMs - Setup ^& Update
echo ─────────────────────────────────────────
echo.

REM ========================================
REM STEP 1: Check Python Installation
REM ========================================
echo.
echo [1/6] Checking Python Installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed or not in PATH
    echo     Please install Python 3.8+ from https://www.python.org/
    echo     Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo     [OK] Python %PYTHON_VERSION% detected

REM ========================================
REM STEP 2: Virtual Environment Setup
REM ========================================
echo.
echo [2/6] Virtual Environment Setup...
if not exist "venv" (
    echo     [..] First-time setup - creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo     [X] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo     [OK] Virtual environment created
    set FIRST_INSTALL=1
) else (
    echo     [OK] Virtual environment exists
    set FIRST_INSTALL=0
)

REM ========================================
REM STEP 3: Activate Virtual Environment
REM ========================================
echo.
echo [3/6] Activating Environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo     [X] Failed to activate virtual environment
    pause
    exit /b 1
)
echo     [OK] Virtual environment activated

REM ========================================
REM STEP 4: Check for Updates
REM ========================================
echo.
echo [4/6] Checking for Updates...

REM Check if git is available and update
git --version >nul 2>&1
if not errorlevel 1 (
    echo     [..] Checking for code updates from Git...
    git fetch >nul 2>&1

    for /f %%i in ('git rev-parse HEAD') do set LOCAL_COMMIT=%%i
    for /f %%i in ('git rev-parse @{u} 2^>nul') do set REMOTE_COMMIT=%%i

    if not "!LOCAL_COMMIT!"=="!REMOTE_COMMIT!" (
        if not "!REMOTE_COMMIT!"=="" (
            echo     [..] Updates available! Pulling latest changes...
            git pull
            set NEEDS_UPDATE=1
        )
    ) else (
        echo     [OK] Code is up to date
    )
) else (
    echo     [--] Git not available, skipping code update check
)

REM Check if requirements changed or first install
set NEEDS_UPDATE=0

if !FIRST_INSTALL!==1 (
    set NEEDS_UPDATE=1
    echo     [..] First installation - will install all dependencies
) else (
    if exist "venv\.requirements_hash" (
        for /f "delims=" %%i in ('certutil -hashfile requirements.txt MD5 ^| find /v "hash"') do set NEW_HASH=%%i
        set /p OLD_HASH=<venv\.requirements_hash
        if not "!NEW_HASH!"=="!OLD_HASH!" (
            echo     [..] Dependencies changed - updating packages...
            set NEEDS_UPDATE=1
        )
    ) else (
        echo     [..] No hash found - will update dependencies
        set NEEDS_UPDATE=1
    )
)

REM ========================================
REM STEP 5: Install/Update Dependencies
REM ========================================
echo.
echo [5/6] Managing Dependencies...

if "!NEEDS_UPDATE!"=="1" (
    echo     [..] Upgrading pip...
    python -m pip install --upgrade pip --quiet

    echo     [..] Installing/updating dependencies...
    echo.
    pip install -r requirements.txt --upgrade
    echo.
    if errorlevel 1 (
        echo     [X] Failed to install dependencies
        echo     [X] Please check your internet connection and try again
        pause
        exit /b 1
    )

    for /f "delims=" %%i in ('certutil -hashfile requirements.txt MD5 ^| find /v "hash"') do echo %%i>venv\.requirements_hash
    echo     [OK] Dependencies updated successfully
) else (
    echo     [OK] Dependencies are up to date
)

REM ========================================
REM STEP 6: Environment Setup
REM ========================================
echo.
echo [6/6] Environment Configuration...

if not exist ".env" (
    if exist ".env.example" (
        echo     [..] Creating .env from template...
        copy .env.example .env >nul
        echo     [OK] .env file created
        echo     [!!] Please edit .env to configure your LLM settings
        echo.
        notepad .env
    ) else (
        echo     [!!] .env.example not found, skipping .env creation
    )
) else (
    echo     [OK] .env configuration exists
)

if not exist "translated_files" (
    mkdir translated_files
    echo     [OK] Created output directory: translated_files
)

REM Quick Integrity Check (Silent)
if exist "fix_installation.py" (
    python fix_installation.py >nul 2>&1
)

REM ========================================
REM SETUP COMPLETE
REM ========================================
echo.
echo ─────────────────────────────────────────
echo [OK] Setup complete!
echo.
echo To start the application, run: start.bat
echo ─────────────────────────────────────────
echo.
pause
