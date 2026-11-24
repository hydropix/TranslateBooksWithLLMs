#!/bin/bash
# ============================================
# TranslateBookWithLLM - Smart Launcher
# Installation + Update + Launch All-in-One
# ============================================

echo ""
echo "============================================"
echo "TranslateBookWithLLM - Smart Launcher"
echo "============================================"
echo ""

# ========================================
# STEP 1: Check Python Installation
# ========================================
echo "[1/7] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    echo ""
    echo "Please install Python 3.11-3.14 using one of these methods:"
    echo "  - Homebrew: brew install python3"
    echo "  - Download from https://www.python.org/"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[OK] Python $PYTHON_VERSION detected"
echo ""

# ========================================
# STEP 2: Virtual Environment Setup
# ========================================
echo "[2/7] Checking virtual environment..."
FIRST_INSTALL=0
if [ ! -d "venv" ]; then
    echo "[INFO] First-time setup detected - creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[OK] Virtual environment created"
    FIRST_INSTALL=1
else
    echo "[OK] Virtual environment exists"
fi
echo ""

# ========================================
# STEP 3: Activate Virtual Environment
# ========================================
echo "[3/7] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
echo "[OK] Virtual environment activated"
echo ""

# ========================================
# STEP 4: Check for Updates
# ========================================
echo "[4/7] Checking for updates..."
NEEDS_UPDATE=0

# Check if git is available and update
if command -v git &> /dev/null; then
    echo "[INFO] Checking for code updates from Git..."
    git fetch &> /dev/null

    LOCAL_COMMIT=$(git rev-parse HEAD 2>/dev/null)
    REMOTE_COMMIT=$(git rev-parse @{u} 2>/dev/null)

    if [ -n "$REMOTE_COMMIT" ] && [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
        echo "[INFO] Updates available! Pulling latest changes..."
        git pull
        NEEDS_UPDATE=1
    else
        echo "[OK] Code is up to date"
    fi
else
    echo "[INFO] Git not available, skipping code update check"
fi

# Check if requirements changed or first install
if [ $FIRST_INSTALL -eq 1 ]; then
    NEEDS_UPDATE=1
    echo "[INFO] First installation - will install all dependencies"
elif [ -f "venv/.requirements_hash" ]; then
    # Compare requirements.txt hash
    NEW_HASH=$(md5sum requirements.txt 2>/dev/null || md5 -q requirements.txt 2>/dev/null)
    OLD_HASH=$(cat venv/.requirements_hash 2>/dev/null)
    if [ "$NEW_HASH" != "$OLD_HASH" ]; then
        echo "[INFO] Dependencies changed - updating packages..."
        NEEDS_UPDATE=1
    fi
else
    echo "[INFO] No hash found - will update dependencies"
    NEEDS_UPDATE=1
fi
echo ""

# ========================================
# STEP 5: Install/Update Dependencies
# ========================================
echo "[5/7] Managing dependencies..."

if [ $NEEDS_UPDATE -eq 1 ]; then
    echo "[INFO] Upgrading pip..."
    python3 -m pip install --upgrade pip --quiet

    echo "[INFO] Installing/updating dependencies from requirements.txt..."
    pip install -r requirements.txt --upgrade
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        echo "[ERROR] Please check your internet connection and try again"
        exit 1
    fi

    # Save requirements hash (works on both macOS and Linux)
    if command -v md5sum &> /dev/null; then
        md5sum requirements.txt > venv/.requirements_hash
    else
        md5 -q requirements.txt > venv/.requirements_hash
    fi

    echo "[OK] Dependencies updated successfully"
else
    echo "[OK] Dependencies are up to date"
    echo "[INFO] If you suspect missing packages, delete venv/.requirements_hash and rerun"
fi
echo ""

# ========================================
# STEP 6: Environment Setup
# ========================================
echo "[6/7] Checking environment configuration..."

# Create .env if missing
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[INFO] Creating .env from template..."
        cp .env.example .env
        echo "[OK] .env file created"
        echo "[WARNING] Please edit .env to configure your LLM settings"
        echo ""
        # Try to open with default editor
        if [ -n "$EDITOR" ]; then
            $EDITOR .env
        elif command -v nano &> /dev/null; then
            nano .env
        elif command -v vim &> /dev/null; then
            vim .env
        else
            echo "[INFO] Please manually edit .env file"
        fi
    else
        echo "[WARNING] .env.example not found, skipping .env creation"
    fi
else
    echo "[OK] .env configuration exists"
fi

# Create output directory
if [ ! -d "translated_files" ]; then
    mkdir translated_files
    echo "[INFO] Created output directory: translated_files"
fi
echo ""

# ========================================
# STEP 7: Quick Integrity Check (Silent)
# ========================================
if [ -f "fix_installation.py" ]; then
    python3 fix_installation.py &> /dev/null
fi

# ========================================
# LAUNCH APPLICATION (restart loop)
# ----------------------------------------
# The Python process can request a restart by exiting with code 42 (used by
# the in-app auto-update flow). Any other exit code stops the loop.
# ========================================
echo "============================================"
echo "Setup Complete! Starting Application..."
echo "============================================"
echo ""
echo "Web interface will be available at:"
echo "http://localhost:5000"
echo ""
echo "The browser will open automatically in a few seconds."
echo "Please wait..."
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================"
echo ""

while true; do
    python3 translation_api.py
    EXITCODE=$?
    if [ "$EXITCODE" -eq 42 ]; then
        echo ""
        echo "============================================"
        echo "Restart requested by in-app updater."
        echo "Re-installing dependencies if requirements.txt changed..."
        echo "============================================"
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt --upgrade --quiet
        fi
        echo "Relaunching..."
        echo ""
        continue
    fi
    break
done

echo ""
echo "============================================"
echo "Server stopped (exit code $EXITCODE)"
echo "============================================"
