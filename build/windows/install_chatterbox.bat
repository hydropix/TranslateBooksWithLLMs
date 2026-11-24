@echo off
REM ============================================================================
REM Script d'installation robuste pour Chatterbox TTS
REM Inclut PyTorch avec CUDA et toutes les dépendances nécessaires
REM ============================================================================

echo ========================================
echo Installation de Chatterbox TTS
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez installer Python 3.11-3.14 depuis https://www.python.org/downloads/
    echo N'oubliez pas de cocher "Add Python to PATH" lors de l'installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python est installe:
python --version
echo.

REM Obtenir le chemin du script
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Vérifier si l'environnement virtuel existe
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creation de l'environnement virtuel...
    python -m venv venv
    if errorlevel 1 (
        echo [ERREUR] Impossible de creer l'environnement virtuel.
        echo Assurez-vous que le module venv est installe.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Environnement virtuel cree.
) else (
    echo [OK] Environnement virtuel detecte.
)
echo.

REM Activer l'environnement virtuel
echo [INFO] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel.
    echo.
    pause
    exit /b 1
)
echo [OK] Environnement virtuel active.
echo.

REM Mettre à jour pip, setuptools et wheel
echo [INFO] Mise a jour de pip, setuptools et wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ATTENTION] La mise a jour de pip a echoue, mais on continue...
)
echo.

REM Détecter la présence de CUDA
echo [INFO] Detection de CUDA...
set CUDA_AVAILABLE=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set CUDA_AVAILABLE=1
    echo [OK] GPU NVIDIA detecte avec CUDA:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
) else (
    echo [ATTENTION] Aucun GPU NVIDIA/CUDA detecte.
    echo L'installation se fera en mode CPU (plus lent).
    echo.
)

REM Installer les dépendances de base depuis requirements.txt
echo [INFO] Installation des dependances de base...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation des dependances de base.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Dependances de base installees.
) else (
    echo [ATTENTION] Fichier requirements.txt non trouve, on passe cette etape.
)
echo.

REM Désinstaller PyTorch existant pour éviter les conflits
echo [INFO] Nettoyage des installations PyTorch existantes...
python -m pip uninstall -y torch torchaudio torchvision 2>nul
echo.

REM Installer PyTorch avec CUDA ou CPU selon la disponibilité
if "%CUDA_AVAILABLE%"=="1" (
    echo [INFO] Installation de PyTorch avec support CUDA 12.1...
    echo Cette etape peut prendre plusieurs minutes...
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation de PyTorch avec CUDA.
        echo Tentative d'installation de la version CPU...
        python -m pip install torch torchaudio
        if errorlevel 1 (
            echo [ERREUR] Echec de l'installation de PyTorch.
            echo.
            pause
            exit /b 1
        )
    ) else (
        echo [OK] PyTorch avec CUDA installe.
    )
) else (
    echo [INFO] Installation de PyTorch en mode CPU...
    echo Cette etape peut prendre plusieurs minutes...
    python -m pip install torch torchaudio
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation de PyTorch.
        echo.
        pause
        exit /b 1
    )
    echo [OK] PyTorch CPU installe.
)
echo.

REM Vérifier l'installation de PyTorch
echo [INFO] Verification de l'installation de PyTorch...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERREUR] PyTorch n'est pas correctement installe.
    echo.
    pause
    exit /b 1
)
echo.

REM Vérifier CUDA dans PyTorch
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Nombre de GPU: {torch.cuda.device_count()}') if torch.cuda.is_available() else None" 2>nul
echo.

REM Installer Chatterbox TTS
echo [INFO] Installation de Chatterbox TTS...
python -m pip install chatterbox-tts
if errorlevel 1 (
    echo [ERREUR] Echec de l'installation de Chatterbox TTS.
    echo.
    echo Tentative d'installation alternative avec --no-deps puis resolution des dependances...
    python -m pip install --no-deps chatterbox-tts
    python -m pip install gruut-ipa typing_extensions transformers soundfile phonemizer pysbd
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation alternative.
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Chatterbox TTS installe.
echo.

REM Installer ffmpeg si nécessaire (pour Edge TTS)
echo [INFO] Verification de ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [ATTENTION] ffmpeg n'est pas installe ou n'est pas dans le PATH.
    echo ffmpeg est recommande pour la conversion audio.
    echo Telechargez-le depuis: https://ffmpeg.org/download.html
    echo.
) else (
    echo [OK] ffmpeg est installe.
    echo.
)

REM Créer un script de test
echo [INFO] Creation du script de test...
echo import sys > test_chatterbox_install.py
echo import torch >> test_chatterbox_install.py
echo print("=" * 60) >> test_chatterbox_install.py
echo print("TEST D'INSTALLATION DE CHATTERBOX TTS") >> test_chatterbox_install.py
echo print("=" * 60) >> test_chatterbox_install.py
echo print(f"Python version: {sys.version}") >> test_chatterbox_install.py
echo print(f"PyTorch version: {torch.__version__}") >> test_chatterbox_install.py
echo print(f"CUDA disponible: {torch.cuda.is_available()}") >> test_chatterbox_install.py
echo if torch.cuda.is_available(): >> test_chatterbox_install.py
echo     print(f"Nombre de GPU: {torch.cuda.device_count()}") >> test_chatterbox_install.py
echo     for i in range(torch.cuda.device_count()): >> test_chatterbox_install.py
echo         print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") >> test_chatterbox_install.py
echo         print(f"    VRAM totale: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB") >> test_chatterbox_install.py
echo else: >> test_chatterbox_install.py
echo     print("Mode CPU uniquement") >> test_chatterbox_install.py
echo print() >> test_chatterbox_install.py
echo try: >> test_chatterbox_install.py
echo     from chatterbox import ChatterboxTTS >> test_chatterbox_install.py
echo     print("[OK] Chatterbox TTS importe avec succes") >> test_chatterbox_install.py
echo     print() >> test_chatterbox_install.py
echo     print("Installation terminee avec succes!") >> test_chatterbox_install.py
echo     print("Vous pouvez maintenant utiliser Chatterbox TTS dans l'application.") >> test_chatterbox_install.py
echo except Exception as e: >> test_chatterbox_install.py
echo     print(f"[ERREUR] Impossible d'importer Chatterbox TTS: {e}") >> test_chatterbox_install.py
echo     sys.exit(1) >> test_chatterbox_install.py
echo print("=" * 60) >> test_chatterbox_install.py

echo [OK] Script de test cree.
echo.

REM Exécuter le test
echo [INFO] Test de l'installation...
python test_chatterbox_install.py
if errorlevel 1 (
    echo.
    echo [ERREUR] Le test d'installation a echoue.
    echo Veuillez verifier les messages d'erreur ci-dessus.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation terminee avec succes!
echo ========================================
echo.
echo Pour utiliser Chatterbox TTS:
echo 1. Lancez l'application avec start.bat
echo 2. Dans l'interface web, selectionnez "Chatterbox TTS" comme fournisseur TTS
echo 3. Optionnellement, uploadez un echantillon audio pour le clonage de voix
echo.
echo Pour tester manuellement Chatterbox TTS:
echo   venv\Scripts\activate
echo   python test_chatterbox_install.py
echo.

if "%CUDA_AVAILABLE%"=="0" (
    echo [NOTE] Vous utilisez le mode CPU.
    echo Pour de meilleures performances, installez un GPU NVIDIA avec CUDA.
    echo.
)

echo.
pause
