@echo off
REM This script sets up the virtual environment and launches the captioning application.

REM --- Python Check ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python was not found.
    echo Please ensure Python is installed and added to your system's PATH.
    pause
    exit /b
)

REM --- Virtual Environment Check and Creation ---
if not exist ".\venv\Scripts\activate.bat" (
    echo.
    echo Virtual environment not found. Creating it now...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo FAILED to create the virtual environment.
        pause
        exit /b
    )

    echo.
    echo Activating environment...
    call ".\venv\Scripts\activate.bat"

    echo.
    echo Installing dependencies. This step may take several minutes...
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install torchvision --index-url https://download.pytorch.org/whl/cu128
    
    echo.
    echo Installation complete!
) else (
    echo.
    echo Virtual environment found. Activating...
    call ".\venv\Scripts\activate.bat"
)

REM --- Launch the Python Application ---
echo.
echo Launching the Gradio application...
python app.py

echo.
echo Application closed. Press any key to exit.
pause