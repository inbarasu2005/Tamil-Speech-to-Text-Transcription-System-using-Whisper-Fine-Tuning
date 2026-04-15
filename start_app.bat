@echo off
echo ========================================================
echo Tamil Speech-to-Text Transcription App (Auto-Updater)
echo ========================================================

IF NOT EXIST venv\Scripts\activate.bat (
    echo [INFO] Virtual environment not found. Creating one now...
    python -m venv venv
)

echo.
echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo.
echo [INFO] Checking and updating environment dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ========================================================
echo [INFO] Launching the Gradio Web UI...
echo ========================================================
venv\Scripts\python.exe app.py

pause
