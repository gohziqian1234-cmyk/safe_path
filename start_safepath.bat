@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo SafePath environment not found.
    echo Follow the Windows setup steps in README.md first.
    pause
    exit /b 1
)

echo Starting SafePath AI...
".venv\Scripts\python.exe" -m streamlit run app.py
