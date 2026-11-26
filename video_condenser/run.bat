@echo off
cd /d %~dp0

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg install aagilla. Modalu ee command run madi: winget install Gyan.FFmpeg
  echo Install aada mele ee window close madi, run.bat matte double-click madi.
  pause
  exit /b
)

if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
start http://127.0.0.1:5000
python app.py
pause
