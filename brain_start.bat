@echo off
title Mosca brain
rem Starts the brain server (scripts\02_server.py) with the project's .venv.
rem Extra options can be added at the end of the python line, e.g. --spawn-rt 1.2
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo The .venv is missing: follow step 1 of the README.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\02_server.py
pause
