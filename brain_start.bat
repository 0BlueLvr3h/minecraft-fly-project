@echo off
title Cervello mosca
rem Avvia il server dei cervelli (scripts\02_server.py) con il .venv del progetto.
rem Opzioni extra si possono aggiungere in fondo alla riga di python, es. --spawn-rt 1.2
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Manca il .venv: segui il punto 1 del README.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\02_server.py
pause
