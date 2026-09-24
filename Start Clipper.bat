@echo off
rem Start the Clipper web page. Runs from the project folder so .env and runs\ are found.
title Clipper web
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv-gpu\Scripts\clipper.exe" (
    ".venv-gpu\Scripts\clipper.exe" web
) else (
    ".venv\Scripts\clipper.exe" web
)
if errorlevel 1 (
    echo.
    echo Clipper stopped with an error. If it says the port is in use, Clipper is already running:
    echo open http://127.0.0.1:8765 in the browser.
    pause
)
