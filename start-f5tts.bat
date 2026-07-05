@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows) - F5-TTS-Backend
rem  Doppelklick startet den Wrapper mit dem F5-TTS-Backend: holt uv ->
rem  installiert beim ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten,
rem  Modelle) -> serviert. Fuer XTTS stattdessen start-xtts.bat verwenden.
rem  Optionale Argumente (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start --tts-backend f5 %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
