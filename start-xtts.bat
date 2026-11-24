@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows) - XTTS-v2-Backend
rem  Doppelklick startet den Wrapper mit dem XTTS-v2-Backend: holt uv ->
rem  installiert beim ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten,
rem  Modelle) -> serviert. Fuer F5-TTS stattdessen start-f5tts.bat verwenden.
rem  XTTS ist mehrsprachig (Sprache pro Request); optionale Argumente
rem  (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start --tts-backend xtts %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
