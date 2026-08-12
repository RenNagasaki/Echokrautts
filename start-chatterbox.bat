@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows) - Chatterbox-Multilingual-Backend
rem  Doppelklick startet den Wrapper mit Resembles Chatterbox Multilingual: holt
rem  uv -> installiert beim ersten Mal alles (Python, GPU-Erkennung,
rem  Abhaengigkeiten, Modelle) -> serviert. Fuer F5-TTS bzw. XTTS stattdessen
rem  start-f5tts.bat bzw. start-xtts.bat verwenden.
rem  Chatterbox ist mehrsprachig (23 Sprachen, Sprache pro Request), klont ohne
rem  Transkript, kennt aber KEIN Streaming und keinen speed-Parameter.
rem  Optionale Argumente (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start --tts-backend chatterbox %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
