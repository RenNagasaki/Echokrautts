@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows)
rem  Doppelklick startet den F5-TTS-Wrapper: holt uv -> installiert beim ersten
rem  Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) -> serviert.
rem  Optionale Argumente (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
