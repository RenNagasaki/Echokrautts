@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows) - Pocket-TTS-Backend (Kyutai, CPU)
rem  Doppelklick startet den Wrapper mit dem Pocket-TTS-Backend (Kyutai, CPU): holt uv ->
rem  installiert beim ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten,
rem  Modelle) -> serviert. Laeuft auf der CPU; ein Modell je Sprache (kein Japanisch).
rem  Sprache per --language (de/en/fr/it/es/pt); optionale Argumente
rem  (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start --tts-backend pocket %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
