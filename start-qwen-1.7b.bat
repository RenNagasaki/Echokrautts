@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows) - Qwen-1.7b-Backend (Qwen3-TTS)
rem  Doppelklick startet den Wrapper mit dem Qwen-1.7b-Backend (Qwen3-TTS): holt uv ->
rem  installiert beim ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten,
rem  Modelle) -> serviert. Braucht eine NVIDIA-Grafikkarte (AMD/ROCm experimentell).
rem  Qwen ist mehrsprachig (Sprache pro Request); optionale Argumente
rem  (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start --tts-backend qwen-1.7b %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
