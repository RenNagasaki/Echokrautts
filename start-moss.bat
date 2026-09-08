@echo off
rem ============================================================================
rem  Echokrautts - One-Click-Starter (Windows) - MOSS-TTS-Nano-Backend
rem  Doppelklick startet den Wrapper mit MOSS-TTS-Nano: holt uv -> installiert
rem  beim ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) ->
rem  serviert. Fuer die anderen Engines start-f5tts.bat bzw. start-xtts.bat.
rem
rem  MOSS ist das kleinste der drei Modelle (312 MB) und laeuft auf der CPU
rem  praktisch genauso schnell wie auf der GPU. Es startet deshalb ABSICHTLICH
rem  auf der CPU und laesst die Grafikkarte dem Spiel - dafuer ist kein Argument
rem  noetig. Wer die GPU trotzdem will:
rem      start-moss.bat --moss-device cuda
rem  Optionale Argumente (z.B. --language en) werden durchgereicht.
rem ============================================================================
setlocal
cd /d "%~dp0wrapper"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "bootstrap\install_win.ps1" --start --tts-backend moss %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo === Echokrautts beendet (Exit %EXITCODE%) ===
pause
exit /b %EXITCODE%
