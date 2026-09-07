@echo off
rem ============================================================================
rem  Echokrautts - Release-Archiv bauen (Windows)
rem  Doppelklick erzeugt wrapper\EchokrauTTS.zip. Danach nur noch das Release
rem  auf GitHub anlegen und diese Datei als Asset hochladen.
rem  Argumente werden durchgereicht, z.B.:  build-release-zip.bat --list
rem  (zeigt nur, was hineinkaeme, und schreibt nichts).
rem ============================================================================
setlocal
cd /d "%~dp0"

rem Interpreter suchen, statt "python" anzunehmen: auf Windows ist das oft nur
rem der WindowsApps-Platzhalter, der den Store oeffnet statt Python zu starten.
rem Reihenfolge: Test-venv des Projekts -> py-Launcher -> python -> uv.
set "PY="
if exist "wrapper\.venv-test\Scripts\python.exe" set "PY=wrapper\.venv-test\Scripts\python.exe"
if not defined PY (
    py -3 --version >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "PY=%USERPROFILE%\.local\bin\uv.exe run --no-project python"
)
if not defined PY (
    echo FEHLER: kein Python gefunden ^(weder .venv-test, py-Launcher, python noch uv^).
    echo Das Skript braucht nur die Standardbibliothek - jedes Python 3 genuegt.
    pause
    exit /b 2
)

%PY% build-release-zip.py %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo === Fertig. Archiv: wrapper\EchokrauTTS.zip ===
) else (
    echo === FEHLGESCHLAGEN ^(Exit %EXITCODE%^) ===
)
pause
exit /b %EXITCODE%
