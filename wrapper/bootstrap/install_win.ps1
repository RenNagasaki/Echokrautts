# Optional Windows starter for the F5-TTS wrapper (SPEC §3).
# Used only when no Python is available to run bootstrap.py directly: it fetches
# uv, then runs bootstrap.py via uv's managed Python. Invoke hidden from C#:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File install_win.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$uvDir = Join-Path $root ".uv"
$uv = Join-Path $uvDir "uv.exe"

if (-not (Test-Path $uv)) {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) {
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        Copy-Item $existing.Source $uv
    } else {
        $asset = "uv-x86_64-pc-windows-msvc.zip"
        $url = "https://github.com/astral-sh/uv/releases/latest/download/$asset"
        $zip = Join-Path $env:TEMP $asset
        Invoke-WebRequest -Uri $url -OutFile $zip
        New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
        Expand-Archive -Path $zip -DestinationPath $uvDir -Force
        Remove-Item $zip -Force
    }
}

# Run the bootstrap with uv's managed Python, forwarding all args (e.g. --start).
# --no-project is REQUIRED: this script's cwd may be the wrapper dir (which has a
# pyproject.toml). Without it, `uv run` treats that as a project and auto-creates
# AND syncs `.venv` from pyproject (f5-tts → torch 2.12+cpu + torchcodec from
# PyPI) *before* bootstrap.py runs — clobbering the carefully pinned torch
# (2.7.0+cu128, no torchcodec) that step_deps installs. bootstrap.py owns .venv.
& $uv run --no-project --python 3.11 python (Join-Path $PSScriptRoot "bootstrap.py") @args
exit $LASTEXITCODE
