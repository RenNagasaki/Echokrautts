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
& $uv run --python 3.11 python (Join-Path $PSScriptRoot "bootstrap.py") @args
exit $LASTEXITCODE
