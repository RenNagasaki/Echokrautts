<#
.SYNOPSIS
  Windows test of the real install path: install_win.ps1 -> bootstrap (uv, Python,
  GPU detection, deps, weights) -> server -> /health, /samples, /tts.

.DESCRIPTION
  Runs ON the Windows PC. Normally started from the Mac by tools/windows-test.sh (ssh to the
  gaming PC, copies the working tree, one call per engine, fetches windows-test-out/ back).
  The first call installs into wrapper/ (.uv/.venv/.state/models); later calls in the same
  folder reuse that install. -Cold also points uv's wheel cache into the folder.

  Windows PowerShell 5.1 compatible.
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("xtts", "f5", "moss")] [string] $Backend,
    [int]    $Port           = 8765,
    [int]    $ReadyMinutes   = 120,
    [string] $ExpectBackend  = "",   # detection.json "backend", e.g. "cuda"; empty = don't check
    [string] $ExpectDevice   = "",   # /health "device", e.g. "cuda"; empty = don't check
    [switch] $Cold
)

$ErrorActionPreference = "Stop"
$repo    = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $repo "wrapper"
$out     = Join-Path $repo "windows-test-out"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$base    = "http://127.0.0.1:$Port"

function Step($msg) { Write-Host "==> $msg" }

# Keep the PC awake while this script runs (a desktop goes to sleep mid-install otherwise).
Add-Type -Namespace Ci -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
[Ci.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

Step "Environment"
Get-PSDrive -PSProvider FileSystem | Format-Table -AutoSize Name, @{n="FreeGB";e={[math]::Round($_.Free/1GB,1)}}
try { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv } catch { Write-Host "nvidia-smi not available" }

# uv's managed Python must live in a folder this (elevated, ssh) process created itself: uv links
# cpython-3.11-… as a junction, and Windows refuses to let an elevated process traverse a junction a
# non-elevated process made (os error 448 "untrusted mount point") — the user's normal
# %APPDATA%\uv\python is exactly that. Small (~30 MB), re-downloaded with every fresh test folder.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repo ".windows-test-cache\python"
$env:UV_PYTHON_BIN_DIR     = Join-Path $repo ".windows-test-cache\bin"   # not %USERPROFILE%\.local\bin (same junction issue)
$env:UV_PYTHON_PREFERENCE  = "only-managed"                              # ignore pythons found on PATH

if ($Cold) {
    $env:UV_CACHE_DIR = Join-Path $repo ".windows-test-cache\uv"
    Step "Cold run: UV_CACHE_DIR=$env:UV_CACHE_DIR"
}

$stdout = Join-Path $out "bootstrap-$Backend.ndjson"
$stderr = Join-Path $out "bootstrap-$Backend.err.log"
Step "Starting install_win.ps1 --tts-backend $Backend"
$proc = Start-Process -FilePath "powershell.exe" -PassThru -NoNewWindow `
    -WorkingDirectory $wrapper `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "bootstrap\install_win.ps1",
                    "--start", "--tts-backend", $Backend, "--host", "127.0.0.1", "--port", "$Port")
$null = $proc.Handle   # without touching Handle, Start-Process -PassThru never reports ExitCode

function Stop-Server {
    try { Invoke-RestMethod -Method Post -Uri "$base/shutdown" -TimeoutSec 10 | Out-Null } catch { }
    if (-not $proc.WaitForExit(30000)) { & taskkill /PID $proc.Id /T /F | Out-Null }
}

function Fail($msg) {
    Write-Host "FAIL: $msg" -ForegroundColor Red
    Write-Host "--- last bootstrap events ---"
    if (Test-Path $stdout) { Get-Content $stdout -Tail 40 }
    Write-Host "--- stderr ---"
    if (Test-Path $stderr) { Get-Content $stderr -Tail 60 }
    Stop-Server
    exit 1
}

try {
    Step "Waiting for /health (max $ReadyMinutes min)"
    $deadline = (Get-Date).AddMinutes($ReadyMinutes)
    $health = $null
    $lastLine = ""
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) { Fail "bootstrap exited with $($proc.ExitCode) before the server was ready" }
        try { $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 5; break } catch { }
        if (Test-Path $stdout) {
            $line = Get-Content $stdout -Tail 1
            if ($line -and $line -ne $lastLine) { Write-Host "   $line"; $lastLine = $line }
        }
        Start-Sleep -Seconds 10
    }
    if (-not $health) { Fail "server not ready after $ReadyMinutes min" }
    $health | ConvertTo-Json -Depth 5 | Tee-Object -FilePath (Join-Path $out "health-$Backend.json")

    $detFile = Join-Path $wrapper ".state\detection.json"
    if (Test-Path $detFile) { Copy-Item $detFile (Join-Path $out "detection-$Backend.json") }
    if ($ExpectBackend) {
        $det = Get-Content $detFile -Raw | ConvertFrom-Json
        if ($det.backend -ne $ExpectBackend) { Fail "GPU detection chose '$($det.backend)', expected '$ExpectBackend' ($($det.detail))" }
    }
    if ($ExpectDevice -and $health.device -ne $ExpectDevice) {
        Fail "/health device is '$($health.device)', expected '$ExpectDevice'"
    }
    if ($health.tts_backend -ne $Backend) { Fail "/health tts_backend is '$($health.tts_backend)', expected '$Backend'" }

    Step "GET /samples"
    $samples = (Invoke-RestMethod -Uri "$base/samples" -TimeoutSec 30).samples
    if (-not $samples -or $samples.Count -eq 0) {
        # A native install ships no voices by design (the Echokraut plugin installs them; only the
        # Docker entrypoint fetches the pack). Do what the plugin would: fetch the pack once.
        # samples/ is read live per request, so the running server sees it without a restart.
        Step "samples/ empty -> python -m src.voicepack (as the plugin would provide voices)"
        Push-Location $wrapper
        try { & (Join-Path $wrapper ".venv\Scripts\python.exe") -m src.voicepack } finally { Pop-Location }
        $samples = (Invoke-RestMethod -Uri "$base/samples" -TimeoutSec 30).samples
    }
    if (-not $samples -or $samples.Count -eq 0) { Fail "no samples, even after fetching the voice pack" }
    $sample = [System.IO.Path]::GetFileNameWithoutExtension([string]$samples[0])
    Write-Host "   $($samples.Count) samples, using '$sample'"

    Step "POST /tts"
    $body = @{ sample = $sample; text = "Dies ist ein kurzer Test der Sprachausgabe."; language = "de"; format = "wav" } | ConvertTo-Json
    $wavPath = Join-Path $out "tts-$Backend.wav"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/tts" -ContentType "application/json; charset=utf-8" `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) -OutFile $wavPath -TimeoutSec 1800
    $sw.Stop()
    $bytes = [IO.File]::ReadAllBytes($wavPath)
    if ($bytes.Length -lt 44 -or [Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne "RIFF") { Fail "/tts did not return a WAV" }
    $rate    = [BitConverter]::ToInt32($bytes, 24)
    $seconds = ($bytes.Length - 44) / (2.0 * $rate)
    Write-Host ("   {0:N2} s audio at {1} Hz in {2:N1} s" -f $seconds, $rate, $sw.Elapsed.TotalSeconds)
    if ($seconds -lt 0.5) { Fail "audio too short ($seconds s)" }

    Step "OK"
} finally {
    Stop-Server
}
exit 0
