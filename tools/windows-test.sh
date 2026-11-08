#!/usr/bin/env bash
# Windows-Test des echten Installationswegs, vom Mac aus auf dem Gaming-PC (per ssh):
# Arbeitskopie hinüberkopieren -> install_win.ps1 (uv, Python, GPU-Erkennung, Pakete, Modelle) ->
# Server -> /health, /samples, /tts je Engine -> Ergebnisse zurück nach ./windows-test-out/.
# Läuft unter dem normalen Konto des PCs (ssh-Alias aus dem streaming-pc-Repo, mac/install-mac.sh).
#
#   tools/windows-test.sh [Engine …]       Standard: xtts f5 moss (nacheinander, eine Installation)
#     --keep      vorhandene Installation auf dem PC weiterverwenden (sonst jedes Mal frisch)
#     --cold      zusätzlich uvs Wheel-Cache umgehen (echte Erstinstallation inkl. Downloads)
#     --no-wake   PC nicht per Wake-on-LAN wecken, wenn er nicht erreichbar ist
#
# Umgebung (optional):
#   GAMING_PC        ssh-Host-Alias               (Standard: gaming-pc)
#   GAMING_PC_MAC    MAC-Adresse für WoL           (Standard: 08:BF:B8:1D:5A:98)
#   REMOTE_TEST_DIR  Testordner auf dem PC         (Standard: E:/EchokrauttsTest — C: hat zu wenig Platz)
set -euo pipefail

HOST="${GAMING_PC:-gaming-pc}"
MAC="${GAMING_PC_MAC:-08:BF:B8:1D:5A:98}"
DIR="${REMOTE_TEST_DIR:-E:/EchokrauttsTest}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/windows-test-out"
KEEP=0; COLD=0; WAKE=1; ENGINES=()

for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        --cold) COLD=1 ;;
        --no-wake) WAKE=0 ;;
        xtts|f5|moss) ENGINES+=("$a") ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unbekanntes Argument: $a" >&2; exit 2 ;;
    esac
done
[[ ${#ENGINES[@]} -gt 0 ]] || ENGINES=(xtts f5 moss)

reachable() { ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" exit >/dev/null 2>&1; }
remote()    { ssh -o BatchMode=yes "$HOST" "$1"; }

# 1. PC erreichbar machen
if ! reachable; then
    [[ $WAKE -eq 1 ]] || { echo "$HOST nicht erreichbar (--no-wake)." >&2; exit 1; }
    echo "==> $HOST schläft, sende Wake-on-LAN an $MAC"
    python3 - "$MAC" <<'EOF'
import socket, sys
mac = bytes.fromhex(sys.argv[1].replace(':', '').replace('-', ''))
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
for addr in ('255.255.255.255', '192.168.178.255'):
    s.sendto(b'\xff' * 6 + mac * 16, (addr, 9))
EOF
    for _ in $(seq 1 30); do reachable && break; sleep 10; done
    reachable || { echo "$HOST nach 5 min nicht erreichbar." >&2; exit 1; }
fi

# 2. Testordner vorbereiten: alte Server beenden, ohne --keep alles löschen
echo "==> Testordner $DIR auf $HOST vorbereiten"
remote "\$d = '$DIR'
Get-CimInstance Win32_Process | Where-Object { \$_.ExecutablePath -and \$_.ExecutablePath -like ((\$d -replace '/', '\\') + '*') } |
    ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }
if ($KEEP -eq 0 -and (Test-Path \$d)) { cmd /c rd /s /q (\$d -replace '/', '\\') }
New-Item -ItemType Directory -Force \$d | Out-Null
\$o = Join-Path \$d 'windows-test-out'
if (Test-Path \$o) { Remove-Item \$o -Recurse -Force }
exit 0"   # ohne exit meldet powershell -c den Status des letzten Befehls

# 3. Arbeitskopie übertragen (getrackte + neue, nicht ignorierte Dateien — auch Uncommittetes)
echo "==> Arbeitskopie übertragen"
(cd "$REPO" && git ls-files -co --exclude-standard -z | COPYFILE_DISABLE=1 xargs -0 tar -cf -) |
    ssh -o BatchMode=yes "$HOST" "tar -xf - -C '$DIR'"

# 4. Je Engine testen. xtts/f5 müssen auf CUDA laufen, MOSS läuft absichtlich auf der CPU (moss_device).
failed=()
for e in "${ENGINES[@]}"; do
    args="-Backend $e -ExpectBackend cuda"
    [[ "$e" == moss ]] || args="$args -ExpectDevice cuda"
    [[ $COLD -eq 1 ]] && args="$args -Cold"
    echo; echo "==> Engine $e"
    if remote "powershell -NoProfile -ExecutionPolicy Bypass -File '$DIR/tools/windows-test.ps1' $args; exit \$LASTEXITCODE"; then
        echo "==> $e: OK"
    else
        echo "==> $e: FEHLGESCHLAGEN"; failed+=("$e")
    fi
    COLD=0   # nur die erste Engine installiert, die weiteren nutzen die Installation
done

# 5. Ergebnisse holen
# scp statt tar: PowerShell 5.1 kann die Binärausgabe eines nativen Programms (WAV im tar) umkodieren.
rm -rf "$OUT"
scp -q -r -o BatchMode=yes "$HOST:$DIR/windows-test-out" "$REPO/" || echo "Warnung: Ergebnisse nicht abholbar" >&2
echo; echo "Ergebnisse: $OUT"
if [[ ${#failed[@]} -gt 0 ]]; then echo "Fehlgeschlagen: ${failed[*]}"; exit 1; fi
echo "Alle Engines OK: ${ENGINES[*]}"
