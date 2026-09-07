#!/usr/bin/env bash
# Resumable, retrying downloader for claudelocal assets. Safe to re-run: finished files are skipped.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/fetch.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

get() { # get <url> <dest>
  local url="$1" dest="$2" want tries=0
  want=$(curl -sIL "$url" | tr -d '\r' | grep -i '^x-linked-size:' | tail -1 | awk '{print $2}')
  [ -z "$want" ] && want=$(curl -sIL "$url" | tr -d '\r' | grep -i '^content-length:' | tail -1 | awk '{print $2}')
  mkdir -p "$(dirname "$dest")"
  while :; do
    local have=0; [ -f "$dest" ] && have=$(stat -c %s "$dest")
    if [ -n "$want" ] && [ "$have" = "$want" ]; then say "OK   $(basename "$dest") ($have bytes)"; return 0; fi
    tries=$((tries+1)); [ $tries -gt 60 ] && { say "FAIL $(basename "$dest") after $tries tries"; return 1; }
    say "GET  $(basename "$dest") try $tries  have=$have want=${want:-?}"
    curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors \
         --limit-rate 0 -C - -o "$dest" "$url" >>"$LOG" 2>&1
    sleep 2
  done
}

B="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
say "=== claudelocal asset fetch starting ==="
get "$B/Qwen3.8-27B-UD-Q4_K_M.gguf"     "$ROOT/models/Qwen3.8-27B-UD-Q4_K_M.gguf"
get "$B/mmproj-F16.gguf"                "$ROOT/models/mmproj-Qwen3.8-27B-F16.gguf"
get "$B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"  "$ROOT/models/mtp-Qwen3.8-27B-Q4_0.gguf"
say "=== all model assets present ==="
