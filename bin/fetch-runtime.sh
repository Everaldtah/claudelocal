#!/usr/bin/env bash
# Pull the prebuilt llama.cpp Windows CUDA runtime into claudelocal/runtime.
# Build must be >= b10450 (commit ece963f41) or Qwen3.8's DeltaNet layers emit garbage on CUDA.
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-b10819}"
TMP="$ROOT/runtime/.dl"; mkdir -p "$TMP" "$ROOT/runtime"
base="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD"
for z in "llama-$BUILD-bin-win-cuda-12.4-x64.zip" "cudart-llama-bin-win-cuda-12.4-x64.zip"; do
  echo "[runtime] $z"
  curl -L --fail --retry 5 --retry-delay 4 --retry-all-errors -C - -o "$TMP/$z" "$base/$z"
done
for z in "$TMP"/*.zip; do
  echo "[runtime] unzip $(basename "$z")"
  powershell -NoLogo -NoProfile -Command "Expand-Archive -LiteralPath '$(cygpath -w "$z")' -DestinationPath '$(cygpath -w "$ROOT/runtime")' -Force"
done
echo "$BUILD" > "$ROOT/runtime/BUILD.txt"
rm -rf "$TMP"
"$ROOT/runtime/llama-server.exe" --version 2>&1 | head -5 || true
