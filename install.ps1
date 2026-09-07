# claudelocal installer - puts `claudelocal` on PATH and checks the machine can run the model.
# Pure ASCII, no BOM on purpose (AV engines on this machine hard-lock BOM'd .ps1 files).

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$bin  = Join-Path $root 'bin'

Write-Host ''
Write-Host 'claudelocal installer' -ForegroundColor Cyan
Write-Host ''

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host '  node.exe is not on PATH. Install Node 20+ and re-run this.' -ForegroundColor Red
    exit 1
}
Write-Host "  node        $(node --version)"

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Host '  claude      not on PATH - claudelocal will look in %USERPROFILE%\.local\bin' -ForegroundColor Yellow
} else {
    Write-Host "  claude      $((Get-Command claude).Source)"
}

# --- PATH -------------------------------------------------------------------------------------
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$bin*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$bin", 'User')
    Write-Host "  PATH        added $bin  (open a new terminal to pick it up)" -ForegroundColor Green
} else {
    Write-Host "  PATH        already contains $bin"
}
$env:Path = "$env:Path;$bin"

# --- runtime ----------------------------------------------------------------------------------
$exe = Join-Path $root 'runtime\llama-server.exe'
if (-not (Test-Path $exe)) {
    Write-Host '  runtime     downloading the llama.cpp CUDA build ...'
    & node (Join-Path $root 'src\cli.mjs') install
} else {
    Write-Host "  runtime     $exe"
}

# --- models -----------------------------------------------------------------------------------
# Anything already installed in LM Studio or ollama counts, so there may be nothing to download.
$found = (& node (Join-Path $root 'src\cli.mjs') models --rescan) 2>$null
$count = ($found | Select-String -Pattern '^\d+ local models').ToString()
if ($count) {
    Write-Host "  models      $count"
    Write-Host '              run: claudelocal --cl-models   to see them'
} else {
    Write-Host '  models      none found - install one in LM Studio or ollama,' -ForegroundColor Yellow
    Write-Host '              or run: claudelocal --cl-pull   for the bundled Qwen3.8-27B' -ForegroundColor Yellow
}

Write-Host ''
& node (Join-Path $root 'src\cli.mjs') doctor
Write-Host ''
Write-Host 'Done. Open a new terminal and run:  claudelocal' -ForegroundColor Green
Write-Host ''
