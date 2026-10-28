# claudelocal - load a local model and hand Claude Code to it.
#
# Every GGUF on this machine - claudelocal's own, LM Studio's and ollama's - is written into a
# `modelPicker` block and handed to the child `claude` through --settings, so /model lists them and
# picking one routes the next request to it. The settings file dies with the session; the user's
# own settings.json is never touched.
#
# Plain `claude` is untouched: every provider variable is set for this child process only and
# restored afterwards, so this cannot strand the user's normal Claude Code on a dead local model.
#
# Pure ASCII, no BOM on purpose - several AV engines on this machine hard-lock BOM'd .ps1 files.

$ErrorActionPreference = 'Stop'

$script:Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $script:Root 'src\cli.mjs'))) {
    $script:Root = Join-Path $env:USERPROFILE 'claudelocal'
}
$cli = Join-Path $script:Root 'src\cli.mjs'

function Show-Help {
    Write-Host ''
    Write-Host 'claudelocal - Claude Code on any local model on this machine' -ForegroundColor Cyan
    Write-Host '  claudelocal                 start Claude Code with NO model loaded (/model loads one)'
    Write-Host '  claudelocal -p "prompt"     any normal claude argument passes through'
    Write-Host ''
    Write-Host '  claudelocal --cl-models     every model found, and what /model will offer'
    Write-Host '  claudelocal --cl-remotes    remote models from remotes.json (--cl-check to call them)'
    Write-Host '  claudelocal --cl-codex-sync refresh ChatGPT subscription models from your Codex login'
    Write-Host '  claudelocal --cl-model A    load a specific model by alias at startup'
    Write-Host '  claudelocal --cl-load       load the last-used model at startup, as it used to'
    Write-Host '  claudelocal --cl-min-ctx N  drop models below N tokens, raising the shared window'
    Write-Host '  claudelocal --cl-rescan     re-read every GGUF header'
    Write-Host ''
    Write-Host '  claudelocal --cl-plan       memory plan + the context/speed trade-off table'
    Write-Host '  claudelocal --cl-status     server / proxy / guard / RAM / VRAM'
    Write-Host '  claudelocal --cl-bench      measure tokens per second'
    Write-Host '  claudelocal --cl-tune       try both VRAM strategies, keep the faster'
    Write-Host '  claudelocal --cl-doctor     full environment dump'
    Write-Host '  claudelocal --cl-down       unload the model, free all RAM and VRAM'
    Write-Host '  claudelocal --cl-pull       download / resume the model weights'
    Write-Host ''
    Write-Host '  claudelocal --cl-ctx 65536       smaller context, more of the model on the GPU'
    Write-Host '  claudelocal --cl-kv q8_0         better KV cache quality (needs more RAM)'
    Write-Host '  claudelocal --cl-strategy ssm-pack'
    Write-Host '  claudelocal --cl-restart         reload even if a server is already up'
    Write-Host ''
}

if (-not (Test-Path $cli)) { Write-Host "[claudelocal] cli not found at $cli" -ForegroundColor Red; exit 1 }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Write-Host '[claudelocal] node.exe is not on PATH (Node 20+ required)' -ForegroundColor Red; exit 1 }

# --- split our own flags from the ones meant for claude ---------------------------------------
$upArgs    = @()
$passthru  = @()
$subcmd    = $null
# Starting up used to mean loading a model - whichever one happened to be loaded last - which is
# minutes of waiting and 15 GB of RAM spent before a single word is typed. Now nothing is loaded
# until a model is picked in /model; --cl-load or --cl-model asks for the old behaviour.
$preload   = $false
$list = @($args)
for ($i = 0; $i -lt $list.Count; $i++) {
    switch ($list[$i]) {
        '--cl-models'   { $subcmd = 'models' }
        '--cl-remotes'  { $subcmd = 'remotes' }
        '--cl-codex-sync' { $subcmd = 'codex-sync' }
        '--cl-check'    { $upArgs += '--check' }
        '--cl-model'    { $i++; $upArgs += @('--model', $list[$i]); $preload = $true }
        '--cl-load'     { $preload = $true }
        '--cl-min-ctx'  { $i++; $upArgs += @('--min-ctx', $list[$i]) }
        '--cl-rescan'   { $upArgs += '--rescan' }
        '--cl-plan'     { $subcmd = 'plan' }
        '--cl-status'   { $subcmd = 'status' }
        '--cl-bench'    { $subcmd = 'bench' }
        '--cl-tune'     { $subcmd = 'tune' }
        '--cl-doctor'   { $subcmd = 'doctor' }
        '--cl-down'     { $subcmd = 'down' }
        '--cl-pull'     { $subcmd = 'pull' }
        '--cl-install'  { $subcmd = 'install' }
        '--cl-ctx'      { $i++; $upArgs += @('--ctx', $list[$i]) }
        '--cl-kv'       { $i++; $upArgs += @('--kv', $list[$i]) }
        '--cl-strategy' { $i++; $upArgs += @('--strategy', $list[$i]) }
        '--cl-ram-margin' { $i++; $upArgs += @('--ram-margin', $list[$i]) }
        '--cl-restart'  { $upArgs += '--restart' }
        '--cl-force'    { $upArgs += '--force' }
        '--cl-vision'    { $upArgs += '--vision' }
        '--cl-help'     { Show-Help; exit 0 }
        default         { $passthru += $list[$i] }
    }
}

if ($subcmd) { & node $cli $subcmd @upArgs; exit $LASTEXITCODE }

if (-not $preload) { $upArgs += '--no-load' }
& node $cli up @upArgs
if ($LASTEXITCODE -ne 0) {
    $why = if ($preload) { 'the model did not load' } else { 'the proxy did not come up' }
    Write-Host "[claudelocal] $why - not starting Claude Code." -ForegroundColor Red
    exit 1
}

$stateFile = Join-Path $script:Root 'state.json'
$state = Get-Content $stateFile -Raw | ConvertFrom-Json

# --- the /model picker ---------------------------------------------------------------------------
# Claude Code will curate its own model picker from a `modelPicker` block, but only when that block
# arrives through managed settings, --settings or user settings - never from a project checkout. So
# the list of local models is generated fresh here and passed with --settings, which keeps it scoped
# to this one child process: the user's ~/.claude/settings.json is not touched, and plain `claude`
# still sees Anthropic's own lineup.
$settingsFile = Join-Path ([IO.Path]::GetTempPath()) ("claudelocal-models-{0}.json" -f $PID)
$settingsJson = (& node $cli settings @upArgs) -join "`n"
if ($LASTEXITCODE -ne 0 -or -not $settingsJson) {
    Write-Host '[claudelocal] could not build the model list - /model will show the built-in lineup.' -ForegroundColor Yellow
    $settingsFile = $null
} else {
    # PowerShell 5.1's -Encoding utf8 means utf8-WITH-BOM, and a leading U+FEFF makes Claude Code's
    # JSON parse of the settings file fail with nothing useful said about why.
    [IO.File]::WriteAllText($settingsFile, $settingsJson, (New-Object Text.UTF8Encoding($false)))
    $passthru = @('--settings', $settingsFile) + $passthru
}

# Claude Code's built-in WebSearch is executed server-side by Anthropic, so against a local model
# it silently returns nothing and the model burns turns retrying. WebFetch runs client-side and
# does work, so it stays. Skipped if the caller passes their own --disallowedTools.
if (-not ($passthru -contains '--disallowedTools' -or $passthru -contains '--disallowed-tools')) {
    $passthru = @('--disallowedTools', 'WebSearch') + $passthru
}

# Leave room under the context window for the reply plus a margin, so Claude Code compacts before
# llama-server has to truncate. This threshold is fixed for the life of the session while the model
# behind it can change at any moment, so it is measured against the SMALLEST context in the picker
# (sessionCtx) rather than the model that happens to be loaded now.
$window = if ($state.sessionCtx) { [int]$state.sessionCtx } else { [int]$state.ctx }
$compact = [Math]::Max(16384, $window - 16384)

# Every model-shaped variable points at the auto alias rather than at a real model. Claude Code
# fires Haiku-class side-requests (conversation titles, summaries) alongside the main conversation,
# and naming a real model in these would make each one swap the model out mid-turn - minutes of
# reloading for a request whose answer is a six-word title. The router serves the auto alias from
# whatever is already resident, so only an actual /model choice moves anything.
$vars = [ordered]@{
    ANTHROPIC_BASE_URL                       = $state.baseUrl
    ANTHROPIC_AUTH_TOKEN                     = 'claudelocal'
    ANTHROPIC_MODEL                          = 'claudelocal-auto'
    ANTHROPIC_DEFAULT_OPUS_MODEL             = 'claudelocal-auto'
    ANTHROPIC_DEFAULT_SONNET_MODEL           = 'claudelocal-auto'
    ANTHROPIC_DEFAULT_HAIKU_MODEL            = 'claudelocal-auto'
    ANTHROPIC_SMALL_FAST_MODEL               = 'claudelocal-auto'
    CLAUDE_CODE_SUBAGENT_MODEL               = 'claudelocal-auto'
    # Without this, a model Claude Code does not recognise is assumed to have a 200k window - so a
    # session pinned to 32768 would sail past its real limit and only find out when llama-server
    # rejected the prompt. Claude Code says as much on startup if it is left unset.
    CLAUDE_CODE_MAX_CONTEXT_TOKENS           = "$window"
    CLAUDE_CODE_AUTO_COMPACT_WINDOW          = "$compact"
    CLAUDE_CODE_MAX_OUTPUT_TOKENS            = '8192'
    # A 27B model mostly running out of system RAM answers in minutes, not seconds.
    API_TIMEOUT_MS                           = '3600000'
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
    DISABLE_TELEMETRY                        = '1'
    DISABLE_ERROR_REPORTING                  = '1'
    # Clear anything that would hijack the request to another provider, for this run only:
    ANTHROPIC_API_KEY                        = $null
    CLAUDE_CODE_USE_OPENAI                   = $null
    OPENAI_BASE_URL                          = $null
    OPENAI_API_KEY                           = $null
    CLAUDE_CODE_USE_BEDROCK                  = $null
    CLAUDE_CODE_USE_VERTEX                   = $null
}

$saved = @{}
foreach ($k in $vars.Keys) {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k)
    if ($null -eq $vars[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    else { Set-Item "Env:$k" $vars[$k] }
}

# A remote has no memory plan to report - nothing was loaded on this machine - so the usual
# ctx/KV/layers line would print three empty fields.
if (-not $state.alias) {
    Write-Host "[claudelocal] Claude Code -> nothing loaded yet  (session context $($state.sessionCtx))" -ForegroundColor DarkGray
    Write-Host "[claudelocal] press /model and pick one - choosing a model is what loads it." -ForegroundColor DarkGray
} elseif ($state.source -eq 'remote') {
    Write-Host "[claudelocal] Claude Code -> $($state.alias) (remote) @ $($state.baseUrl)  (ctx $($state.ctx), nothing loaded locally)" -ForegroundColor DarkGray
    Write-Host "[claudelocal] /model switches between local models and remotes - a remote switch is instant." -ForegroundColor DarkGray
} else {
    Write-Host "[claudelocal] Claude Code -> $($state.alias) @ $($state.baseUrl)  (ctx $($state.ctx), KV $($state.kvType), $($state.nGpuLayers) layers on GPU)" -ForegroundColor DarkGray
    Write-Host "[claudelocal] /model switches between every model - a local swap costs one model load." -ForegroundColor DarkGray
}
try {
    $claude = Get-Command claude -ErrorAction SilentlyContinue
    if ($claude) { & $claude.Source @passthru }
    else { & "$env:USERPROFILE\.local\bin\claude.exe" @passthru }
}
finally {
    foreach ($k in $vars.Keys) {
        if ($null -eq $saved[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
        else { Set-Item "Env:$k" $saved[$k] }
    }
    if ($settingsFile -and (Test-Path $settingsFile)) { Remove-Item $settingsFile -Force -ErrorAction SilentlyContinue }
}
