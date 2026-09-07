# claudelocal

Claude Code, running against **any local model on this machine** — switchable from `/model` without
restarting — on a 6 GB RTX 3050 and 32 GB of RAM, without freezing the desktop.

```
claudelocal
```

Then, inside Claude Code:

```
/model
```

and every GGUF on the machine is in the list. Pick one and the next message loads it.

---

## What this is

Three model managers usually end up installed on the same box, and none of them agree on anything:

| | where the weights live | how a model is named |
|---|---|---|
| **claudelocal** | `models/*.gguf` | the filename |
| **LM Studio** | `~/.lmstudio/models/<publisher>/<repo>/` | the filename, which may be `ggml-model-f16.gguf` |
| **ollama** | content-addressed blobs — the file on disk is called `sha256-3ab8…` | only in an OCI manifest |

All three are plain GGUF files that `llama-server` can open directly, so nothing is converted or
copied. claudelocal finds them, gives each one a short stable alias, works out a memory plan for it
from its own tensor table, and puts the whole list into Claude Code's `/model` picker.

On this machine that is **12 usable models across three managers**, from 774 MB to 19.7 GB (five more
are found and rejected, with the reason):

```
$ claudelocal --cl-models

12 local models  (source: claudelocal/models, LM Studio, ollama)

            alias                       size     ctx    arch        vision  where
     qwen3.5-35b-a3b               19.7 GB  128000 qwen35moe   yes     lmstudio
     glm-4.7-flash-heretic         16.2 GB  128000 deepseek2   -       lmstudio
  -> qwen3.8-27b                   15.3 GB  128000 qwen35      yes     claudelocal
     gpt-oss-20b                   11.3 GB  128000 gpt-oss     -       lmstudio
     gemma-4-12b-it-heretic         6.9 GB  128000 gemma4      -       ollama
     ...

  session context 128000 - the smallest native window in this list. Every model in the picker
  shares one window, because Claude Code fixes its context limit at startup while the model
  behind it can change at any moment. Claude Code's opening prompt is ~33522 tokens of it.
  lfm2.5-1.2b pins it. '--min-ctx 128001' drops it and the rest share 131072.

Not offered:
  Qwen2.5-0.5B-Instruct-Q4_K_M.gguf   incomplete download (40 MB of 379 MB)
  qwen2.5-coder:7b                    native context 32768 cannot hold Claude Code's 33522-token opening prompt
  nomic-embed-text                    native context 2048 cannot hold Claude Code's 33522-token opening prompt
```

The `ctx` column is the *shared* window every model in the picker has to agree on, not each model's
native maximum — see below.

There are four parts:

1. **A registry** that scans all three managers, reads each GGUF header, and rejects what cannot
   work — with the reason.
2. **A planner** that reads the real tensor table and live free VRAM/RAM, then decides how many
   layers go on the GPU, what the KV cache is made of, and whether the plan is safe.
3. **A guard** that runs `llama-server` inside a Windows Job Object with a commit ceiling, at
   below-normal priority, under a watchdog on available physical memory.
4. **A router** — the shim proxy — that watches the `model` field on every request and swaps the
   loaded model when it changes, holding the request open while it does.

## How `/model` switching works

Claude Code sends the model it wants on **every** request, so a `/model` choice arrives at the proxy
as a different `model` field. One 6 GB card means exactly one model can be resident, so a switch is
an unload and a load. The proxy does it inside the request that asked for it, because there is
nowhere else to put the wait.

Two things make it usable rather than a curiosity:

**The picker is generated, not hand-written.** Claude Code will curate its own model list from a
`modelPicker` block in settings, but only when that block comes from managed settings, `--settings`
or user settings. So the launcher generates one per run and passes it with `--settings`, scoped to
that child process — the user's `~/.claude/settings.json` is never touched, and plain `claude` still
sees Anthropic's own lineup:

```json
{
  "modelPicker": {
    "options": [
      { "model": "claudelocal-auto", "label": "Auto (whatever is loaded)",
        "description": "Serve from the model already in memory - never triggers a swap",
        "behavesAs": "claude-sonnet-4-5" },
      { "model": "qwen3.8-27b", "label": "Qwen3.8-27B",
        "description": "27B · Q4_K_M · 15.3 GB on disk · 256k ctx · claudelocal",
        "behavesAs": "claude-sonnet-4-5" }
    ],
    "replaceBuiltInOptions": true
  }
}
```

`behavesAs` is not optional. Claude Code will not offer a row for a model it has never heard of
unless it is told whose client-side handling to borrow, and without it the model is also assumed to
have a 200k context window. It says so itself if you leave it out.

**Side-requests do not move the model.** Claude Code interleaves small Haiku-class calls —
conversation titles, summaries — with the main conversation. If those named a real model, every one
would swap the 15 GB model out mid-turn: minutes of reloading to produce a six-word title. So every
model-shaped environment variable (`ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`,
`CLAUDE_CODE_SUBAGENT_MODEL`, …) points at the alias `claudelocal-auto`, which the router serves
from whatever is already resident. Only a real `/model` choice moves anything.

Measured on this machine, switching between two small models:

```
[04:04:59] switching model: qwen3.5-0.8b -> gemma-3-4b-it
[04:05:07] switched to gemma-3-4b-it in 8s
```

During a longer swap the connection is held open with synthetic SSE `ping` frames — 12 of them over
a 125-second switch, so the client never sees more than ten seconds of silence.

### The one real cost: a shared context window

Claude Code fixes its auto-compact threshold once, at startup, but the model behind it can change at
any moment. A window larger than some model's native context means that model is handed a
conversation it cannot hold. So the session is pinned to the **smallest** context in the picker, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` is set to match.

This is also why a model needs a **native window of at least 65536** to be offered at all. Claude
Code's opening prompt - system prompt, tool definitions, environment block - is **~33,500 tokens
before you have typed anything**, measured on this machine. A 32768-token model therefore fails on
the *first* message, and auto-compact cannot rescue it because there is no conversation to compact
yet:

```
API Error: 400 request (33096 tokens) exceeds the available context size (32768 tokens)
```

Worse, such a model dragged every *other* model down to 32768 with it just by being in the picker.
So models that cannot hold the opening prompt are kept out by default, and the four 32k Qwen2.5
models here are listed under "Not offered" with that reason.

The remaining 12 share 128000. Narrowing further trades models for window:

```
claudelocal --cl-min-ctx 128001     11 models, 131072 shared window
claudelocal --cl-min-ctx 262144      5 models, 262144 shared window
```

`claudelocal --cl-models` always prints which model is pinning the window and what to pass to drop
it, and warns before starting if the window cannot hold the opening prompt. If a turn overflows
anyway, the proxy rewrites llama-server's error into the flag that fixes it rather than passing on
"try increasing it".

## The memory plan

The planner is the reason a 27B model and a 262144-token context fit on a 6 GB card at all. It reads
the real per-tensor byte counts out of the GGUF and the live free VRAM/RAM, and splits its RAM
accounting along the line that actually matters on Windows:

- **hard budget** — committed memory (KV cache + runtime). Overrunning this is a refusal to start.
- **soft budget** — mmap'd weights. Overrunning this only means pages get re-read from disk: slower
  per token, never a freeze. It is a warning, not a refusal.

Every model in the picker, planned on an idle desktop (23.4 GB RAM free, 5.5 GB VRAM free) at the
shared 128000-token window:

| model | arch | on disk | KV | KV size | GPU layers | committed RAM |
|---|---|---|---|---|---|---|
| `qwen3.5-35b-a3b` | qwen35moe | 19.7 GB | f16 | 2.44 GB | 7/40 | 5.81 GB |
| `glm-4.7-flash-heretic` | deepseek2 | 16.2 GB | q8_0 | 6.48 GB | 9/47 | 8.89 GB |
| `qwen3.8-27b` | qwen35 | 15.3 GB | q8_0 | 4.41 GB | 13/65 | 11.57 GB |
| `gpt-oss-20b` | gpt-oss | 11.3 GB | f16 | 5.86 GB | 6/24 | 13.51 GB |
| `gemma-4-12b-it-heretic` | gemma4 | 6.9 GB | f16 | 2.27 GB | 24/48 | 6.73 GB |
| `qwen3.5-9b` | qwen35 | 5.2 GB | f16 | 3.91 GB | 16/32 | 9.51 GB |
| `gemma-4-e4b-it` | gemma4 | 5.0 GB | f16 | 3.45 GB | 30/42 | 7.27 GB |
| `deepseek-r1-7b` | qwen2 | 4.4 GB | f16 | 6.84 GB | 11/28 | 13.63 GB |
| `gemma-3-4b-it` | gemma3 | 2.3 GB | f16 | 2.55 GB | 34/34 | 5.44 GB |
| `qwen3.5-0.8b-toolcalling` | qwen35 | 1.4 GB | f16 | 1.46 GB | 24/24 | 4.16 GB |
| `lfm2.5-1.2b` | lfm2 | 1.2 GB | f16 | 1.46 GB | 16/16 | 3.08 GB |
| `qwen3.5-0.8b` | qwen35 | 774 MB | f16 | 1.46 GB | 24/24 | 3.94 GB |

`claudelocal --cl-plan` prints the full working for the loaded one, plus a context/speed trade-off
table.

### Four families, four different KV caches

A single `n_head_kv × head_dim × 2` figure is right for a plain dense transformer and wrong for
everything else in that table — each in a different direction, and each worth gigabytes:

- **Hybrid** (Qwen3.8, LFM2). `head_count_kv` is a per-layer *array* and its zeros are
  recurrent/short-convolution layers that keep no KV at all. LFM2.5-1.2B has 16 blocks and only 6 of
  them cache anything.
- **Sliding-window** (Gemma 3/4). Most layers never cache more than `sliding_window` tokens no
  matter how long the context is, and they use a smaller `key_length_swa`. Gemma 3 declares the
  window but *not* the pattern, because its 5-local-then-1-global ratio is fixed in the
  architecture; llama.cpp hard-codes the same ratio. Missing that billed `gemma-3-4b-it` **17 GB**
  of KV at 131072 tokens instead of 2.6 GB, and wrote it off as unloadable.
- **MLA** (DeepSeek, GLM). One latent cache, `key_length` 576 against `value_length` 512, and no
  `attn_k` tensor at all — the projection is called `attn_kv_a_mqa`. Detecting attention layers by
  `attn_k` alone reported GLM-4.7-Flash as having *no KV cache whatsoever*: a 6 GB hole in the
  budget rather than a rounding error.
- **MoE** (Qwen3.5-35B-A3B, gpt-oss). Every expert's weights must be resident even though only a few
  run per token, so the weight side is sized like a dense model of the full parameter count.

So KV is modelled per layer, from `key_length` and `value_length` separately, clamped by each
layer's own attention window. Where a model declares a sliding window but no pattern and the
architecture is not one we know the ratio for, the discount is **not** taken — overestimating KV
only costs a shorter context, while underestimating it is what makes a load die half-way through.

## Why it does not freeze the machine

Four independent brakes, weakest to strongest:

| # | Brake | What it prevents |
|---|---|---|
| 1 | `--load-mode mmap`, never `mlock` | Model weights are **clean file-backed pages**. Under memory pressure Windows drops them and re-reads from disk. They are never written to the pagefile — so 15 GB of weights cannot stall the system on their own. |
| 2 | `--prio -1`, 4 of 12 threads | The desktop always wins a scheduling fight. The machine stays interactive while the model saturates memory bandwidth. |
| 3 | **Job Object commit cap** | The KV cache and llama.cpp's host buffers *are* committed memory — the only part of the footprint that can reach the pagefile. The cap makes an overrun fail inside `llama-server` instead of pushing the rest of Windows to disk. |
| 4 | **Memory watchdog** | Samples available physical RAM every 2 s. Below the soft floor it trims the server's working set; sustained below the hard floor it stops the server. A dead model beats a frozen machine. |

Closing the Job Object handle kills anything inside it (`KILL_ON_JOB_CLOSE`), so a model switch — or
`claudelocal --cl-down` — can never leave 15 GB of mapped weights behind.

## The flagship configuration

`Qwen/Qwen3.8-27B` (Apache 2.0) is a **dense** 27B multimodal model with a **hybrid attention
stack**: 48 DeltaNet (linear/SSM) layers that keep a small fixed recurrent state, and 16 full-
attention layers that keep a real KV cache. That ratio is why a 262144 context is reachable on a
consumer box at all — the KV cache costs about 19–68 KB per token depending on how it is quantized,
where a conventional 27B dense model would want roughly 130 KB.

Q4_K_M contains **every one of the 27B parameters**, quantized to ~4.8 bits each with an importance
matrix. Nothing is pruned. Full BF16 would be 53.8 GB, which does not fit in 32 GB at any context
length.

```
Ryzen 5 5500 (6c/12t) - 32 GB RAM - RTX 3050 6 GB

context    262144 tokens          KV q4_0 = 4.78 GB at 19.1 KB/token
VRAM       12 of 65 layers  ->  2.95 GB weights + 1.13 GB KV + 1.25 GB state/reserve  = 5.32 GB
RAM  hard  3.66 KV + 4.78 MTP draft KV + 1.64 pinned tensors + 2.15 runtime = 12.22 GB committed
RAM  soft  10.74 GB of weights mmap'd, 0.00 GB streamed from disk
```

Four of those numbers are not obvious, and each was worth about a gigabyte of the estimate:

- **The MTP draft cache is full-size.** Speculative decoding builds a second context against the
  same model at the same `n_ctx`, and it gets no GPU layers of its own — so it commits a *whole*
  second KV cache in RAM.
- **`-ot`-pinned tensors are copied, not mapped.** `token_embd` and `output.weight` are 1.64 GB
  here, and they are committed memory, not evictable page cache.
- **The logits buffer scales with the vocabulary.** Qwen3.8's vocabulary is 248,320 entries, so
  llama.cpp's host output buffer is `248320 × n_batch × 4` bytes — a full gigabyte at `-b 1024`,
  larger than every other runtime allocation combined.
- **`--cache-ram` defaults to 8192 MiB.** That is 8 GB of committed memory for the server's prompt
  cache, allocated when a request *finishes*. Left alone it walks into the Job Object cap, and the
  failure looks like llama-server dying for no reason the moment a turn ends.

`claudelocal --cl-status` prints the estimate against the peak the guard actually measured, so the
model stays honest:

```
peak committed 13356 MB vs 12515 MB planned (+841 MB), cap 15417 MB
```

Shorter contexts spend KV *quality* rather than KV *size* — the planner takes the best cache that
fits, so the cache stays about the same size and gets more accurate:

| ctx | KV type | GPU layers | KV size | committed RAM | prompt cache |
|---|---|---|---|---|---|
| **262144** | q4_0 | 12 | 4.78 GB | 12.22 GB | none |
| 131072 | q8_0 | 12 | 4.52 GB | 11.75 GB | none |
| 98304 | q8_0 | 13 | 3.39 GB | 10.49 GB | 0.73 GB |
| 49152 | f16 | 13 | 3.19 GB | 10.49 GB | 1.08 GB |
| 32768 | f16 | 14 | 2.13 GB | 10.70 GB | 3.29 GB |

The KV type is chosen automatically: the best-quality cache that still leaves every CPU-side weight
resident in the page cache. Picking on the committed budget alone is a trap — an f16 cache at 262k
technically "fits" in free RAM and then evicts 11 GB of weights to disk.

### The prompt cache is the real cost of the full window

At 262144 tokens there is nothing left for a prompt cache. `llama-server` runs one slot (`-np 1`,
which MTP requires), Claude Code interleaves small side-requests with the main conversation, and any
one of them evicts that slot. With a prompt cache the conversation is restored; without one it is
re-read from the first token — **about five minutes** on this machine. So `--cl-ctx 32768` is not
only faster per token, it is the difference between an interruption costing seconds and costing five
minutes.

## Speed

Measured at ctx 262144, KV q4_0, 12 layers on the GPU:

| | |
|---|---|
| model load, 15.3 GB at full 262144 ctx | **33 s** |
| model load, 774 MB | 6 s |
| model **switch** between two small models | 8 s |
| prompt processing | 71.7 tok/s (21,273-token Claude Code prompt in 297 s) |
| generation, short context | 2.61 tok/s (88% MTP acceptance, mean accepted run 2.76) |
| generation, at 21k context | 1.56 tok/s (68% MTP acceptance, mean accepted run 2.37) |

Generation is memory-bandwidth bound: ~10.7 GB of mmap'd weights plus 1.6 GB of pinned tensors are
read from system RAM for every token. Two things claw some back:

- **MTP speculative decoding** (`--spec-type draft-mtp`). The multi-token-prediction head ships
  inside the GGUF as `blk.64.nextn.*`. A verification pass costs one sweep over the weights whether
  it confirms one token or three, so accepted drafts are nearly free — worth roughly 2.4× here.
- **`--cache-reuse 256`.** Claude Code edits the *middle* of its prompt constantly: system-reminders
  appear and vanish, tool results get trimmed, the todo list is rewritten in place. Plain prefix
  matching gives up at the first changed byte and reprocesses everything after it. This lets the
  server KV-shift the unchanged chunks back into place instead.

`claudelocal --cl-tune` measures both VRAM strategies end to end and keeps the faster:

- `layers` — whole blocks from the top down (llama.cpp's native `-ngl N`). One CPU/GPU boundary
  crossing per token. The safe default, and the only option on a non-hybrid model.
- `ssm-pack` — at long context a full-attention layer drags ~280 MB of KV onto the card with it,
  against 168 MB for a whole DeltaNet layer's weights, so VRAM buys ~40% more model if it holds only
  DeltaNet layers. Costs a boundary crossing per layer, and depends on how llama.cpp places the KV
  for `-ot`-overridden layers — which is why it is *measured* rather than assumed.

## Commands

```
claudelocal                    load a model and start Claude Code
claudelocal -p "..."           any normal claude argument passes straight through

claudelocal --cl-models        every local model found, and why anything was left out
claudelocal --cl-model ALIAS   load a specific model
claudelocal --cl-min-ctx N     drop models below N tokens, raising the shared window
claudelocal --cl-rescan        re-read every GGUF header instead of trusting models.json

claudelocal --cl-plan          memory plan + context/speed trade-off table
claudelocal --cl-status        server / proxy / guard / RAM / VRAM
claudelocal --cl-bench         measure tokens per second
claudelocal --cl-tune          try both VRAM strategies, keep the faster
claudelocal --cl-doctor        full environment dump
claudelocal --cl-down          unload, free all RAM and VRAM
claudelocal --cl-pull          download / resume the bundled Qwen3.8-27B weights
claudelocal --cl-install       download the pinned llama.cpp CUDA runtime

claudelocal --cl-ctx 65536         cap the context, putting more model on the GPU
claudelocal --cl-kv q8_0           better KV quality (needs more RAM)
claudelocal --cl-strategy ssm-pack
claudelocal --cl-restart           reload even if a server is already up
```

Plain `claude` is untouched — every provider variable is set for the child process only and restored
afterwards, and the model picker is passed with `--settings` so it dies with the session.

## Install

```
git clone https://github.com/Everaldtah/claudelocal
cd claudelocal
powershell -ExecutionPolicy Bypass -File install.ps1
```

That puts `claudelocal` on PATH, fetches the pinned llama.cpp CUDA runtime, and reports what it
found. If LM Studio or ollama already have models installed, there is nothing else to download —
`claudelocal --cl-models` will list them. Otherwise `claudelocal --cl-pull` fetches Qwen3.8-27B
(~17 GB, resumable).

## Layout

```
src/registry.mjs   finds every GGUF across claudelocal / LM Studio / ollama, names them
src/gguf.mjs       GGUF header + tensor table reader (exact byte sizes, works on a partial download)
src/sys.mjs        live RAM / VRAM / CPU / disk probes
src/plan.mjs       the planner: layer placement, per-layer KV modelling, hard vs soft RAM budgets
src/guard.ps1      Job Object commit cap, below-normal priority, memory watchdog
src/serve.mjs      starts llama-server under the guard, waits for health, switches models
src/proxy.mjs      the router + Claude Code -> llama-server /v1/messages normalizing shim
src/cli.mjs        models / settings / plan / up / down / status / bench / tune / doctor / pull
bin/claudelocal.ps1   the launcher: loads a model, builds the /model picker, execs claude
runtime/           llama.cpp CUDA build (b10819)   [not in git]
models/            GGUF weights                    [not in git]
logs/              server.log, guard.log, guard.json, proxy.log
```

## Requirements and gotchas

Most of these cost hours to find, so they are written down rather than fixed silently.

- **Never wait on a process whose grandchild outlives it.** The guard is launched through
  PowerShell's `Start-Process`, and Node's `execFileSync` resolves when the child's *pipes* close,
  not when it exits. `llama-server`, three processes down, inherits those handles and holds them for
  its entire life — so the launcher call blocked until its own 120-second timeout on **every single
  start**. It was invisible because the health poll that follows then succeeded immediately, so the
  wasted two minutes looked like load time. Spawning with `stdio: "ignore"` and waiting on `close`
  took a full-context 15.3 GB load from 123 s to **33 s**, and a small-model switch to 8 s.
- **Model loading must not block the event loop.** The router loads models from inside a live HTTP
  request, and the only thing stopping Claude Code from abandoning that request is a ping written
  every few seconds. Any synchronous child-process call in the load path stops those timers dead.
- **A slow local model needs synthetic SSE pings.** Claude Code treats a stream that produces
  nothing for about a minute as a dead connection: it aborts and retries with a growing backoff.
  `llama-server` emits essentially nothing while it processes the prompt, and a Claude Code prompt is
  ~21k tokens — five minutes of silence here — so every attempt was killed at ~74 s and restarted, a
  loop that can never finish a turn. Anthropic's own API sends periodic `ping` events for exactly
  this reason, so the proxy synthesizes them. It is driven by an **idle** timer, not "ping until the
  first byte": llama.cpp sends one bare `:` comment early and then goes quiet again, which is enough
  to stop a naive pinger and not enough to keep the client happy.
- **Claude Code's opening prompt is ~33,500 tokens**, not the ~21k the system prompt alone accounts
  for — tool definitions and the environment block make up the rest. A 32768-token model therefore
  cannot run Claude Code *at all*: the first message overflows and auto-compact has nothing to
  compact. Those models are kept out of the picker, both because they cannot work and because one of
  them in the picker pins every other model to its window.
- **llama.cpp build ≥ b10450.** Earlier builds have a broken CUDA path for Qwen3.8's DeltaNet layers
  and produce fluent garbage while looking completely healthy. `claudelocal --cl-doctor` checks the
  build number. This repo pins b10819.
- **`--jinja` is mandatory**, and a GGUF without a chat template is skipped: there is no reliable
  turn boundary and tool calling does not work.
- **Flash attention on.** Required for a quantized KV cache, and it removes an attention scratch
  buffer that would otherwise eat the VRAM reserve at long context.
- **`--cache-ram` must be set explicitly.** See above — its 8192 MiB default is enough on its own to
  walk into the commit cap.
- **Windows PowerShell 5.1's `-Encoding utf8` means utf8-with-BOM**, and `JSON.parse` throws on a
  leading U+FEFF. `guard.ps1` writes the status file every 2 s; when that file was BOM'd, every
  Node-side reader silently got `null` — including the one whose whole job is to notice llama-server
  dying during a load, which then hung for the full 15-minute timeout instead. Everything here
  writes UTF-8 without a BOM and strips one on read. The generated settings file is written the same
  way for the same reason.
- **`Process` performance counters are snapshots.** Without `$proc.Refresh()` the guard reported
  llama-server's start-up working set for the entire run, so the commit cap could never be seen
  approaching.
- **A GGUF header parses perfectly while the file is still downloading** — the header is written
  first. The registry compares declared tensor bytes against the file on disk and reports the
  partial download rather than handing it to llama-server.
- Node 20+, an NVIDIA GPU with a CUDA 12.4-capable driver, and disk for whatever models you want.

## Sources

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [unsloth GGUF quants](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [llama.cpp server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164) — the DeltaNet CUDA bug and the b10450 fix
- [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) — the MTP speculative-decoding flags

## License

MIT
