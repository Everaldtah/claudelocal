#!/usr/bin/env node
// claudelocal - run Claude Code against a local Qwen3.8-27B without freezing the machine.
import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { paths, ram, cpu, gpu, diskFree, fmtGB, GiB, readState, writeState, httpJSON } from "./sys.mjs";
import { inspect, plan, explain, ladder, serverArgs } from "./plan.mjs";
import { up, down, health, findModel, findMmproj, serverExe, guardStatus, tailLog, DEFAULT_PORT, DEFAULT_PROXY_PORT } from "./serve.mjs";
import { PROXY_VERSION } from "./proxy.mjs";
import { discover, resolve as resolveModel, pickerRows, human, AUTO_ALIAS, MIN_CTX, CLAUDE_CODE_BASE_TOKENS } from "./registry.mjs";
import { isRemote, remoteKey, REMOTES_FILE } from "./remotes.mjs";

const argv = process.argv.slice(2);
const cmd = argv[0] ?? "help";
const flag = (n) => argv.includes(`--${n}`);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : d; };
const num = (n, d) => { const v = opt(n, null); return v === null ? d : parseInt(v, 10); };

const PORT = num("port", DEFAULT_PORT);
const PROXY_PORT = num("proxy-port", DEFAULT_PROXY_PORT);

// --- the model list ------------------------------------------------------------------------------
const MIN_PICKER_CTX = num("min-ctx", MIN_CTX);
const registry = () => discover({ refresh: flag("rescan"), minCtx: MIN_PICKER_CTX });

// Which model a bare `claudelocal` should load. `--model` wins; otherwise whatever was loaded last,
// so switching in /model sticks across restarts; otherwise the model this repo downloaded itself.
function bootModel(models) {
  const asked = opt("model", null);
  if (asked) {
    const m = resolveModel(asked, models);
    if (!m) die(`no local model matches "${asked}"\n  run: claudelocal models`);
    return m;
  }
  // `alias` is what is loaded RIGHT NOW and is null while nothing is; `lastAlias` is what was
  // loaded last, which is what "the same model as yesterday" has to mean once a bare start loads
  // nothing at all.
  const st = readState();
  const last = st.alias ?? st.lastAlias;
  return (last && resolveModel(last, models))
    ?? models.find((m) => m.source === "claudelocal")
    ?? models[0];
}

// Claude Code fixes its auto-compact window once, at startup, but the model can change underneath
// it every time the user picks a different row. A window larger than some model's native context
// means that model is asked to hold a conversation it cannot, so the session is pinned to the
// SMALLEST context in the picker. Narrowing the picker with --min-ctx is what buys it back.
function sessionCtx(models) {
  const cap = num("ctx", 0);
  const per = models.map((m) => (cap ? Math.min(cap, m.nCtxTrain) : m.nCtxTrain));
  return per.length ? Math.min(...per) : (cap || MIN_CTX);
}

// A window that cannot hold Claude Code's opening prompt does not degrade gracefully - the very
// first message comes back as a 400 and nothing works at all. Say so up front rather than letting
// it be discovered one turn in.
function warnIfTooSmall(ctx) {
  if (ctx > CLAUDE_CODE_BASE_TOKENS) return;
  say();
  say(`WARNING  the session context is ${ctx} tokens, and Claude Code's opening prompt alone is`);
  say(`         about ${CLAUDE_CODE_BASE_TOKENS}. The first message will fail with "exceeds the available`);
  say("         context size\". Raise --ctx, or --min-ctx to drop the model pinning it.");
  say();
}

const planOpts = () => ({
  ctx: num("ctx", undefined),
  kvType: opt("kv", "auto"),
  strategy: opt("strategy", readState().strategy ?? "layers"),
  ramMarginGB: parseFloat(opt("ram-margin", "2")),
  ubatch: num("ubatch", 256),
  batch: num("batch", 1024),
  vision: flag("vision"),
  force: flag("force"),
  restart: flag("restart"),
  port: PORT,
});

const die = (m) => { console.error(`\n[claudelocal] ${m}\n`); process.exit(1); };
const say = (m = "") => console.log(m);

// --- proxy lifecycle ---------------------------------------------------------------------------
async function proxyAlive(port) {
  const r = await httpJSON(`http://127.0.0.1:${port}/claudelocal/health`, { timeout: 1500 });
  return r.ok ? r.json : null;
}

async function proxyUp(port, target, ctx = 0) {
  const cur = await proxyAlive(port);
  // A proxy left over from an earlier run keeps whatever rewriting rules it was started with, and
  // those rules are the only thing standing between Claude Code's request shape and a 400 from the
  // chat template. If it is older than this checkout, or pointed at a different llama-server,
  // replace it rather than trusting it.
  // A proxy started for a different session context would load a switched-to model at the wrong
  // window, so that counts as stale too.
  if (cur && cur.version === PROXY_VERSION && cur.target === target && cur.ctx === ctx) return cur;
  if (cur) {
    await proxyDown(port);
    for (let i = 0; i < 20 && await proxyAlive(port); i++) await new Promise((r) => setTimeout(r, 100));
  }
  const child = spawn(process.execPath, [path.join(paths.root, "src", "proxy.mjs"), "--port", String(port), "--target", String(target), "--ctx", String(ctx)],
    { detached: true, stdio: "ignore", windowsHide: true });
  child.unref();
  for (let i = 0; i < 40; i++) {
    const h = await proxyAlive(port);
    if (h) return h;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`proxy did not come up on :${port}`);
}

async function proxyDown(port) {
  await httpJSON(`http://127.0.0.1:${port}/claudelocal/quit`, { timeout: 1500 }).catch(() => {});
}

// --- commands --------------------------------------------------------------------------------
async function cmdPlan() {
  const m = requireModel();
  const info = inspect(m);
  // The planner reads LIVE free RAM and VRAM, so planning while a server is already up measures a
  // machine that is holding the model - every number comes out as if a second copy had to fit, and
  // it reports a plan that cannot possibly work. Say so rather than let it be read as a verdict.
  if (await health(PORT)) {
    const st = readState();
    say();
    say(`NOTE  a server is already running on :${PORT} holding this model, so the free RAM and VRAM`);
    say("      below are what is left AFTER it loaded - not what a fresh start would see.");
    say(`      It was started with: ctx ${st.ctx}, KV ${st.kvType}, ${st.nGpuLayers} layers on the GPU.`);
    say("      Run 'claudelocal down' first for a plan you can act on.");
  }
  const p = plan(info, planOpts());
  say();
  say(explain(p));
  say();
  say("What the other context lengths would cost on this machine right now:");
  say("  ctx        KV      GPU layers   GPU weights   KV size     committed RAM   prompt cache   streamed from disk");
  for (const r of ladder(info, planOpts())) {
    say(`  ${String(r.ctx).padEnd(10)} ${r.kvType.padEnd(7)} ${String(r.ngl).padEnd(12)} ${fmtGB(r.gpuWeights).padEnd(13)} ${fmtGB(r.kv).padEnd(11)} ${fmtGB(r.committed).padEnd(15)} ${(r.cacheRam ? fmtGB(r.cacheRam) : "none").padEnd(14)} ${fmtGB(r.streamed)}${r.ok ? "" : "   <- will not fit"}`);
  }
  say();
  say("  'prompt cache' is what survives a side-request evicting the slot. With none, a conversation");
  say("  that gets bumped is re-read from the first token - about 5 minutes for a Claude Code prompt.");
  say();
  say("llama-server command line:");
  // Match what `up` would actually run: the vision projector is only loaded when --vision is
  // passed, so printing it unconditionally showed a command line nobody would ever get.
  say("  " + serverArgs(p, { port: PORT, mmproj: flag("vision") ? findMmproj() : null, alias: ALIAS }).join(" "));
  say();
}

function requireModel() {
  const { models } = registry();
  // `plan` and `bench` measure THIS machine, so a remote has nothing for them to measure.
  const local = models.filter((x) => !isRemote(x));
  const boot = local.length ? bootModel(local) : null;
  if (boot && isRemote(boot)) die(`${boot.alias} runs remotely - there is no local memory plan for it`);
  const m = boot ? boot.file : findModel(opt("model", null));
  if (!m) die(`no usable local model found\n  run: claudelocal models   (or: claudelocal pull)`);
  const size = fs.statSync(m).size;
  if (size < 1 * GiB) die(`${path.basename(m)} looks incomplete (${fmtGB(size)}) - the download may still be running.\n  run: claudelocal pull   (it resumes)`);
  return m;
}

async function cmdUp() {
  const t0 = Date.now();
  const { models } = registry();
  if (!models.length) die(`no usable local models found\n  run: claudelocal pull   (or install one in LM Studio / ollama)`);
  const ctx = sessionCtx(models);

  // --- the idle start ---------------------------------------------------------------------------
  // A bare `claudelocal` loads NOTHING. Only the proxy goes up - which is all that /model needs to
  // list every model and to load the one that gets picked. Starting the CLI should not commit two
  // minutes and 15 GB to a model chosen for you by what you happened to run last.
  if (flag("no-load") && !opt("model", null)) {
    warnIfTooSmall(ctx);
    const st = readState();
    const live = await health(PORT);
    // A server left up by an earlier session is not evicted - it is already resident, so nothing
    // new gets loaded either way and switching back to it costs nothing.
    if (live && st.model) {
      say(`[claudelocal] ${st.alias} is still loaded from an earlier session (ctx ${st.ctx})`);
    } else {
      writeState({
        lastAlias: st.alias ?? st.lastAlias ?? null,
        alias: null, modelName: null, source: null, model: null,
        ctx: null, kvType: null, ngl: null, nGpuLayers: null, vision: false,
      });
      say(`[claudelocal] nothing loaded - open /model and pick one; that is what loads it`);
    }
    const px = await proxyUp(PROXY_PORT, PORT, ctx);
    say(`[claudelocal] proxy on http://127.0.0.1:${PROXY_PORT} (pid ${px.pid}), ${models.length} models switchable from /model`);
    writeState({ proxyPort: PROXY_PORT, baseUrl: `http://127.0.0.1:${PROXY_PORT}`, sessionCtx: ctx });
    return { idle: true, port: PROXY_PORT };
  }

  const boot = bootModel(models);
  warnIfTooSmall(ctx);

  // A remote needs no llama-server at all: nothing is planned, nothing is loaded, and the machine's
  // RAM and VRAM are never touched. The proxy still goes up - it is what Claude Code is pointed at
  // and what routes a later /model switch back to a local model.
  if (isRemote(boot)) {
    if (!remoteKey(boot) && boot.remote.apiKeyEnv) {
      say(`[claudelocal] WARNING  ${boot.alias} authenticates with ${boot.remote.apiKeyEnv}, which is not set.`);
      say(`[claudelocal]          Set it, or put "apiKey" in ${REMOTES_FILE}, or requests will fail.`);
    }
    writeState({
      alias: boot.alias, lastAlias: boot.alias, modelName: boot.name, source: "remote", model: null,
      ctx: Math.min(ctx, boot.nCtxTrain), kvType: null, ngl: null, nGpuLayers: null, vision: false,
    });
    say(`[claudelocal] ${boot.alias} is remote (${boot.remote.baseUrl}) - nothing loaded on this machine`);
    const pxr = await proxyUp(PROXY_PORT, PORT, ctx);
    say(`[claudelocal] proxy on http://127.0.0.1:${PROXY_PORT} (pid ${pxr.pid})`);
    say(`[claudelocal] ${models.length} models switchable from /model`);
    writeState({ proxyPort: PROXY_PORT, baseUrl: `http://127.0.0.1:${PROXY_PORT}`, sessionCtx: ctx });
    return { remote: true, port: PROXY_PORT };
  }

  say(`[claudelocal] starting ${boot.alias} (${boot.name}, ${human(boot.bytes)}, ${boot.source}) ...`);
  const r = await up({
    ...planOpts(),
    modelPath: boot.file, alias: boot.alias, source: boot.source,
    mmproj: planOpts().vision === false ? null : (boot.mmproj ?? null),
    ctx: Math.min(ctx, boot.nCtxTrain),
    onProgress: (line, g) => {
      const mem = g ? `  avail ${g.availMB} MB` : "";
      if (line) process.stdout.write(`\r  ${line.slice(0, 100).padEnd(102)}${mem}`);
    },
  });
  process.stdout.write("\r".padEnd(120) + "\r");
  if (r.already) { say(`[claudelocal] already running on :${r.port}`); }
  else { say(`[claudelocal] llama-server ready on :${r.port} in ${((Date.now() - t0) / 1000).toFixed(0)}s`); }
  // The proxy is told the session context so a model switched to later is loaded at the same
  // window the auto-compact threshold was computed against.
  const px = await proxyUp(PROXY_PORT, PORT, ctx);
  say(`[claudelocal] proxy on http://127.0.0.1:${PROXY_PORT} -> :${PORT} (pid ${px.pid})`);
  say(`[claudelocal] ${models.length} local models switchable from /model`);
  writeState({ proxyPort: PROXY_PORT, baseUrl: `http://127.0.0.1:${PROXY_PORT}`, sessionCtx: ctx });
  return r;
}

// Everything the launcher needs to configure Claude Code, as one JSON blob. Emitted rather than
// written to a file so that plain `claude` is never touched: it is handed to the child process
// through --settings and dies with it.
function cmdSettings() {
  const { models } = registry();
  const ctx = sessionCtx(models);
  const rows = pickerRows(models, { ctxOf: (m) => Math.min(ctx, m.nCtxTrain) });
  say(JSON.stringify({
    env: {},
    modelPicker: {
      options: rows,
      // Anthropic's hosted models are not reachable from here - the base URL points at
      // llama-server - so offering them would just be a list of rows that error.
      replaceBuiltInOptions: true,
    },
  }, null, 2));
}

async function cmdModels() {
  const { models, skipped } = registry();
  const st = readState();
  const ctx = sessionCtx(models);
  say();
  say(`${models.length} models  (claudelocal/models, LM Studio, ollama, and remotes.json)`);
  say();
  say("            alias                       size     ctx    arch        vision  where");
  for (const m of models) {
    const mark = m.alias === st.alias ? "  ->" : "    ";
    const size = isRemote(m) ? "remote" : human(m.bytes);
    say(`${mark} ${m.alias.padEnd(28)} ${size.padStart(8)} ${String(Math.min(ctx, m.nCtxTrain)).padStart(7)} ${m.arch.padEnd(11)} ${(m.mmproj ? "yes" : "-").padEnd(7)} ${m.source}`);
  }
  say();
  say(`  session context ${ctx} - the smallest native window in this list. Every model in the picker`);
  say("  shares one window, because Claude Code fixes its context limit at startup while the model");
  say(`  behind it can change at any moment. Claude Code's opening prompt is ~${CLAUDE_CODE_BASE_TOKENS} tokens of it.`);
  if (models.length) {
    const smallest = models.reduce((a, b) => (a.nCtxTrain <= b.nCtxTrain ? a : b));
    if (smallest.nCtxTrain === ctx && models.some((m) => m.nCtxTrain > ctx)) {
      const next = Math.min(...models.filter((m) => m.nCtxTrain > ctx).map((m) => m.nCtxTrain));
      say(`  ${smallest.alias} pins it. '--min-ctx ${ctx + 1}' drops it and the rest share ${next}.`);
    }
  }
  warnIfTooSmall(ctx);
  if (skipped.length) {
    say();
    say("Not offered:");
    for (const s of skipped) say(`  ${(s.ref ?? path.basename(s.file)).slice(0, 52).padEnd(54)} ${s.reason}`);
  }
  say();
  say(`  claudelocal --model <alias>    load one now`);
  say(`  /model inside Claude Code      switch without restarting (the swap takes a load's worth of time)`);
  say();
}

// What is configured in remotes.json, whether each one has the key it needs, and - with --check -
// whether it actually answers. The reachability probe matters most for a tunnelled llama-server:
// a Kaggle session hands out a new URL every time it starts, so "is this URL still alive" is the
// question asked most often.
async function cmdRemotes() {
  const { models } = registry();
  const rem = models.filter(isRemote);
  say();
  if (!fs.existsSync(REMOTES_FILE)) {
    say(`No remotes configured.`);
    say(`  Create ${REMOTES_FILE} - copy remotes.example.json beside it and edit.`);
    say();
    return;
  }
  if (!rem.length) {
    say(`${REMOTES_FILE} has no usable entries (a disabled or malformed one is listed by 'claudelocal models').`);
    say();
    return;
  }
  say(`${rem.length} remote models  (${REMOTES_FILE})`);
  say();
  say("     alias                       ctx   key                     endpoint");
  for (const m of rem) {
    const key = remoteKey(m);
    const ks = m.remote.apiKeyEnv ? (key ? `${m.remote.apiKeyEnv} set` : `${m.remote.apiKeyEnv} MISSING`) : (key ? "inline" : "none needed");
    say(`     ${m.alias.padEnd(24)} ${String(m.nCtxTrain).padStart(7)}   ${ks.padEnd(26)}  ${m.remote.baseUrl}`);
    const via = m.remote.api === "openai" ? " via the OpenAI bridge" : "";
    say(`     ${"".padEnd(24)} ${"".padStart(7)}   ${"".padEnd(26)}  sends model id "${m.remote.model}"${via}`);
    if (m.remote.fallback.length) {
      say(`     ${"".padEnd(24)} ${"".padStart(7)}   ${"".padEnd(26)}  falls back to ${m.remote.fallback.join(", ")} on 429 / 5xx`);
    }
  }
  if (flag("check")) {
    say();
    say("Probing each endpoint with a one-token request ...");
    for (const m of rem) {
      const key = remoteKey(m);
      // Probing without the key the config says to use would either be rejected upstream or -
      // worse, against an endpoint that does not enforce auth - come back "ok" and hide the fault.
      if (!key && m.remote.apiKeyEnv) {
        say(`  skip  ${m.alias.padEnd(24)} ${m.remote.apiKeyEnv} is not set - nothing to authenticate with`);
        continue;
      }
      // The probe has to speak whatever the endpoint speaks, or an OpenAI-only provider answers
      // 404 for a path it never published and the entry looks broken when it is fine.
      const bridge = m.remote.api === "openai";
      const headers = {};
      if (bridge) { if (key) headers.authorization = `Bearer ${key}`; }
      else {
        headers["anthropic-version"] = "2023-06-01";
        if (key) { headers.authorization = `Bearer ${key}`; headers["x-api-key"] = key; }
      }
      const url = bridge
        ? `${m.remote.baseUrl}${m.remote.path ?? "/v1/chat/completions"}`
        : `${m.remote.baseUrl}/v1/messages`;
      // The one-token probe body happens to be legal under both schemas, so only the path and the
      // auth headers differ.
      const body = { model: m.remote.model, max_tokens: 1, messages: [{ role: "user", content: "hi" }] };
      const t0 = Date.now();
      const r = await httpJSON(url, { method: "POST", timeout: 60000, headers, body: JSON.stringify(body) });
      const ms = Date.now() - t0;
      if (r.ok) say(`  ok    ${m.alias.padEnd(24)} answered in ${ms} ms`);
      // 429 means the endpoint is alive and the key is good - it is only saying "not right now",
      // which for a free tier is the most likely answer to a probe fired at every entry in a row.
      else if (r.status === 429) say(`  busy  ${m.alias.padEnd(24)} 429 rate-limited after ${ms} ms - endpoint and key are fine`);
      else say(`  FAIL  ${m.alias.padEnd(24)} ${r.status || "no response"}  ${String(r.error ?? JSON.stringify(r.json ?? {})).slice(0, 160)}`);
    }
  } else {
    say();
    say("  claudelocal remotes --check   actually call each one and time it");
  }
  say();
}

async function cmdDown() {
  await proxyDown(PROXY_PORT);
  const ok = await down(PORT);
  // `alias` means "resident right now", so unloading has to clear it or the next start would
  // report a model that is no longer in memory - and the router would skip loading it back.
  const st = readState();
  if (ok) writeState({ lastAlias: st.alias ?? st.lastAlias ?? null, alias: null, model: null, modelName: null, source: null, ctx: null, kvType: null, ngl: null, nGpuLayers: null, vision: false });
  say(`[claudelocal] ${ok ? "stopped" : "stop requested (server still responding - check Task Manager)"}`);
}

async function cmdStatus() {
  const st = readState();
  const h = await health(PORT);
  const px = await proxyAlive(PROXY_PORT);
  const g = guardStatus();
  const v = gpu(); const r = ram(); const c = cpu();
  say();
  say(`server     ${h ? `up on :${PORT}  (${h.status ?? "ok"})` : `down (:${PORT})`}`);
  say(`proxy      ${px ? `up on :${PROXY_PORT}  pid ${px.pid}` : `down (:${PROXY_PORT})`}`);
  const routed = px ? await httpJSON(`http://127.0.0.1:${PROXY_PORT}/claudelocal/models`, { timeout: 2000 }) : null;
  if (routed?.json?.switching) say(`switching  ${routed.json.switching}  (a load is in flight)`);
  if (!st.alias) say(`loaded     nothing${st.lastAlias ? `  (last used: ${st.lastAlias} - pick it in /model to load it)` : ""}`);
  if (st.alias) say(`loaded     ${st.alias}${st.source ? `  from ${st.source}` : ""}${st.vision ? "  vision on" : ""}`);
  if (st.modelName) say(`model      ${st.modelName}  ctx ${st.ctx}  KV ${st.kvType}  ngl ${st.ngl}  strategy ${st.strategy}${st.mtp ? "  MTP on" : ""}`);
  // The router resolves any model on the machine, while the picker only offers those that clear
  // --min-ctx - so these are two different numbers and saying so avoids a puzzle later.
  if (routed?.json?.models) say(`routable   ${routed.json.models.length} models the proxy can switch to  (claudelocal models)`);
  if (g) {
    say(`guard      pid ${g.pid}  ${g.running ? "running" : `exited (${g.exitCode}, ${g.reason})`}`);
    if (g.running) say(`           working set ${g.workingSetMB} MB, committed ${g.commitMB} MB of a ${g.commitCapMB} MB cap, ${g.trims} trims`);
    say(`           lowest free RAM seen: ${g.minAvailMB} MB`);
    // The planner's committed estimate is what sizes the cap, so show it against what actually
    // happened. A persistent gap here means the model of llama.cpp's allocations is wrong, and the
    // cap is only working by luck.
    if (g.peakCommitMB && st.est?.committed) {
      const estMB = Math.round(st.est.committed / (1024 * 1024));
      const d = g.peakCommitMB - estMB;
      say(`           peak committed ${g.peakCommitMB} MB vs ${estMB} MB planned (${d >= 0 ? "+" : ""}${d} MB), cap ${g.commitCapMB} MB`);
    }
  }
  say();
  say(`RAM        ${fmtGB(r.free)} free of ${fmtGB(r.total)}`);
  if (v) say(`VRAM       ${fmtGB(v.free)} free of ${fmtGB(v.total)}   (${v.name})`);
  say(`CPU        ${c.model}  ${c.physical} cores / ${c.logical} threads`);
  say();
}

async function cmdDoctor() {
  const v = gpu(); const r = ram(); const c = cpu();
  const exe = serverExe();
  const m = findModel(opt("model", null));
  say();
  say("=== claudelocal doctor ===");
  say(`root        ${paths.root}`);
  say(`node        ${process.version}`);
  say(`runtime     ${fs.existsSync(exe) ? exe : "MISSING - run: claudelocal install"}`);
  if (fs.existsSync(exe)) {
    try { say(`            ${execFileSync(exe, ["--version"], { encoding: "utf8", timeout: 20000 }).trim().split("\n")[0]}`); } catch (e) { say(`            version check failed: ${e.message}`); }
    const build = parseInt((fs.readFileSync(path.join(paths.runtime, "BUILD.txt"), "utf8").match(/\d+/) ?? [0])[0], 10);
    say(`            build b${build} ${build >= 10450 ? "(>= b10450, has the DeltaNet CUDA fix)" : "TOO OLD - Qwen3.8 emits garbage below b10450"}`);
  }
  say(`model       ${m ? `${m}  ${fmtGB(fs.statSync(m).size)}` : "MISSING - run: claudelocal pull"}`);
  say(`mmproj      ${findMmproj() ?? "none (vision disabled)"}`);
  say(`CPU         ${c.model}  ${c.physical}c/${c.logical}t`);
  say(`RAM         ${fmtGB(r.free)} free of ${fmtGB(r.total)}`);
  say(`GPU         ${v ? `${v.name}  ${fmtGB(v.free)} free of ${fmtGB(v.total)}` : "no NVIDIA GPU detected"}`);
  say(`disk        ${fmtGB(diskFree(paths.root) ?? 0)} free`);
  const claude = (() => { try { return execFileSync("where", ["claude"], { encoding: "utf8" }).trim().split(/\r?\n/)[0]; } catch { return null; } })();
  say(`claude      ${claude ?? "not on PATH"}`);
  if (m) { say(); say(explain(plan(m, planOpts()))); }
  say();
}

async function cmdBench() {
  const h = await health(PORT);
  if (!h) die("server is not running - run: claudelocal up");
  const prompt = opt("prompt", "Write a Python function that merges two sorted lists. Explain it in two sentences.");
  const nTok = num("tokens", 128);
  say(`[claudelocal] generating ${nTok} tokens ...`);
  const t0 = Date.now();
  const r = await httpJSON(`http://127.0.0.1:${PORT}/v1/messages`, {
    method: "POST", timeout: 30 * 60 * 1000,
    body: JSON.stringify({ model: ALIAS, max_tokens: nTok, messages: [{ role: "user", content: prompt }] }),
  });
  const secs = (Date.now() - t0) / 1000;
  if (!r.ok) die(`bench failed (${r.status}): ${JSON.stringify(r.json ?? r.error).slice(0, 500)}`);
  const usage = r.json?.usage ?? {};
  const out = usage.output_tokens ?? nTok;
  say();
  say(`  ${out} tokens in ${secs.toFixed(1)}s  =  ${(out / secs).toFixed(2)} tok/s`);
  say(`  prompt ${usage.input_tokens ?? "?"} tokens`);
  const text = (r.json?.content ?? []).map((b) => b.text ?? "").join("");
  say();
  say(text.slice(0, 600));
  say();
  return out / secs;
}

// Measures the two VRAM strategies end to end instead of trusting the model of llama.cpp's KV
// placement, and remembers the winner.
async function cmdTune() {
  const results = {};
  for (const strategy of ["layers", "ssm-pack"]) {
    say(`\n=== ${strategy} ===`);
    try {
      await down(PORT);
      await up({ ...planOpts(), strategy, restart: true, onProgress: (l) => l && process.stdout.write(`\r  ${l.slice(0, 100).padEnd(102)}`) });
      process.stdout.write("\r".padEnd(110) + "\r");
      const v = gpu();
      say(`  VRAM in use: ${fmtGB(v ? v.used : 0)}`);
      const tps = await cmdBench();
      results[strategy] = tps;
    } catch (e) {
      say(`  failed: ${e.message.split("\n")[0]}`);
      results[strategy] = 0;
    }
  }
  const best = Object.entries(results).sort((a, b) => b[1] - a[1])[0];
  say(`\n[claudelocal] fastest strategy: ${best[0]} at ${best[1].toFixed(2)} tok/s`);
  for (const [k, v] of Object.entries(results)) say(`   ${k.padEnd(10)} ${v.toFixed(2)} tok/s`);
  writeState({ strategy: best[0], tuned: results, tunedAt: new Date().toISOString() });
  say(`\nSaved. Future launches use --strategy ${best[0]} unless you pass another.`);
}

function cmdPull() {
  const script = path.join(paths.root, "bin", "fetch.sh");
  say("[claudelocal] downloading model assets (resumable - safe to interrupt and re-run)");
  const r = spawn("bash", [script], { stdio: "inherit" });
  r.on("exit", (c) => process.exit(c ?? 0));
}

function cmdInstall() {
  const script = path.join(paths.root, "bin", "fetch-runtime.sh");
  const r = spawn("bash", [script, opt("build", "b10819")], { stdio: "inherit" });
  r.on("exit", (c) => process.exit(c ?? 0));
}

function cmdHelp() {
  say(`
claudelocal - Claude Code on any local model on this machine

  claudelocal                    open Claude Code with no model loaded - /model loads one
  claudelocal -p "..."           any normal claude argument passes straight through

  claudelocal models             every model found - local and remote - and what /model offers
  claudelocal remotes            remote models, their keys, and (--check) whether they answer
  claudelocal --model <alias>    load a specific one
  claudelocal plan               show the memory plan and the context/speed trade-off table
  claudelocal up                 load the model only (no Claude Code)
  claudelocal down               stop the model and free all RAM and VRAM
  claudelocal status             server, proxy, guard, RAM and VRAM
  claudelocal bench              measure tokens/sec
  claudelocal tune               measure both VRAM strategies and keep the faster one
  claudelocal doctor             full environment dump
  claudelocal pull               download / resume the model weights
  claudelocal install            download the llama.cpp CUDA runtime

Options
  --model ALIAS      which local model to load (default: the last one used)
  --no-load          bring the proxy up but load no model; /model does the loading
  --min-ctx N        leave models with a smaller native window out of the picker, which
                     raises the session context every model has to share (default 16384)
  --rescan           re-read every GGUF header instead of trusting models.json
  --ctx N            cap the context length (default: each model's native maximum)
  --kv TYPE          KV cache type: auto (default), f16, q8_0, q5_1, q4_0
  --strategy S       VRAM strategy: layers (default) or ssm-pack
  --ram-margin GB    RAM to keep free for the rest of Windows (default 2)
  --port / --proxy-port
  --vision           also load the vision projector (costs ~0.9 GB of committed RAM)
  --force            start even if the planner says it will not fit
  --restart          reload even if a server is already up
`);
}

// --- dispatch ---------------------------------------------------------------------------------
const table = {
  plan: cmdPlan, up: cmdUp, down: cmdDown, stop: cmdDown, status: cmdStatus,
  doctor: cmdDoctor, bench: cmdBench, tune: cmdTune, pull: cmdPull,
  install: cmdInstall, help: cmdHelp,
  models: cmdModels, settings: cmdSettings, remotes: cmdRemotes,
};

const fn = table[cmd];
if (!fn) { cmdHelp(); process.exit(cmd === "help" ? 0 : 1); }
try { await fn(); } catch (e) { die(e.message); }
