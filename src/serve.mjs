// Starts llama-server under the guard and waits for it to become healthy.
//
// Nothing here talks to llama-server directly - guard.ps1 owns the process so that the Job Object
// commit cap and the memory watchdog are in force from before the first byte is mapped.
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import path from "node:path";

// Deliberately NOT execFileSync. The router loads models from inside a live HTTP request, and the
// only thing keeping Claude Code from abandoning that request is a ping written every few seconds.
// A synchronous PowerShell call blocks Node's event loop for as long as it runs - up to two minutes
// here - so the timers never fire, the client sees two minutes of silence and retries, and the
// retry starts a second load against memory the first one is still taking.
const run = promisify(execFile);
import { paths, httpJSON, readJSON, readState, writeState, GiB, MiB, fmtGB, ram, gpu } from "./sys.mjs";
import { plan, serverArgs, explain } from "./plan.mjs";

export const DEFAULT_PORT = 20180;
export const DEFAULT_PROXY_PORT = 20181;

export const serverExe = () => path.join(paths.runtime, "llama-server.exe");

export function findModel(pattern) {
  if (pattern && fs.existsSync(pattern)) return pattern;
  const files = fs.existsSync(paths.models) ? fs.readdirSync(paths.models) : [];
  const ggufs = files.filter((f) => f.endsWith(".gguf") && !/^mmproj|^mtp-/i.test(f));
  if (!ggufs.length) return null;
  const pick = pattern ? ggufs.find((f) => f.toLowerCase().includes(pattern.toLowerCase())) : ggufs[0];
  return pick ? path.join(paths.models, pick) : null;
}

export function findMmproj() {
  const files = fs.existsSync(paths.models) ? fs.readdirSync(paths.models) : [];
  const f = files.find((x) => /^mmproj.*\.gguf$/i.test(x));
  return f ? path.join(paths.models, f) : null;
}

export async function health(port) {
  const r = await httpJSON(`http://127.0.0.1:${port}/health`, { timeout: 2000 });
  return r.ok ? (r.json ?? { status: "ok" }) : null;
}

export function guardStatus() {
  return readJSON(path.join(paths.logs, "guard.json"));
}

export function tailLog(file, n = 30) {
  try {
    const txt = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "");
    return txt.split(/\r?\n/).filter(Boolean).slice(-n).join("\n");
  } catch { return "(no log)"; }
}

// The Job Object cap covers committed memory only - the mmap'd weights are file-backed and are not
// charged to commit. So the cap is sized from the KV cache plus llama.cpp's host buffers, with
// generous slack: it is a runaway brake, not a tuning knob, and clipping a legitimate allocation
// would just crash the load.
function commitCapMB(p, mmproj) {
  // The KV cache is allocated as one contiguous CPU buffer, so the cap has to clear it outright -
  // a cap even slightly under it turns into "failed to allocate buffer for kv cache" at load time.
  // The vision projector is committed too when it is kept off the GPU.
  const mmprojBytes = mmproj ? (fs.statSync(mmproj).size ?? 0) : 0;
  const need = p.est.committed + mmprojBytes;
  // The 50% slack this used to carry was really covering an estimate that modelled neither the
  // full-size MTP draft cache, the -ot-pinned tensors, nor the n_vocab-sized logits buffer - it
  // came out ~4.5 GB under what the guard then measured, and the multiplier absorbed the error.
  // Now that those are all budgeted the estimate lands within a few hundred MB, so the cap can go
  // back to being what it is meant to be: a runaway brake just above the real footprint, not a
  // number so loose that a genuine leak fits underneath it.
  const cap = Math.max(3 * GiB, need * 1.15 + 1 * GiB);
  const room = Math.max(3 * GiB, ram().free - 1.5 * GiB);
  return Math.round(Math.min(cap, room) / MiB);
}

// VRAM accounting can only ever be an estimate: the desktop's own usage moves while we load, and
// llama.cpp's graph allocator sizes some buffers only once it sees the real shapes. So rather than
// guessing conservatively enough to always be right - which would waste a good half gigabyte of a
// 6 GB card - the reserve starts tight and backs off 512 MB per attempt if CUDA says it is short.
const OOM = /out of memory|failed to allocate|cudaMalloc|ggml_backend_cuda_buffer_type_alloc|CUDA error/i;


// Wait until free VRAM and free RAM stop moving, so the planner measures a settled machine.
async function settle({ tries = 12, quietMs = 700 } = {}) {
  let prev = null, stable = 0;
  for (let i = 0; i < tries; i++) {
    const v = gpu();
    const now = `${v ? Math.round(v.free / MiB / 64) : 0}:${Math.round(ram().free / MiB / 64)}`;
    if (now === prev) { if (++stable >= 2) return; } else { stable = 0; }
    prev = now;
    await new Promise((r) => setTimeout(r, quietMs));
  }
}

export async function up(opts = {}) {
  const attempts = opts.attempts ?? 3;
  let extraReserve = 0;
  for (let i = 0; ; i++) {
    try {
      return await start({ ...opts, vramReserve: (opts.vramReserve ?? 768 * MiB) + extraReserve });
    } catch (e) {
      const last = i >= attempts - 1;
      if (last || !OOM.test(e.message)) throw e;
      extraReserve += 512 * MiB;
      opts.onProgress?.(`VRAM was short - retrying with ${fmtGB(extraReserve)} more held back`, null);
      await down(opts.port ?? DEFAULT_PORT);
    }
  }
}

async function start(opts = {}) {
  const port = opts.port ?? DEFAULT_PORT;
  const exe = serverExe();
  if (!fs.existsSync(exe)) throw new Error(`llama-server not found at ${exe} - run: claudelocal install`);

  // A model can arrive three ways: an explicit path from the registry (how the router switches
  // models), a name fragment, or nothing at all - in which case the single file this repo
  // downloaded itself is meant.
  const modelPath = opts.modelPath ?? findModel(opts.model);
  if (!modelPath) throw new Error(`no .gguf found in ${paths.models} - run: claudelocal pull`);
  const alias = opts.alias ?? "local";

  const existing = await health(port);
  if (existing) {
    const st = readState();
    const sameModel = st.model === modelPath;
    if (!opts.restart && sameModel && (!opts.ctx || st.ctx === opts.ctx) && (!opts.strategy || st.strategy === opts.strategy)) {
      return { already: true, port, state: st };
    }
    await down(port);
  }

  // Freeing 5 GB of VRAM and 16 GB of page cache is not instant: for a second or two after the old
  // llama-server dies, nvidia-smi still reports its allocations and Windows has not released the
  // mapped file pages. Planning against those stale numbers silently produces a worse layout - it
  // is what put only 6 layers on the card and picked a KV cache twice the size it needed to be.
  await settle();

  const p = plan(modelPath, opts);
  if (!p.ok && !opts.force) {
    throw new Error(`${explain(p)}\n\nRefusing to start - pass --force to override, or lower --ctx.`);
  }

  // The registry pairs each model with the projector sitting beside it, so vision follows the
  // model that is actually loaded rather than whatever happens to be in claudelocal/models.
  const mmproj = opts.vision === false ? null
    : (opts.mmproj !== undefined ? opts.mmproj : findMmproj());
  const args = serverArgs(p, { port, mmproj, alias });

  fs.mkdirSync(paths.logs, { recursive: true });
  const argsFile = path.join(paths.logs, "server-args.json");
  const logFile = path.join(paths.logs, "server.log");
  const statusFile = path.join(paths.logs, "guard.json");
  fs.writeFileSync(argsFile, JSON.stringify(args, null, 2));
  try { fs.unlinkSync(statusFile); } catch { /* first run */ }

  const cap = commitCapMB(p, mmproj);
  const guardArgs = [
    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", path.join(paths.root, "src", "guard.ps1"),
    "-Exe", exe,
    "-ArgsFile", argsFile,
    "-LogFile", logFile,
    "-StatusFile", statusFile,
    "-CommitCapMB", String(cap),
    "-SoftFloorMB", String(opts.softFloorMB ?? 2048),
    "-HardFloorMB", String(opts.hardFloorMB ?? 1024),
  ];

  // Detaching via Node's `detached: true` does not work on this machine: PowerShell is created but
  // never runs its command line, exits 0 immediately and writes nothing (reproduced against a bare
  // `-Command Write-Output`; the same spawn without `detached` works). A non-detached spawn that
  // calls Start-Process gets the same independence without tripping over it, and Start-Process
  // also gives us the guard's own stdout/stderr on disk - so a PowerShell-level failure before
  // guard.ps1 can open its log is visible instead of silent.
  const spawnLog = path.join(paths.logs, "guard-spawn.log");
  const spawnErr = path.join(paths.logs, "guard-spawn.err.log");
  const launchErr = path.join(paths.logs, "guard-launch.err.log");
  try { fs.unlinkSync(launchErr); } catch { /* no previous failure */ }
  const psList = guardArgs.map((a) => `'${String(a).replace(/'/g, "''")}'`).join(",");
  const launcher =
    `$ErrorActionPreference='Stop';` +
    `try { Start-Process -FilePath 'powershell' -ArgumentList @(${psList}) ` +
    `-WindowStyle Hidden -RedirectStandardOutput '${spawnLog}' -RedirectStandardError '${spawnErr}' } ` +
    `catch { $_ | Out-File -FilePath '${launchErr}' -Encoding ascii; exit 1 }`;

  // stdio MUST be "ignore" here, and this is not a detail. With pipes, Node resolves the call when
  // the pipes close rather than when PowerShell exits - and llama-server, three processes down,
  // inherits those same handles and holds them open for its entire life. So the launcher call sat
  // blocked until its own timeout on EVERY start: measured at 120 s of pure waiting for a model
  // that was already answering after 12. (The wait was invisible because the health poll that
  // follows then succeeded on its first try, so the time looked like load time.)
  //
  // `detached: true` is still not usable on this machine - PowerShell is created, never runs its
  // command line, and exits 0 having written nothing - so this stays a normal child that happens
  // to launch a grandchild and return.
  await new Promise((resolve, reject) => {
    const child = spawn("powershell", ["-NoLogo", "-NoProfile", "-Command", launcher],
      { stdio: "ignore", windowsHide: true });
    const timer = setTimeout(() => { child.kill(); reject(new Error("timed out starting the guard process")); }, 60000);
    child.on("error", (e) => { clearTimeout(timer); reject(new Error(`could not start the guard process: ${e.message}`)); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) return resolve();
      reject(new Error(`could not start the guard process (exit ${code})\n${tailLog(launchErr, 10)}`));
    });
  });

  writeState({
    port, model: modelPath, modelName: p.model.name, alias, source: opts.source ?? null,
    vision: !!mmproj,
    ctx: p.ctx, kvType: p.kvType, strategy: p.strategy,
    ngl: p.ngl, nGpuLayers: p.nGpuLayers, mtp: p.mtp,
    commitCapMB: cap, startedAt: new Date().toISOString(),
    est: p.est,
  });

  // A 15 GB model has to be read off disk and part of it pushed over PCIe, so first load is slow;
  // it is much faster afterwards while the file is still in the page cache.
  const deadline = Date.now() + (opts.timeoutMs ?? 15 * 60 * 1000);
  let lastNote = 0;
  while (Date.now() < deadline) {
    const h = await health(port);
    if (h && h.status !== "loading model") return { already: false, port, plan: p, health: h };
    const g = guardStatus();
    if (g && g.running === false) {
      throw new Error(`llama-server exited (code ${g.exitCode}, ${g.reason}).\n\n${tailLog(logFile, 40)}`);
    }
    if (opts.onProgress && Date.now() - lastNote > 5000) {
      lastNote = Date.now();
      opts.onProgress(tailLog(logFile, 1), g);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`timed out waiting for llama-server on :${port}\n\n${tailLog(logFile, 40)}`);
}

// --- switching models ---------------------------------------------------------------------------
// One machine, one GPU, 32 GB of RAM: exactly one model can be resident, so switching means
// unloading the old one and loading the new one from scratch. That takes anywhere from two seconds
// for a 0.5B to two minutes for a 20 GB quant, which is why the router holds the HTTP request open
// and keeps it alive with pings rather than returning an error and asking the user to try again.
//
// Everything is serialised through this one promise. Claude Code fires several requests per turn,
// and two concurrent loads would race for the same VRAM and both lose - the second would plan
// against memory the first had already taken.
let switching = Promise.resolve();

export function currentModel() {
  const st = readState();
  return st.model ? { file: st.model, alias: st.alias, name: st.modelName, ctx: st.ctx } : null;
}

/**
 * Make `entry` (a registry row) the loaded model, if it is not already. Returns
 * { switched, from, to }. Serialised: concurrent callers queue behind one another.
 */
export function ensureModel(entry, opts = {}) {
  const run = async () => {
    const port = opts.port ?? DEFAULT_PORT;
    const st = readState();
    const live = await health(port);
    if (live && st.model === entry.file && st.alias === entry.alias) {
      return { switched: false, from: entry.alias, to: entry.alias };
    }
    const from = st.alias ?? null;
    opts.onProgress?.(`loading ${entry.alias}`);
    await up({
      ...opts,
      modelPath: entry.file,
      alias: entry.alias,
      // Vision is opt-in, the same as it is on the command line. The router used to default it ON
      // because it only checked for an explicit `false`, so switching to a model with a projector
      // beside it silently spent ~0.9 GB of committed RAM on a capability the session was not
      // using - RAM that the KV cache and the weights both wanted.
      mmproj: opts.vision ? (entry.mmproj ?? null) : null,
      source: entry.source,
      // Each model has its own native window, so a context asked for on the command line is a
      // ceiling here rather than a target - nothing may exceed what the model was trained for.
      ctx: opts.ctx ? Math.min(opts.ctx, entry.nCtxTrain) : entry.nCtxTrain,
      restart: true,
    });
    return { switched: true, from, to: entry.alias };
  };
  // Chain onto the previous switch whether it succeeded or failed, so one bad model cannot wedge
  // the queue for the rest of the session.
  const next = switching.then(run, run);
  switching = next.then(() => {}, () => {});
  return next;
}

export async function down(port = DEFAULT_PORT) {
  // Killing the guard closes the Job Object handle, and KILL_ON_JOB_CLOSE takes llama-server with
  // it - so there is no way to leave 15 GB of mapped weights behind.
  const script = `
$ErrorActionPreference='SilentlyContinue'
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
  Where-Object { $_.CommandLine -like '*guard.ps1*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
`;
  try { await run("powershell", ["-NoLogo", "-NoProfile", "-Command", script], { timeout: 20000, windowsHide: true }); } catch { /* nothing to stop */ }
  writeState({ stoppedAt: new Date().toISOString() });
  for (let i = 0; i < 10; i++) {
    if (!(await health(port))) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}
