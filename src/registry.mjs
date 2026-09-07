// Every local model on this machine, in one list.
//
// Three model managers live on a typical box and none of them agree on where a GGUF goes:
//
//   claudelocal/models/   flat files this repo downloaded itself
//   LM Studio             ~/.lmstudio/models/<publisher>/<repo>/<file>.gguf
//   Ollama                content-addressed blobs, so the file on disk is called
//                         sha256-3ab8... and the only place the NAME exists is a manifest
//
// All three are ordinary GGUF files that llama-server can open directly, so the job here is
// discovery and naming, not conversion. Ollama in particular needs its manifest read to find out
// that sha256-2bada8a74506 is "qwen2.5-coder:7b" - there is nothing in the blob itself that says so.
//
// Reading a GGUF header costs a couple of hundred KB, but doing it for twenty models on every
// launch is a visible pause, so the result is cached in models.json and re-read only when a file's
// size or mtime changes.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readGGUF, archInfo } from "./gguf.mjs";
import { paths, readJSON, GiB, MiB } from "./sys.mjs";

// What Claude Code sends before the conversation has started at all: system prompt, tool
// definitions, environment block. MEASURED on this machine at 33,096-33,522 tokens across several
// sessions - not the ~21k the prompt alone accounts for.
//
// This number is the whole reason a 32768-token model cannot run Claude Code. It is not that such a
// model runs out quickly; it is that the FIRST message already exceeds its window, and auto-compact
// cannot help because there is no conversation to compact yet. The failure is an immediate
//   request (33096 tokens) exceeds the available context size (32768 tokens)
export const CLAUDE_CODE_BASE_TOKENS = 33522;

// The smallest native window a model may have and still be offered in the picker: the baseline plus
// roughly as much again for actual work. Anything below this is listed by `claudelocal models` with
// the reason, and kept out of the picker - both because it cannot work, and because a model in the
// picker drags the whole session's context window down to its own size.
export const MIN_CTX = 65536;

// The alias reserved for "whatever is loaded right now". Claude Code fires small side-requests
// (conversation titles, file summaries) at the Haiku-class model while the main conversation runs
// on the chosen one; pointing those at a real alias would swap the model out from under the
// conversation several times a turn. They are pointed here instead, and the router serves them
// from the model already in memory.
export const AUTO_ALIAS = "claudelocal-auto";

const CACHE = path.join(paths.root, "models.json");
const norm = (p) => path.resolve(p);

// Quantization and shard tokens, stripped when turning a filename into an alias.
const QUANT = /[-_.](?:UD-)?(?:IQ|Q)\d(?:_[0-9KSMLXNR]+)*|[-_.](?:BF16|F16|F32|MXFP4|TQ\d_\d)|[-_.]imat(?:rix)?|[-_.]\d{5}-of-\d{5}/gi;

function walk(dir, out = [], depth = 0) {
  if (depth > 4) return out;
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out, depth + 1);
    else out.push(p);
  }
  return out;
}

// --- where the three managers keep their files -------------------------------------------------
function lmStudioHome() {
  // LM Studio can be pointed at another drive; when it has been, the real path is written into
  // this pointer file rather than into any config we could parse.
  const pointer = path.join(os.homedir(), ".lmstudio-home-pointer");
  try {
    const p = fs.readFileSync(pointer, "utf8").replace(/^﻿/, "").trim();
    if (p && fs.existsSync(p)) return p;
  } catch { /* fall through to the default */ }
  const dflt = path.join(os.homedir(), ".lmstudio");
  return fs.existsSync(dflt) ? dflt : null;
}

function ollamaRoot() {
  const env = process.env.OLLAMA_MODELS;
  if (env && fs.existsSync(env)) return env;
  const dflt = path.join(os.homedir(), ".ollama", "models");
  return fs.existsSync(dflt) ? dflt : null;
}

// Ollama's manifests are OCI image manifests: the model weights are the layer whose mediaType ends
// in `.image.model`, and its digest names a file in blobs/. The projector, template and license
// ride along as separate layers of the same manifest.
function ollamaCandidates() {
  const root = ollamaRoot();
  if (!root) return [];
  const manifests = path.join(root, "manifests");
  const out = [];
  for (const f of walk(manifests)) {
    let man;
    try { man = JSON.parse(fs.readFileSync(f, "utf8")); } catch { continue; }
    if (!Array.isArray(man.layers)) continue;
    const blobOf = (l) => path.join(root, "blobs", String(l.digest).replace(":", "-"));
    const model = man.layers.find((l) => /\.image\.model$/.test(l.mediaType ?? ""));
    if (!model || !fs.existsSync(blobOf(model))) continue;
    const proj = man.layers.find((l) => /\.image\.projector$/.test(l.mediaType ?? ""));
    // manifests/<registry>/<namespace>/<name>/<tag>  ->  "name:tag", the way ollama shows it
    const rel = path.relative(manifests, f).split(path.sep);
    const tag = rel.pop();
    const name = rel.pop();
    out.push({
      source: "ollama",
      ref: tag === "latest" ? name : `${name}:${tag}`,
      file: blobOf(model),
      mmproj: proj && fs.existsSync(blobOf(proj)) ? blobOf(proj) : null,
    });
  }
  return out;
}

function fileCandidates() {
  const roots = [];
  if (fs.existsSync(paths.models)) roots.push({ source: "claudelocal", dir: paths.models });
  const lm = lmStudioHome();
  if (lm) {
    for (const sub of ["models", path.join(".internal", "bundled-models")]) {
      const d = path.join(lm, sub);
      if (fs.existsSync(d)) roots.push({ source: "lmstudio", dir: d });
    }
  }
  for (const extra of (process.env.CLAUDELOCAL_MODEL_DIRS ?? "").split(";").filter(Boolean)) {
    if (fs.existsSync(extra)) roots.push({ source: "custom", dir: extra });
  }

  const out = [];
  for (const { source, dir } of roots) {
    const files = walk(dir).filter((f) => f.toLowerCase().endsWith(".gguf"));
    const projectors = files.filter((f) => /[\\/]mmproj[^\\/]*$/i.test(f));
    for (const f of files) {
      const base = path.basename(f);
      if (/^mmproj/i.test(base)) continue;
      // A speculative-decoding draft model is not something you would chat with on its own.
      if (/^mtp-/i.test(base)) continue;
      // Multi-part models: llama.cpp is given part one and finds the rest itself.
      const shard = base.match(/-(\d{5})-of-\d{5}\.gguf$/i);
      if (shard && shard[1] !== "00001") continue;
      out.push({
        source, ref: path.relative(dir, f).split(path.sep).join("/"), file: f,
        mmproj: projectors.find((p) => path.dirname(p) === path.dirname(f)) ?? null,
      });
    }
  }
  return out;
}

// --- naming ------------------------------------------------------------------------------------
// The alias is what the user picks in /model and what Claude Code then sends as the model id, so it
// has to be short, stable and typeable. None of the three managers gives us one directly: Ollama
// has a real name, LM Studio has a filename that may be anything from `Qwen3.5-9B-Q4_K_M.gguf` to
// `ggml-model-f16.gguf`, and a repacked community quant can carry sixty characters of provenance.
const GENERIC = /^(ggml-model|model|final|merged|output)$/;

function clean(s) {
  return String(s)
    .replace(/[-_]?GGUF$/i, "")
    .replace(/[-_]?instruct$/i, "")
    .replace(/[:_\s]+/g, "-")
    .replace(/[^A-Za-z0-9.\-]/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "")
    .toLowerCase();
}

function aliasFrom(c, info) {
  if (c.source === "ollama") return clean(c.ref) || "model";

  let s = clean(path.basename(c.file).replace(/\.gguf$/i, "").replace(QUANT, ""));
  // `ggml-model-f16.gguf` says nothing; the directory it sits in is the actual model name.
  if (GENERIC.test(s)) s = clean(path.basename(path.dirname(c.file)).replace(QUANT, ""));
  // A community requant can put its entire lineage in the filename. general.name is curated, so
  // once the alias is long enough to be unusable in a picker, prefer it - minus the `Publisher_`
  // prefix HuggingFace conversions prepend, which is never part of the model's name.
  if (s.length > 24 && info?.name) {
    const fromName = clean(String(info.name).replace(/^[A-Za-z0-9]+_/, ""));
    if (fromName && fromName.length < s.length) s = fromName;
  }
  return s || "model";
}

function quantOf(file) {
  const m = path.basename(file).match(/(?:UD-)?(IQ\d[0-9_KSMLXNR]*|Q\d[0-9_KSMLXNR]*|BF16|F16|F32|MXFP4|TQ\d_\d)/i);
  return m ? m[1].toUpperCase() : null;
}

const HUMAN = (b) => (b >= GiB ? `${(b / GiB).toFixed(1)} GB` : `${Math.round(b / MiB)} MB`);

// --- inspection ---------------------------------------------------------------------------------
// Everything that has to be read out of the file itself. Cached by size+mtime.
function probe(file) {
  const st = fs.statSync(file);
  const g = readGGUF(file);
  const a = archInfo(g);
  const tensorBytes = g.tensors.reduce((s, t) => s + t.bytes, 0);
  return {
    size: st.size, mtimeMs: st.mtimeMs,
    arch: a.arch,
    name: g.meta["general.name"] ?? a.arch,
    sizeLabel: g.meta["general.size_label"] ?? null,
    nCtxTrain: a.nCtxTrain,
    nBlocks: a.nLayer,
    hasMTP: a.hasMTP,
    chatTemplate: typeof g.meta["tokenizer.chat_template"] === "string",
    // A GGUF's header is written first, so a file that is still downloading parses perfectly and
    // then runs off the end of its own tensor data. Comparing the declared tensor bytes against
    // the file on disk is the only reliable way to catch that before llama-server does.
    complete: st.size >= g.headerBytes + tensorBytes * 0.999,
    declaredBytes: g.headerBytes + tensorBytes,
  };
}

function loadCache() { return readJSON(CACHE, {}) ?? {}; }
function saveCache(c) {
  try { fs.writeFileSync(CACHE, JSON.stringify(c, null, 2)); } catch { /* cache is an optimisation */ }
}

/**
 * Scan every source and return the usable models plus a reason for each one left out.
 * `refresh: true` re-reads every GGUF header instead of trusting models.json.
 */
export function discover({ refresh = false, minCtx = MIN_CTX } = {}) {
  const cache = refresh ? {} : loadCache();
  const nextCache = {};
  const models = [];
  const skipped = [];

  for (const c of [...fileCandidates(), ...ollamaCandidates()]) {
    const key = norm(c.file);
    let info = cache[key];
    let st;
    try { st = fs.statSync(c.file); } catch { continue; }
    if (!info || info.size !== st.size || info.mtimeMs !== st.mtimeMs) {
      try { info = probe(c.file); }
      catch (e) { skipped.push({ ...c, reason: `unreadable GGUF: ${e.message}` }); continue; }
    }
    nextCache[key] = info;

    // The vision projectors are already attached to their model as `mmproj`; on their own they are
    // not something you can talk to.
    if (info.arch === "clip" || info.arch === "mmproj") continue;
    if (!info.complete) {
      skipped.push({ ...c, ...info, reason: `incomplete download (${HUMAN(info.size)} of ${HUMAN(info.declaredBytes)})` });
      continue;
    }
    if (!info.nBlocks || !info.nCtxTrain) {
      skipped.push({ ...c, ...info, reason: `not a chat model (arch ${info.arch})` });
      continue;
    }
    if (info.nCtxTrain < minCtx) {
      const why = info.nCtxTrain < CLAUDE_CODE_BASE_TOKENS
        ? `native context ${info.nCtxTrain} cannot hold Claude Code's ${CLAUDE_CODE_BASE_TOKENS}-token opening prompt`
        : `native context ${info.nCtxTrain} is below the ${minCtx} floor (--min-ctx to lower it)`;
      skipped.push({ ...c, ...info, reason: why });
      continue;
    }
    if (!info.chatTemplate) {
      // --jinja is mandatory here, and without a template in the GGUF there is no turn boundary
      // and no tool calling - the model will answer, but Claude Code cannot drive it.
      skipped.push({ ...c, ...info, reason: "no chat template in the GGUF (tool calling would not work)" });
      continue;
    }

    models.push({
      alias: aliasFrom(c, info),
      source: c.source, ref: c.ref, file: c.file, mmproj: c.mmproj,
      name: info.name, arch: info.arch, sizeLabel: info.sizeLabel,
      quant: quantOf(c.file), bytes: info.size,
      nCtxTrain: info.nCtxTrain, nBlocks: info.nBlocks, hasMTP: info.hasMTP,
    });
  }

  // Two managers holding the same model, or two quants of one model, collide on a single alias -
  // and an alias that resolves to two different files is worse than an ugly one. Bigger first, so
  // when the same model exists twice the better copy keeps the clean name.
  models.sort((a, b) => b.bytes - a.bytes);
  const seen = new Map();
  for (const m of models) {
    if (!seen.has(m.alias)) { seen.set(m.alias, m); continue; }
    const base = m.alias;
    // Say what actually distinguishes this copy - its quantization, or which manager holds it -
    // before falling back to a meaningless counter.
    const tried = [m.quant?.toLowerCase(), m.source].filter(Boolean).map((s) => `${base}-${clean(s)}`);
    let pick = tried.find((t) => !seen.has(t));
    if (!pick) { let i = 2; while (seen.has(`${base}-${i}`)) i++; pick = `${base}-${i}`; }
    m.alias = pick;
    seen.set(pick, m);
  }

  saveCache(nextCache);
  return { models, skipped };
}

export function resolve(alias, models) {
  if (!alias) return null;
  const want = String(alias).toLowerCase();
  if (want === AUTO_ALIAS) return null;
  return models.find((m) => m.alias === want)
    ?? models.find((m) => m.alias.replace(/[.\-]/g, "") === want.replace(/[.\-]/g, ""))
    ?? models.find((m) => m.alias.includes(want))
    ?? null;
}

// One row per model for Claude Code's /model picker.
//
// The auto alias gets a row of its own, first. It is not vanity: Claude Code refuses to reason
// about a model that is in neither its catalog nor the picker, and warns that it is assuming a
// 200k window - which for a model pinned to 32768 is the difference between compacting on time and
// handing llama-server a prompt it cannot hold. A row with behavesAs is what registers it.
export function pickerRows(models, { ctxOf, includeAuto = true } = {}) {
  const rows = includeAuto ? [{
    model: AUTO_ALIAS,
    label: "Auto (whatever is loaded)",
    description: "Serve from the model already in memory - never triggers a swap",
    behavesAs: "claude-sonnet-4-5",
  }] : [];
  return rows.concat(models.map((m) => {
    const ctx = ctxOf?.(m) ?? Math.min(m.nCtxTrain, 262144);
    const bits = [
      m.sizeLabel ?? null,
      m.quant,
      `${HUMAN(m.bytes)} on disk`,
      `${Math.round(ctx / 1024)}k ctx`,
      m.source,
    ].filter(Boolean);
    return {
      model: m.alias,
      // HuggingFace conversions prefix general.name with the uploading org ("Qwen_Qwen3.5 35B
      // A3B"), which is noise in a picker that is already only local models.
      label: String(m.name).replace(/^[A-Za-z0-9]+_/, "").replace(/_/g, " ").trim() || m.alias,
      description: bits.join(" · "),
      // Claude Code will not offer a row for a model it has never heard of unless it is told whose
      // client-side handling to borrow - prompt profile, capability and effort defaults. Sonnet 4.5
      // is the closest fit for a local model: a plain tool-calling profile with no extended
      // thinking and no effort levels to negotiate.
      behavesAs: "claude-sonnet-4-5",
    };
  }));
}

export const human = HUMAN;
