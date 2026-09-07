#!/usr/bin/env node
// Claude Code  ->  http://127.0.0.1:<port>  ->  llama-server /v1/messages
//
// llama-server speaks the Anthropic Messages API natively, so this is a normalizing shim rather
// than a translator. Claude Code sends a handful of things a llama.cpp build does not know about,
// and a jinja chat template has stricter shape requirements than Anthropic's API does:
//
//   - top-level `thinking`, `context_management`, `output_config`, `betas`, `metadata`
//   - `cache_control` markers on content blocks (an Anthropic prompt-caching hint)
//   - `role: "system"` messages sitting inside `messages`
//   - consecutive same-role turns, which most chat templates render as a malformed conversation
//
// Everything else, streaming included, is passed straight through.
//
// It is also the ROUTER. Claude Code sends the model it wants on every request, so `/model` picking
// a different local model shows up here as a different `model` field - and since only one model can
// be resident on a 6 GB card at a time, that means unloading one and loading the other. The switch
// happens inside the request that asked for it, with the connection held open and pinged, because
// there is nowhere else to put a two-minute wait.
//
//   node src/proxy.mjs [--port 20181] [--target 20180]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { discover, resolve as resolveModel, AUTO_ALIAS } from "./registry.mjs";
import { ensureModel } from "./serve.mjs";
import { readState } from "./sys.mjs";

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : d; };
const PORT = parseInt(opt("port", process.env.CLAUDELOCAL_PROXY_PORT || "20181"), 10);
const TARGET = parseInt(opt("target", process.env.CLAUDELOCAL_PORT || "20180"), 10);
const MAX_OUTPUT = parseInt(opt("max-output", process.env.CLAUDELOCAL_MAX_OUTPUT || "8192"), 10);
// Claude Code's auto-compact window is fixed for the life of a session, so every model switched to
// mid-session has to be loaded at a context no smaller than the one that number was computed
// against. The launcher works it out across the whole picker and passes it here.
const CTX_CAP = parseInt(opt("ctx", process.env.CLAUDELOCAL_CTX || "0"), 10);
const LOG = path.join(path.dirname(path.dirname(fileURLToPath(import.meta.url))), "logs", "proxy.log");

// Bump when the rewriting rules change: `claudelocal up` replaces a proxy reporting an older one.
export const PROXY_VERSION = 6;

// The registry is read from models.json, so this is cheap; it is re-read when a request names a
// model we do not recognise, which is how a model downloaded mid-session becomes switchable
// without restarting anything.
let registry = discover();
const rescan = () => { registry = discover({ refresh: true }); return registry; };
// Non-null while a model swap is in flight, so `claudelocal status` can say what is happening
// instead of just reporting the server as down for two minutes.
let switchNote = null;

// How often to emit a keep-alive ping on a streaming response, and how long to wait for upstream
// headers before committing to a stream of our own. Claude Code gives up somewhere around a
// minute, so both sit comfortably inside that.
const PING_MS = 10000;        // never let the stream go quiet for longer than this
const PING_TICK_MS = 2000;    // how often the idle check runs
const PRE_HEADER_MS = 15000;

const DROP_TOP_LEVEL = ["thinking", "context_management", "output_config", "betas", "mcp_servers"];

function log(msg) {
  try {
    fs.mkdirSync(path.dirname(LOG), { recursive: true });
    fs.appendFileSync(LOG, `[${new Date().toISOString()}] ${msg}\n`);
  } catch { /* logging must never break a request */ }
}

const asBlocks = (c) =>
  typeof c === "string" ? [{ type: "text", text: c }] : Array.isArray(c) ? c : c ? [c] : [];

// `cache_control` is an Anthropic prompt-caching hint. llama.cpp caches by prefix on its own and
// rejects or ignores the field depending on build, so it is stripped rather than forwarded.
function stripBlock(b) {
  if (!b || typeof b !== "object") return b;
  const { cache_control, ...rest } = b;
  return rest;
}

function normalize(body) {
  for (const f of DROP_TOP_LEVEL) delete body[f];

  if (Array.isArray(body.system)) {
    body.system = body.system.map(stripBlock);
  }

  if (Array.isArray(body.messages)) {
    const out = [];
    for (const m of body.messages) {
      let role = m.role;
      let content = asBlocks(m.content).map(stripBlock);
      if (role !== "user" && role !== "assistant") {
        // Mid-conversation system/developer notes. Keep them where they are, but deliver them as a
        // user-side reminder so the chat template has a role it understands.
        role = "user";
        content = content.map((b) =>
          b && b.type === "text" ? { ...b, text: `<system-reminder>\n${b.text}\n</system-reminder>` } : b);
      }
      const prev = out[out.length - 1];
      if (prev && prev.role === role) prev.content = [...prev.content, ...content];
      else out.push({ role, content });
    }
    body.messages = out;
  }

  // Claude Code asks for very large max_tokens. At a few tokens per second an unbounded generation
  // is an hour of silence, so it is clamped to something a local model can actually finish.
  if (typeof body.max_tokens === "number" && body.max_tokens > MAX_OUTPUT) body.max_tokens = MAX_OUTPUT;
  return body;
}

// llama-server's context-overflow error is accurate and completely unactionable from inside Claude
// Code: it names two numbers and tells you to "try increasing it", with no hint that the thing to
// increase lives in a launcher flag, or that the window is small because of some OTHER model in the
// picker rather than the one you are talking to. Rewrite it into the fix.
function explainContextOverflow(text) {
  let err;
  try { err = JSON.parse(text); } catch { return text; }
  const e = err?.error ?? {};
  if (e.type !== "exceed_context_size_error") return text;
  const nCtx = e.n_ctx ?? 0;
  const need = e.n_prompt_tokens ?? 0;
  const st = readState();
  const model = registry.models.find((m) => m.file === st.model);
  const native = model?.nCtxTrain ?? 0;

  const lines = [`claudelocal: this turn needs ${need} tokens but ${st.alias ?? "the model"} is loaded with a ${nCtx}-token window.`];
  if (native > nCtx) {
    // The common case, and the one worth spelling out: the loaded model is perfectly capable and
    // has been held down to the smallest window in the picker.
    lines.push(`${st.alias} natively supports ${native}, but every model in the /model picker shares one window`);
    lines.push(`and the smallest one sets it. Restart with:  claudelocal --cl-min-ctx ${nCtx + 1}`);
    lines.push("('claudelocal --cl-models' names the model that is pinning it.)");
  } else {
    lines.push(`${nCtx} is this model's own maximum, so it cannot hold this conversation. Switch to a`);
    lines.push("larger-context model with /model, or start a new conversation.");
  }
  e.message = lines.join(" ");
  return JSON.stringify(err);
}

const server = http.createServer((req, res) => {
  const url = req.url ?? "/";

  if (url === "/claudelocal/health") {
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({ ok: true, pid: process.pid, target: TARGET, version: PROXY_VERSION, ctx: CTX_CAP }));
  }
  if (url === "/claudelocal/models") {
    const st = readState();
    res.writeHead(200, { "content-type": "application/json" });
    return res.end(JSON.stringify({
      current: st.alias ?? null,
      switching: switchNote,
      models: registry.models.map((m) => ({ alias: m.alias, name: m.name, source: m.source, bytes: m.bytes, ctx: m.nCtxTrain })),
    }, null, 2));
  }
  if (url === "/claudelocal/quit") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
    return setTimeout(() => process.exit(0), 50);
  }

  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", async () => {
    let bodyBuf = Buffer.concat(chunks);

    let isStream = false;
    let parsed = null;
    if (req.method === "POST" && /\/v1\/messages/.test(url) && bodyBuf.length) {
      try {
        parsed = normalize(JSON.parse(bodyBuf.toString("utf8")));
        isStream = parsed.stream === true;
      } catch (e) { log(`normalize failed, forwarding as-is: ${e.message}`); }
    }
    // Anthropic's own API emits `ping` events while a response is still being produced, and Claude
    // Code's client treats a stream that goes quiet for about a minute as a dead connection: it
    // aborts and retries with a growing backoff ("will retry in 4m 32s - check your network").
    //
    // llama-server emits NOTHING while it processes the prompt, and a Claude Code prompt is ~21k
    // tokens - five minutes of silence on this machine. So every attempt was being killed at ~74 s
    // and restarted, which is a loop that can never finish a turn. Synthesizing the pings the real
    // API would have sent is what makes a slow local model usable at all.
    //
    // The ping is driven by an IDLE timer rather than fired until the first upstream byte: llama.cpp
    // sends a single bare `:` comment early on and then goes quiet again for as long as the prompt
    // takes, so "stop as soon as anything arrives" still leaves a minutes-long silence. Tracking the
    // last write instead guarantees the client never sees more than PING_MS of nothing, wherever the
    // gap falls. During generation tokens arrive far faster than that, so no pings are sent at all.
    let pinger = null;
    let lastWrite = Date.now();
    const touch = () => { lastWrite = Date.now(); };
    const stopPings = () => { if (pinger) { clearInterval(pinger); pinger = null; } };
    const startPings = () => {
      if (pinger || !isStream) return;
      pinger = setInterval(() => {
        if (res.writableEnded || res.destroyed) return stopPings();
        if (Date.now() - lastWrite < PING_MS) return;
        res.write("event: ping\ndata: {\"type\":\"ping\"}\n\n");
        touch();
      }, PING_TICK_MS);
    };

    const beginStream = () => {
      if (res.headersSent) return;
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      startPings();
    };

    const failStream = (message) => {
      stopPings();
      if (res.headersSent) {
        res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message } })}\n\n`);
        return res.end();
      }
      res.writeHead(502, { "content-type": "application/json" });
      res.end(JSON.stringify({ type: "error", error: { type: "api_error", message } }));
    };

    // --- routing ---------------------------------------------------------------------------------
    // `model` is the only thing that says which local model this turn wants. Anything we do not
    // recognise - the auto alias Claude Code's side-requests use, or a name from another provider -
    // is served by whatever is already loaded, which is what stops a conversation-title request
    // from evicting the 15 GB model half-way through a turn.
    if (parsed?.model) {
      let entry = resolveModel(parsed.model, registry.models);
      if (!entry && parsed.model !== AUTO_ALIAS) {
        // A model pulled since this proxy started is worth one rescan before giving up on it.
        entry = resolveModel(parsed.model, rescan().models);
      }
      const st = readState();
      if (entry && entry.file !== st.model) {
        // Loading is the slow part of this whole system - up to two minutes for a 20 GB quant - so
        // the stream is opened and pinged FIRST. Without that the client gives up at about a minute
        // and retries, which starts a second load against memory the first one is still taking.
        beginStream();
        switchNote = `${st.alias ?? "nothing"} -> ${entry.alias}`;
        log(`switching model: ${switchNote}`);
        const t0 = Date.now();
        try {
          await ensureModel(entry, { port: TARGET, ctx: CTX_CAP || undefined });
          log(`switched to ${entry.alias} in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
        } catch (e) {
          log(`switch to ${entry.alias} failed: ${e.message}`);
          return failStream(`claudelocal: could not load ${entry.alias} - ${e.message.split("\n")[0]}`);
        } finally { switchNote = null; }
        touch();
      }
    }

    // llama-server checks the `model` field against the `--alias` it was started with, so the
    // request has to name what is loaded now - which may be a model that was not even resident
    // when this request arrived.
    if (parsed) {
      parsed.model = readState().alias ?? parsed.model;
      bodyBuf = Buffer.from(JSON.stringify(parsed), "utf8");
    }

    const headers = { ...req.headers, host: `127.0.0.1:${TARGET}`, "content-length": String(bodyBuf.length) };
    delete headers["accept-encoding"];   // keep the SSE stream plain
    delete headers["x-api-key"];         // llama-server runs without auth on loopback
    delete headers["authorization"];

    // llama-server may not send response headers until the prompt is done either, so if nothing has
    // come back at all by the time the client would start worrying, commit to a stream ourselves.
    // The wait is long enough that a chat-template rejection - which comes back in well under a
    // second - still surfaces as a real HTTP error instead of being masked by a 200.
    const headTimer = isStream ? setTimeout(beginStream, PRE_HEADER_MS) : null;

    const upstream = http.request(
      { host: "127.0.0.1", port: TARGET, method: req.method, path: url, headers },
      (r) => {
        clearTimeout(headTimer);

        // Capture the body of an upstream rejection: without it a 400 from the chat template is
        // completely opaque from inside Claude Code.
        if ((r.statusCode ?? 0) >= 400) {
          const errChunks = [];
          r.on("data", (c) => errChunks.push(c));
          r.on("end", () => {
            let text = Buffer.concat(errChunks).toString("utf8");
            log(`upstream ${r.statusCode} on ${url}: ${text.slice(0, 2000)}`);
            text = explainContextOverflow(text);
            // If the ping timer already committed us to a 200 stream there is no status code left
            // to send, so deliver the failure the only way an open SSE stream can.
            if (res.headersSent) {
              stopPings();
              res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message: `claudelocal: llama-server ${r.statusCode}: ${text.slice(0, 500)}` } })}\n\n`);
              return res.end();
            }
            // The rewritten message is longer than what llama-server sent, so its content-length
            // header would truncate the body mid-JSON and the client would report a parse error
            // instead of the explanation.
            const outHeaders = { ...r.headers };
            delete outHeaders["content-length"];
            outHeaders["content-length"] = String(Buffer.byteLength(text, "utf8"));
            res.writeHead(r.statusCode ?? 502, outHeaders);
            res.end(text);
          });
          return;
        }

        if (!res.headersSent) res.writeHead(r.statusCode ?? 502, r.headers);
        startPings();
        r.on("data", (c) => { touch(); if (!res.write(c)) r.pause(); });
        res.on("drain", () => r.resume());
        r.on("end", () => { stopPings(); res.end(); });
        r.on("error", () => { stopPings(); res.end(); });
      });

    upstream.on("error", (e) => {
      clearTimeout(headTimer);
      stopPings();
      log(`upstream unreachable: ${e.message}`);
      if (res.headersSent) return res.end();
      res.writeHead(502, { "content-type": "application/json" });
      res.end(JSON.stringify({
        type: "error",
        error: { type: "api_error", message: `claudelocal: llama-server unreachable on :${TARGET} (${e.message}) - try 'claudelocal status'` },
      }));
    });

    // Abort upstream only when the CLIENT goes away mid-response. `req.on("close")` would be wrong:
    // Node fires it as soon as the request body has been read, which would kill the socket before
    // llama-server ever replied.
    res.on("close", () => {
      clearTimeout(headTimer);
      stopPings();
      if (!res.writableFinished && !upstream.destroyed) upstream.destroy();
    });
    upstream.end(bodyBuf);
  });
});

// Generation on a CPU-offloaded 27B model is slow by design - nothing here may time out.
server.keepAliveTimeout = 0;
server.headersTimeout = 0;
server.requestTimeout = 0;

const self = process.argv[1] && path.resolve(process.argv[1]);
if (self && self === path.resolve(fileURLToPath(import.meta.url))) {
  server.listen(PORT, "127.0.0.1", () =>
    console.log(`[claudelocal-proxy] http://127.0.0.1:${PORT} -> llama-server :${TARGET}`));
}
