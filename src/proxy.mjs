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
// A remote configured with `api: "openai"` is the one case that is a real translation: openai.mjs
// rewrites the request, the response and the SSE stream, because that provider has no Messages API
// to pass anything through to. Such a remote may also name FALLBACKS - sibling entries under other
// keys - which are tried in order on 429, 5xx or a refused connection, and the whole chain is
// walked again a few times with backoff before the turn is allowed to fail.
//
// It is also the ROUTER. Claude Code sends the model it wants on every request, so `/model` picking
// a different local model shows up here as a different `model` field - and since only one model can
// be resident on a 6 GB card at a time, that means unloading one and loading the other. The switch
// happens inside the request that asked for it, with the connection held open and pinged, because
// there is nowhere else to put a two-minute wait.
//
//   node src/proxy.mjs [--port 20181] [--target 20180]
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { discover, resolve as resolveModel, AUTO_ALIAS } from "./registry.mjs";
import { isRemote, remoteKey } from "./remotes.mjs";
import { toOpenAIRequest, toAnthropicResponse, streamTranslator, estimateTokens } from "./openai.mjs";
import { ensureModel } from "./serve.mjs";
import { readState, writeState } from "./sys.mjs";

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
export const PROXY_VERSION = 10;

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

// A vLLM worker that has crashed answers 500 forever: the API server survives, so the socket still
// connects and every request comes back `EngineDeadError` until the process is restarted. Retrying
// that is worse than useless - the backoff sweep turns an instant, explainable failure into a
// minute of apparent hanging. Kaggle's TPU kernel hits it whenever a bug in vLLM's speculative
// decoding path is triggered by concurrent requests, which Claude Code produces on every turn.
function isDeadUpstream(text) {
  return /EngineDeadError|EngineCore encountered|engine core .*(died|dead)/i.test(text ?? "");
}

function explainDeadUpstream(text, upstreamName) {
  const lines = [
    `claudelocal: ${upstreamName} accepted the connection but its inference engine has crashed -`,
    "every request will fail until the server process is restarted, so this turn was not retried.",
    "If this is the Kaggle TPU endpoint: the kernel is gone. Relaunch it with",
    "`python launch.py serve --mtp 0` in ~/Projects/kaggle-tpu-lab, then put the new baseUrl and",
    "apiKey from its banner into remotes.json. Meanwhile /model can switch to any other model.",
  ];
  let err;
  try { err = JSON.parse(text); } catch { return lines.join(" "); }
  if (!err || typeof err !== "object") return lines.join(" ");
  err.error = { ...(err.error ?? {}), type: "api_error", message: lines.join(" ") };
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

    // Parsed but NOT yet normalized: the rewriting below exists to satisfy a jinja chat template
    // in llama.cpp, and a hosted Anthropic-compatible endpoint wants none of it. Which of those
    // this turn is headed for is not known until the model has been resolved, so normalization
    // waits until after routing.
    let isStream = false;
    let parsed = null;
    if (req.method === "POST" && /\/v1\/messages/.test(url) && bodyBuf.length) {
      try {
        parsed = JSON.parse(bodyBuf.toString("utf8"));
        isStream = parsed.stream === true;
      } catch (e) { log(`body parse failed, forwarding as-is: ${e.message}`); }
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

    const failStream = (message, status = 502, type = "api_error") => {
      stopPings();
      if (res.headersSent) {
        res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type, message } })}\n\n`);
        return res.end();
      }
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify({ type: "error", error: { type, message } }));
    };

    // Token counting drives a progress meter rather than a turn, and there is no verb to forward it
    // to when nothing is loaded and nothing remote is routed, so it is estimated here.
    const countTokensReply = () => {
      const out = JSON.stringify({ input_tokens: estimateTokens(parsed ?? {}) });
      res.writeHead(200, { "content-type": "application/json", "content-length": String(Buffer.byteLength(out)) });
      res.end(out);
    };

    // --- routing ---------------------------------------------------------------------------------
    // `model` is the only thing that says which local model this turn wants. Anything we do not
    // recognise - the auto alias Claude Code's side-requests use, or a name from another provider -
    // is served by whatever is already loaded, which is what stops a conversation-title request
    // from evicting the 15 GB model half-way through a turn.
    // Non-null when this turn is going out to the network instead of to llama-server.
    let route = null;
    if (parsed?.model) {
      let entry = resolveModel(parsed.model, registry.models);
      if (!entry && parsed.model !== AUTO_ALIAS) {
        // A model pulled since this proxy started is worth one rescan before giving up on it.
        entry = resolveModel(parsed.model, rescan().models);
      }
      const st = readState();
      // The auto alias resolves to nothing on purpose - it means "whatever is current" - and when
      // what is current is a REMOTE there is no local server to serve it from. Claude Code's
      // side-requests all use that alias, so without this they would be sent to a llama-server
      // that a remote-only session never started.
      if (!entry && st.alias) {
        let cur = resolveModel(st.alias, registry.models);
        // remotes.json is edited BETWEEN requests far more often than models appear on disk: a
        // Kaggle tunnel hands out a new URL and key every session, and an entry is typically
        // disabled while its endpoint is down. A proxy that booted before such an edit has a stale
        // view, so the alias the state file calls current resolves to nothing here - and the turn
        // silently falls through to a llama-server that a remote-only session never started, which
        // surfaces as ECONNREFUSED on :20180 rather than anything mentioning the remote. The state
        // file already records `source`, so trust it and re-read rather than guess.
        if (!isRemote(cur) && st.source === "remote") {
          log(`state says ${st.alias} is remote but this proxy has not seen it - re-reading remotes.json`);
          cur = resolveModel(st.alias, rescan().models);
        }
        if (isRemote(cur)) entry = cur;
      }

      // Nothing is resident, and this turn names no model that could be made resident - the auto
      // alias means "whatever is loaded", and a bare `claudelocal` now starts with nothing loaded.
      // Forwarding it would reach a llama-server that was never started, so the reply says what to
      // do instead. It goes back as a 4xx: this is not a transient fault to be retried with backoff.
      if (!entry && !st.alias) {
        if (/count_tokens/.test(url)) return countTokensReply();
        log("request arrived with no model loaded");
        return failStream(
          "claudelocal: no model is loaded. Open /model and pick one - choosing a model is what loads it. "
          + "(Nothing loads at startup by design; 'claudelocal --cl-load' restores the old behaviour.)",
          400, "invalid_request_error");
      }

      if (isRemote(entry)) {
        route = entry;
        if (entry.alias !== st.alias) {
          // A remote is not loaded or unloaded, so switching to one is bookkeeping rather than a
          // two-minute stall: it costs nothing and takes effect on this very request. The local
          // server, if one is up, is deliberately left alone - switching back should be instant.
          writeState({ alias: entry.alias, lastAlias: entry.alias, modelName: entry.name, source: "remote", model: null });
          log(`routing to remote ${entry.alias} (${entry.remote.baseUrl})`);
        }
      } else if (entry && entry.file !== st.model) {
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

    // Last guard before the local path. If the state file says the current model is a remote and
    // nothing is loaded on this machine, a null route can only mean the entry has gone missing from
    // remotes.json - dialling llama-server would spend the whole retry sweep on ECONNREFUSED and
    // report a local failure for a remote model, which is the opposite of a useful error.
    if (!route && readState().source === "remote" && !readState().model) {
      const st = readState();
      if (/count_tokens/.test(url)) return countTokensReply();
      log(`no route for remote ${st.alias} - refusing to fall back to llama-server`);
      return failStream(
        `claudelocal: ${st.alias} is selected but no longer resolves to an entry in remotes.json `
        + "(is it still there, and \"enabled\": true?). Nothing is loaded locally, so this turn was "
        + "not sent to llama-server. Fix the entry, or pick another model with /model.",
        400, "invalid_request_error");
    }

    // A remote whose provider only speaks Chat Completions has no token-counting verb to forward
    // this to. The number only drives a progress meter, so it is estimated here rather than being
    // sent somewhere that would answer 404 and leave Claude Code retrying.
    if (route && route.remote.api === "openai" && /count_tokens/.test(url)) return countTokensReply();

    // --- the fallback chain ----------------------------------------------------------------------
    // A remote may name fallbacks: sibling entries pointing at the same provider under a different
    // key, or somewhere else entirely. NVIDIA's free tier answers 429 as soon as two requests
    // overlap, and Claude Code fires side-requests alongside the main turn, so a second key is not
    // a luxury - it is what makes that endpoint usable at all. A hop is only retried while nothing
    // of the answer has reached Claude Code yet; once a token is out, the turn is committed.
    const chain = [route];
    if (route) {
      const seen = new Set([route.alias]);
      const queue = [...route.remote.fallback];
      while (queue.length) {
        const alias = queue.shift();
        if (seen.has(alias)) continue;
        seen.add(alias);
        // Adding a key is an edit to remotes.json, and the natural time to make it is exactly when
        // the current ones start answering 429 - with a session already open. So a fallback this
        // proxy has never heard of is worth one rescan before it is written off, the same way an
        // unrecognised model name is.
        let next = resolveModel(alias, registry.models);
        if (!isRemote(next)) next = resolveModel(alias, rescan().models);
        if (!isRemote(next)) { log(`fallback "${alias}" is not an enabled remote - skipped`); continue; }
        chain.push(next);
        queue.push(...next.remote.fallback);
      }
    }

    const askedModel = parsed?.model ?? null;

    // The body is rebuilt per hop rather than once: two hops can publish different model ids, and
    // one may be an Anthropic endpoint while the next is an OpenAI one.
    function bodyFor(entry) {
      if (!parsed) return bodyBuf;
      if (!entry) {
        // llama-server checks the `model` field against the `--alias` it was started with, so the
        // request has to name what is loaded NOW - which may be a model that was not even resident
        // when this request arrived.
        normalize(parsed);
        parsed.model = readState().alias ?? parsed.model;
        return Buffer.from(JSON.stringify(parsed), "utf8");
      }
      const r = entry.remote;
      if (r.api === "openai") {
        const oai = toOpenAIRequest(parsed, { model: r.model, maxOutput: r.maxOutput ?? undefined, extra: r.extra ?? undefined });
        return Buffer.from(JSON.stringify(oai), "utf8");
      }
      // An Anthropic-compatible endpoint is built to receive Claude Code's own traffic - pointed
      // straight at one, Claude Code talks to it with no proxy at all - so the body goes out as it
      // arrived and only the model id is swapped for the one the provider publishes. Normalizing
      // here would strip prompt-caching hints that a hosted model actually honours.
      const body = { ...parsed, model: r.model };
      if (r.passthrough === false) normalize(body);
      const cap = r.maxOutput;
      if (cap && typeof body.max_tokens === "number" && body.max_tokens > cap) body.max_tokens = cap;
      return Buffer.from(JSON.stringify(body), "utf8");
    }

    // --- the send --------------------------------------------------------------------------------
    let emitted = false;          // true once any part of the ANSWER has been written downstream
    let current = null;           // the upstream response currently being drained
    let upstream = null;          // the upstream request, so a client hang-up can abort it
    let headTimer = null;
    // NVIDIA's free tier allows about one request in flight per key, and Claude Code fires
    // side-requests alongside the main turn - so "every key is busy" is a normal, transient state,
    // not a failure. Walking the chain again after a moment fixes it; Claude Code's own retry would
    // too, but it backs off in minutes and shows the user a network error while it waits.
    // The backoff is generous because the limit is measured over a window, not a queue depth: a
    // few hundred milliseconds is not long enough to clear one, and a streaming turn costs nothing
    // to wait out - the keep-alive pings hold the connection open the whole time.
    let sweeps = 0;
    let retryAfterMs = 0;
    const MAX_SWEEPS = 4;

    // Registered once rather than per hop: a failover would otherwise leave a drain listener bound
    // to a response object that has already been destroyed.
    res.on("drain", () => { try { current?.resume(); } catch { /* already gone */ } });
    // Abort upstream only when the CLIENT goes away mid-response. `req.on("close")` would be wrong:
    // Node fires it as soon as the request body has been read, which would kill the socket before
    // llama-server ever replied.
    res.on("close", () => {
      clearTimeout(headTimer);
      stopPings();
      if (!res.writableFinished && upstream && !upstream.destroyed) upstream.destroy();
    });

    function attempt(hop) {
      const entry = chain[hop];
      const bridge = !!entry && entry.remote.api === "openai";
      const body = bodyFor(entry);

      let transport = http;
      let reqOpts;
      let upstreamName = `llama-server on :${TARGET}`;

      if (entry) {
        const r = entry.remote;
        const key = remoteKey(entry);
        // A tunnelled llama-server needs no key at all, so a missing one is only an error when the
        // config said where to find it.
        if (!key && r.apiKeyEnv) {
          return failStream(`claudelocal: ${entry.alias} needs an API key - set ${r.apiKeyEnv}, or put "apiKey" in remotes.json`);
        }
        // Claude Code's own path is meaningless to a Chat Completions endpoint, so the bridge
        // substitutes the provider's verb for it.
        const suffix = bridge ? (r.path ?? "/v1/chat/completions") : url;
        const target = new URL(r.baseUrl + suffix);
        transport = target.protocol === "https:" ? https : http;
        upstreamName = `${entry.alias} at ${target.host}`;
        const h = {
          "content-type": "application/json",
          // An OpenAI endpoint decides between SSE and JSON on this header as well as on `stream`;
          // an Anthropic one goes by `stream` alone, so its request keeps whatever arrived.
          accept: bridge ? (isStream ? "text/event-stream" : "application/json")
                         : (req.headers.accept ?? "application/json"),
          ...r.headers,
          "content-length": String(body.length),
        };
        if (!bridge) {
          // Anthropic-compatible endpoints require the version header; Claude Code sends one, but a
          // request that arrived without it must still be valid upstream.
          h["anthropic-version"] = req.headers["anthropic-version"] ?? "2023-06-01";
          if (req.headers["anthropic-beta"]) h["anthropic-beta"] = req.headers["anthropic-beta"];
        }
        if (key) {
          // Anthropic's own API authenticates with `x-api-key`; Z.ai, Moonshot and every OpenAI
          // endpoint document a bearer token. An OpenAI provider has no reason to see the spare
          // header, so only the Anthropic path sends both.
          if (bridge || r.authStyle === "bearer" || r.authStyle === "both") h.authorization = `Bearer ${key}`;
          if (!bridge && (r.authStyle === "x-api-key" || r.authStyle === "both")) h["x-api-key"] = key;
        }
        reqOpts = {
          protocol: target.protocol, host: target.hostname,
          port: target.port || (target.protocol === "https:" ? 443 : 80),
          method: req.method, path: target.pathname + target.search, headers: h,
        };
      } else {
        const headers = { ...req.headers, host: `127.0.0.1:${TARGET}`, "content-length": String(body.length) };
        delete headers["accept-encoding"];   // keep the SSE stream plain
        delete headers["x-api-key"];         // llama-server runs without auth on loopback
        delete headers["authorization"];
        reqOpts = { host: "127.0.0.1", port: TARGET, method: req.method, path: url, headers };
      }

      // llama-server may not send response headers until the prompt is done either, so if nothing
      // has come back at all by the time the client would start worrying, commit to a stream
      // ourselves. The wait is long enough that a chat-template rejection - which comes back in
      // well under a second - still surfaces as a real HTTP error instead of being masked by a 200.
      clearTimeout(headTimer);
      headTimer = isStream ? setTimeout(beginStream, PRE_HEADER_MS) : null;

      const translator = bridge && isStream
        ? streamTranslator({ model: askedModel ?? entry.remote.model, reasoning: entry.remote.reasoning })
        : null;

      upstream = transport.request(reqOpts, (r) => {
        clearTimeout(headTimer);
        current = r;
        const status = r.statusCode ?? 0;

        // Capture the body of an upstream rejection: without it a 400 from a chat template, or a
        // provider's bare `{"status":429}`, is completely opaque from inside Claude Code.
        if (status >= 400) {
          const errChunks = [];
          r.on("data", (c) => errChunks.push(c));
          r.on("end", () => {
            let text = Buffer.concat(errChunks).toString("utf8");
            log(`upstream ${status} from ${upstreamName}: ${text.slice(0, 2000)}`);
            const ra = parseFloat(r.headers["retry-after"] ?? "");
            if (Number.isFinite(ra) && ra > 0) retryAfterMs = Math.min(ra * 1000, 30000);

            // 429 is the expected answer from a free tier under any concurrency, and a 5xx is worth
            // one more key rather than a failed turn.
            // A crashed engine is permanent: a sibling key is still worth trying, because it may be
            // a different process, but sweeping back over the same corpse never is.
            const dead = isDeadUpstream(text);
            if ((status === 429 || status >= 500) && !emitted) {
              if (hop + 1 < chain.length) {
                log(`failing over ${entry.alias} -> ${chain[hop + 1].alias} after ${status}`);
                return attempt(hop + 1);
              }
              if (dead) log(`${upstreamName} reports a dead inference engine - not retrying`);
              if (chain[0] && sweeps < MAX_SWEEPS && !dead) {
                sweeps += 1;
                // A provider that says how long to wait is more accurate than any backoff curve.
                const wait = Math.max(retryAfterMs, 800 * 2 ** sweeps);
                log(`whole chain returned ${status} - retrying from ${chain[0].alias} in ${wait}ms (sweep ${sweeps}/${MAX_SWEEPS})`);
                if (isStream) beginStream();   // start pinging, or the wait reads as a dead connection
                return setTimeout(() => attempt(0), wait);
              }
            }

            text = dead ? explainDeadUpstream(text, upstreamName) : explainContextOverflow(text);
            // If the ping timer already committed us to a 200 stream there is no status code left
            // to send, so deliver the failure the only way an open SSE stream can.
            if (res.headersSent) {
              stopPings();
              res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message: `claudelocal: ${upstreamName} ${status}: ${text.slice(0, 500)}` } })}\n\n`);
              return res.end();
            }
            // The rewritten message is longer than what upstream sent, so its content-length header
            // would truncate the body mid-JSON and the client would report a parse error instead of
            // the explanation.
            const outHeaders = { ...r.headers };
            delete outHeaders["content-length"];
            outHeaders["content-length"] = String(Buffer.byteLength(text, "utf8"));
            res.writeHead(status || 502, outHeaders);
            res.end(text);
          });
          return;
        }

        if (translator) {
          // The OpenAI stream is rewritten frame by frame into the Anthropic one Claude Code
          // expects. Headers go out first so the keep-alive pings can start covering the gap while
          // a reasoning-heavy model thinks.
          beginStream();
          r.on("data", (c) => {
            touch();
            const out = translator.push(c);
            if (out) { emitted = true; if (!res.write(out)) r.pause(); }
          });
          r.on("end", () => { res.write(translator.end()); stopPings(); res.end(); });
          r.on("error", () => { stopPings(); res.end(); });
          return;
        }

        if (bridge) {
          const outChunks = [];
          r.on("data", (c) => outChunks.push(c));
          r.on("end", () => {
            stopPings();
            let out;
            try {
              const oai = JSON.parse(Buffer.concat(outChunks).toString("utf8"));
              out = JSON.stringify(toAnthropicResponse(oai, { model: askedModel, reasoning: entry.remote.reasoning }));
            } catch (e) {
              return failStream(`claudelocal: ${upstreamName} sent a response this bridge could not read - ${e.message}`);
            }
            emitted = true;
            if (res.headersSent) return res.end();
            res.writeHead(200, { "content-type": "application/json", "content-length": String(Buffer.byteLength(out, "utf8")) });
            res.end(out);
          });
          r.on("error", () => { stopPings(); res.end(); });
          return;
        }

        if (!res.headersSent) res.writeHead(status || 502, r.headers);
        startPings();
        r.on("data", (c) => { touch(); emitted = true; if (!res.write(c)) r.pause(); });
        r.on("end", () => { stopPings(); res.end(); });
        r.on("error", () => { stopPings(); res.end(); });
      });

      upstream.on("error", (e) => {
        clearTimeout(headTimer);
        // A provider that refuses the connection outright is exactly what a fallback key is for.
        if (!emitted && hop + 1 < chain.length) {
          log(`${upstreamName} unreachable (${e.message}) - failing over to ${chain[hop + 1].alias}`);
          return attempt(hop + 1);
        }
        if (!emitted && chain[0] && sweeps < MAX_SWEEPS) {
          sweeps += 1;
          const wait = 400 * 2 ** sweeps;
          log(`whole chain unreachable - retrying from ${chain[0].alias} in ${wait}ms (sweep ${sweeps}/${MAX_SWEEPS})`);
          return setTimeout(() => attempt(0), wait);
        }
        stopPings();
        log(`upstream unreachable (${upstreamName}): ${e.message}`);
        if (res.headersSent) return res.end();
        res.writeHead(502, { "content-type": "application/json" });
        res.end(JSON.stringify({
          type: "error",
          error: { type: "api_error", message: `claudelocal: ${upstreamName} unreachable (${e.message}) - try 'claudelocal status'` },
        }));
      });

      upstream.end(body);
    }

    attempt(0);
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
