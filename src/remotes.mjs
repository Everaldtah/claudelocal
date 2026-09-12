// Models that are not on this machine.
//
// Some models cannot be run locally at any quantization: GLM-5.3 is 744B-A40B and its smallest
// useful quant is ~150 GB; Kimi K3 is 2.8T-A104B and its 1-bit dynamic quant is 594 GB. A 6 GB card
// and 32 GB of RAM is not the constraint that fails first - a 20 GB working disk is.
//
// Most are reachable over an Anthropic-compatible endpoint, and so is a llama-server running
// somewhere else (a Kaggle T4x2 behind a tunnel, another box on the LAN). Claude Code already
// speaks that protocol, and so does the proxy, so those need no translation layer at all - only a
// base URL, a model id, and a key. The exception is `api: "openai"`, for a provider that publishes
// Chat Completions and nothing else; openai.mjs bridges it. Together that covers three cases:
//
//   provider API   GLM-5.3 at Z.ai, Kimi K3 at Moonshot or NVIDIA - full size, no quant loss
//   borrowed GPU   llama-server on a Kaggle/Colab/RunPod box, tunnelled to a public URL
//   another box    a llama-server on the LAN
//
// Config lives in remotes.json beside models.json. It is gitignored, because it may hold a key.
import fs from "node:fs";
import path from "node:path";
import { paths, readJSON } from "./sys.mjs";
import { SUBSCRIPTION_AUTH, subscriptionToken } from "./subscription.mjs";

const CONFIG = path.join(paths.root, "remotes.json");
export const REMOTES_FILE = CONFIG;

// A remote's window is a claim in a config file rather than something read out of a GGUF header,
// so it is capped at what Claude Code can actually be told to use. Nothing here is load-bearing on
// a shared session: the session window is the SMALLEST in the picker, and every remote is larger
// than every local model, so a wrong number here changes nothing until the picker is remote-only.
const MAX_CTX = 1000000;
const DEFAULT_CTX = 200000;

// `false` would be a lie for a hosted model - these have no context in the GGUF sense at all - but
// the registry, the planner and the picker all key off these fields, so a remote has to present
// the same shape as a local model or every consumer needs a special case.
function shape(alias, r) {
  // A subscription entry carries no key in the file at all - the token lives in Claude Code's own
  // credential store and is read per request, so `key` stays null here on purpose.
  const sub = r.auth === SUBSCRIPTION_AUTH;
  const key = sub ? null : (r.apiKey ?? (r.apiKeyEnv ? process.env[r.apiKeyEnv] : null) ?? null);
  return {
    alias,
    source: "remote",
    ref: r.baseUrl,
    file: null,             // nothing on disk - every `m.file` comparison must survive a null
    mmproj: null,
    name: r.label ?? alias,
    arch: "remote",
    sizeLabel: r.sizeLabel ?? null,
    quant: null,
    bytes: 0,
    nCtxTrain: Math.min(r.ctx ?? DEFAULT_CTX, MAX_CTX),
    nBlocks: 0,
    hasMTP: false,
    remote: {
      baseUrl: String(r.baseUrl).replace(/\/+$/, ""),
      model: r.model ?? alias,
      apiKey: key,
      apiKeyEnv: sub ? null : (r.apiKeyEnv ?? null),
      // "claude-subscription" means: authenticate as the signed-in Claude subscription, with the
      // OAuth token read from disk at call time. Anything else authenticates with a key.
      auth: sub ? SUBSCRIPTION_AUTH : null,
      // "anthropic" is passthrough. "openai" routes the turn through the translator in openai.mjs,
      // which is what makes NVIDIA's build.nvidia.com catalog - Chat Completions only - usable here.
      api: r.api === "openai" ? "openai" : "anthropic",
      // Where the endpoint puts its completions verb. Only meaningful for the openai bridge, where
      // Claude Code's own `/v1/messages` path is meaningless upstream.
      path: r.path ?? null,
      // Body fields merged in after translation: `reasoning_effort`, `seed`, and anything else a
      // provider adds on top of the Chat Completions schema.
      extra: r.extra && typeof r.extra === "object" ? r.extra : null,
      // What to do with `reasoning_content`: "think" makes a real thinking block, "text" folds it
      // into the answer, "drop" discards it.
      reasoning: ["think", "text", "drop"].includes(r.reasoning) ? r.reasoning : "think",
      // Aliases to try, in order, when this one answers 429 or 5xx. Two keys against the same free
      // NVIDIA endpoint double the rate limit, which is the whole point of having a second entry.
      fallback: Array.isArray(r.fallback) ? r.fallback.filter((x) => typeof x === "string")
              : typeof r.fallback === "string" ? [r.fallback] : [],
      // Anthropic's own API authenticates with `x-api-key`; Z.ai and Moonshot document a bearer
      // token. Sending both satisfies every one of them and no provider rejects the spare header.
      // A subscription token is a bearer token and the API rejects it in `x-api-key`, so the
      // config does not get a say in how it is presented.
      authStyle: sub ? "bearer" : (r.authStyle ?? "both"),
      headers: r.headers ?? {},
      // A hosted model answers in seconds, so the local 8192-token output clamp - which exists
      // because a CPU-offloaded 27B would take an hour to fill a large one - is pointless here.
      maxOutput: r.maxOutput ?? null,
      // These endpoints are built to receive raw Claude Code traffic; Claude Code talks to them
      // directly with no proxy at all. Rewriting the body would only break what already works.
      passthrough: r.passthrough !== false,
      note: r.note ?? null,
    },
  };
}

function valid(alias, r) {
  if (!r || typeof r !== "object") return "not an object";
  if (r.enabled === false) return "disabled";
  if (!r.baseUrl || !/^https?:\/\//i.test(String(r.baseUrl))) return "baseUrl must be an http(s) URL";
  if (!/^[a-z0-9][a-z0-9.\-]*$/.test(alias)) return "alias must be lowercase letters, digits, dots and dashes";
  return null;
}

/**
 * Every configured remote, plus a reason for each one left out. Never throws: a malformed
 * remotes.json must not stop the local models from loading.
 */
export function discoverRemotes() {
  if (!fs.existsSync(CONFIG)) return { models: [], skipped: [] };
  const cfg = readJSON(CONFIG, null);
  if (!cfg || typeof cfg !== "object") {
    return { models: [], skipped: [{ ref: CONFIG, reason: "remotes.json is not valid JSON" }] };
  }
  const models = [];
  const skipped = [];
  // A `remotes` wrapper key is the natural shape to reach for, so accept it as well as the bare map.
  const entries = Object.entries(cfg.remotes && typeof cfg.remotes === "object" ? cfg.remotes : cfg);
  for (const [alias, r] of entries) {
    if (alias === "$schema" || alias === "//") continue;
    const why = valid(alias, r);
    if (why) { skipped.push({ ref: alias, source: "remote", reason: why }); continue; }
    models.push(shape(alias, r));
  }
  return { models, skipped };
}

/** The key for a remote, resolved at call time so an env var set after launch still works. */
export function remoteKey(entry) {
  const r = entry?.remote;
  if (!r) return null;
  // The subscription token rotates hourly, so it is fetched per call and never held anywhere.
  if (r.auth === SUBSCRIPTION_AUTH) return subscriptionToken().token;
  return r.apiKey ?? (r.apiKeyEnv ? process.env[r.apiKeyEnv] ?? null : null);
}

/** Why a remote has no usable credential right now, or null when it has one. */
export function remoteKeyProblem(entry) {
  const r = entry?.remote;
  if (!r) return null;
  if (r.auth === SUBSCRIPTION_AUTH) return subscriptionToken().reason;
  if (!remoteKey(entry) && r.apiKeyEnv) return `${r.apiKeyEnv} is not set`;
  return null;
}

export const isRemote = (m) => m?.source === "remote" && !!m.remote;
