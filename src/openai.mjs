// Anthropic Messages  <->  OpenAI Chat Completions.
//
// Every other upstream in this project speaks the Anthropic Messages API natively - llama-server
// does, and so do Z.ai and Moonshot's own Anthropic-compatible endpoints - which is why
// remotes.mjs could get away with pure passthrough. NVIDIA's build.nvidia.com catalog does not:
// it is OpenAI Chat Completions and nothing else. That catalog is the cheapest way to reach a
// frontier-size model like Kimi K3 (2.8T-A104B, whose 1-bit GGUF is 594 GB), so it earns a
// translator.
//
// The hard parts are not the field names, they are the shape mismatches:
//
//   tool results   Anthropic carries them as `tool_result` blocks inside a USER turn. OpenAI wants
//                  one `role:"tool"` message per result, immediately after the assistant turn that
//                  called them - so one Anthropic message can fan out into several OpenAI ones.
//   reasoning      OpenAI has no thinking block. NVIDIA streams `reasoning_content` deltas
//                  alongside `content`, so they become a real Anthropic thinking block.
//   streaming      OpenAI sends one flat delta channel whose block structure is implied by which
//                  field moved; Anthropic brackets every block with explicit start/stop events.
//                  The stream translator is therefore a state machine, not a mapper.
//
// Thinking blocks are emitted downstream but STRIPPED on the way back up: Anthropic requires a
// valid `signature` on a thinking block replayed into a later turn, and there is no real signature
// to give. Dropping them on the request side keeps the round trip legal.

// ---------------------------------------------------------------------------------------------
// request:  Anthropic  ->  OpenAI
// ---------------------------------------------------------------------------------------------

const asBlocks = (c) =>
  typeof c === "string" ? [{ type: "text", text: c }] : Array.isArray(c) ? c : c ? [c] : [];

const textOf = (blocks) =>
  asBlocks(blocks).filter((b) => b?.type === "text").map((b) => b.text ?? "").join("\n").trim();

// Anthropic images are usually inline base64; OpenAI takes a single URL field, and a data: URI is
// a legal value for it, so both source shapes collapse to the same thing.
function imagePart(b) {
  const s = b?.source;
  if (!s) return null;
  if (s.type === "base64" && s.data) {
    return { type: "image_url", image_url: { url: `data:${s.media_type || "image/png"};base64,${s.data}` } };
  }
  if (s.type === "url" && s.url) return { type: "image_url", image_url: { url: s.url } };
  return null;
}

// A JSON Schema straight out of Claude Code carries keys that strict OpenAI-compatible validators
// reject ($schema chief among them). Everything else is left alone - narrowing a tool's schema
// would change what the model is allowed to send back.
function toolSchema(input_schema) {
  if (!input_schema || typeof input_schema !== "object") return { type: "object", properties: {} };
  const { $schema, ...rest } = input_schema;
  if (!rest.type) rest.type = "object";
  if (rest.type === "object" && !rest.properties) rest.properties = {};
  return rest;
}

// Flatten a tool_result down to what a `role:"tool"` message can hold. Images cannot go there
// under the OpenAI schema, so they are handed back separately for the caller to re-emit as a user
// turn - which is how a screenshot taken by a tool still reaches a vision model.
function toolResultContent(block) {
  const images = [];
  const c = block?.content;
  if (typeof c === "string") return { text: c, images };
  const parts = [];
  for (const b of asBlocks(c)) {
    if (b?.type === "text") parts.push(b.text ?? "");
    else if (b?.type === "image") { const p = imagePart(b); if (p) images.push(p); }
  }
  if (images.length && !parts.length) {
    parts.push(`[${images.length} image(s) returned by the tool, sent in the next message]`);
  }
  return { text: parts.join("\n"), images };
}

/**
 * Translate an Anthropic Messages request body into an OpenAI Chat Completions one.
 *
 * @param {object} body        the Anthropic request, as Claude Code sent it
 * @param {object} opts
 * @param {string} opts.model      the model id the provider publishes
 * @param {number} [opts.maxOutput] clamp on max_tokens
 * @param {object} [opts.extra]     provider-specific body fields, merged in last
 */
export function toOpenAIRequest(body, { model, maxOutput, extra } = {}) {
  const msgs = [];

  // Anthropic keeps the system prompt out of `messages`; OpenAI wants it as the first message.
  const sys = typeof body.system === "string" ? body.system : textOf(body.system);
  if (sys) msgs.push({ role: "system", content: sys });

  for (const m of Array.isArray(body.messages) ? body.messages : []) {
    const blocks = asBlocks(m.content);

    if (m.role === "assistant") {
      const text = blocks.filter((b) => b?.type === "text").map((b) => b.text ?? "").join("");
      const calls = blocks.filter((b) => b?.type === "tool_use").map((b) => ({
        id: b.id,
        type: "function",
        function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) },
      }));
      // `thinking` blocks are deliberately not forwarded - see the header note on signatures.
      if (!text && !calls.length) continue;
      const out = { role: "assistant", content: text || null };
      if (calls.length) out.tool_calls = calls;
      msgs.push(out);
      continue;
    }

    // A user turn may be pure tool results, pure content, or both. The tool messages have to come
    // first and unbroken, or the conversation is rejected as an orphaned tool reply.
    const trailing = [];
    let emittedTool = false;
    for (const b of blocks) {
      if (b?.type !== "tool_result") continue;
      const { text, images } = toolResultContent(b);
      msgs.push({ role: "tool", tool_call_id: b.tool_use_id, content: text || "(no output)" });
      emittedTool = true;
      trailing.push(...images);
    }
    for (const b of blocks) {
      if (b?.type === "text") { if (b.text) trailing.push({ type: "text", text: b.text }); }
      else if (b?.type === "image") { const p = imagePart(b); if (p) trailing.push(p); }
    }
    if (trailing.length) {
      // A lone text part reads better as a plain string, and some providers are pickier about the
      // array form than the spec suggests.
      const only = trailing.length === 1 && trailing[0].type === "text";
      msgs.push({ role: "user", content: only ? trailing[0].text : trailing });
    } else if (!emittedTool) {
      msgs.push({ role: "user", content: "" });
    }
  }

  const req = { model, messages: msgs, stream: body.stream === true };

  const asked = typeof body.max_tokens === "number" ? body.max_tokens : undefined;
  const max = maxOutput ? Math.min(asked ?? maxOutput, maxOutput) : asked;
  if (max) req.max_tokens = max;
  if (typeof body.temperature === "number") req.temperature = body.temperature;
  if (typeof body.top_p === "number") req.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length) req.stop = body.stop_sequences;

  if (Array.isArray(body.tools) && body.tools.length) {
    const fns = body.tools
      // Anthropic server-side tools (web_search, computer use, ...) carry a `type` and no schema.
      // An OpenAI endpoint cannot run them, so they are dropped rather than sent as broken functions.
      .filter((t) => t && t.name && (!t.type || t.type === "custom"))
      .map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description ?? "", parameters: toolSchema(t.input_schema) },
      }));
    if (fns.length) {
      req.tools = fns;
      const tc = body.tool_choice;
      if (tc?.type === "any") req.tool_choice = "required";
      else if (tc?.type === "tool" && tc.name) req.tool_choice = { type: "function", function: { name: tc.name } };
      else if (tc?.type === "none") req.tool_choice = "none";
      else req.tool_choice = "auto";
    }
  }

  // Without this the final chunk carries no usage at all, and Claude Code's context meter never moves.
  if (req.stream) req.stream_options = { include_usage: true };
  return extra && typeof extra === "object" ? { ...req, ...extra } : req;
}

// ---------------------------------------------------------------------------------------------
// response:  OpenAI  ->  Anthropic
// ---------------------------------------------------------------------------------------------

const STOP = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  function_call: "tool_use",
  content_filter: "end_turn",
};

// Claude Code replays a thinking block verbatim on the next turn and the real API validates its
// signature - but this bridge strips thinking on the way up, so nothing ever checks this value.
// It exists because the block's shape requires the field, not because it means anything.
const SIGNATURE = "claudelocal-openai-bridge";

const usageOf = (u) => ({
  input_tokens: u?.prompt_tokens ?? 0,
  output_tokens: u?.completion_tokens ?? 0,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: u?.prompt_tokens_details?.cached_tokens ?? 0,
});

const rid = (p) => `${p}_${Math.random().toString(36).slice(2, 14)}`;

/** Translate a complete (non-streaming) OpenAI response into an Anthropic message. */
export function toAnthropicResponse(oai, { model, reasoning = "think" } = {}) {
  const choice = oai?.choices?.[0] ?? {};
  const m = choice.message ?? {};
  const content = [];

  const think = m.reasoning_content ?? m.reasoning ?? null;
  if (think && reasoning === "think") content.push({ type: "thinking", thinking: think, signature: SIGNATURE });
  else if (think && reasoning === "text") content.push({ type: "text", text: think });

  if (m.content) content.push({ type: "text", text: m.content });

  for (const c of m.tool_calls ?? []) {
    let input = {};
    try { input = JSON.parse(c.function?.arguments || "{}"); } catch { input = {}; }
    content.push({ type: "tool_use", id: c.id || rid("toolu"), name: c.function?.name, input });
  }
  // Anthropic's schema has no empty-content message, and Claude Code renders one as a blank turn.
  if (!content.length) content.push({ type: "text", text: "" });

  return {
    id: oai?.id || rid("msg"),
    type: "message",
    role: "assistant",
    model: model || oai?.model,
    content,
    stop_reason: STOP[choice.finish_reason] ?? "end_turn",
    stop_sequence: null,
    usage: usageOf(oai?.usage),
  };
}

// ---------------------------------------------------------------------------------------------
// streaming
// ---------------------------------------------------------------------------------------------

const sse = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

/**
 * A state machine that turns OpenAI SSE chunks into an Anthropic SSE stream.
 *
 * OpenAI's stream is one flat sequence of deltas whose block structure is implied by which field
 * moved (`reasoning_content`, `content`, `tool_calls[i]`). Anthropic's is explicitly bracketed, so
 * every switch between those channels becomes a content_block_stop / content_block_start pair.
 *
 * Feed it raw bytes with `push()`, then call `end()`. Both return SSE text to write downstream.
 */
export function streamTranslator({ model, reasoning = "think" } = {}) {
  let buf = "";
  let started = false;
  let index = -1;
  let open = null;         // "thinking" | "text" | "tool"
  let toolSlot = null;     // which OpenAI tool_calls index the open block belongs to
  let stopReason = "end_turn";
  let usage = null;
  let messageId = null;
  let sawDone = false;

  function closeBlock() {
    if (!open) return "";
    let out = "";
    // A thinking block is only well-formed once its signature has arrived, and the real API sends
    // that last, so it is emitted at the close rather than up front.
    if (open === "thinking") {
      out += sse("content_block_delta", {
        type: "content_block_delta", index, delta: { type: "signature_delta", signature: SIGNATURE },
      });
    }
    out += sse("content_block_stop", { type: "content_block_stop", index });
    open = null;
    toolSlot = null;
    return out;
  }

  function openBlock(kind, block) {
    let out = closeBlock();
    index += 1;
    open = kind;
    out += sse("content_block_start", { type: "content_block_start", index, content_block: block });
    return out;
  }

  function start() {
    if (started) return "";
    started = true;
    return sse("message_start", {
      type: "message_start",
      message: {
        id: messageId || rid("msg"),
        type: "message", role: "assistant", model: model || "unknown",
        content: [], stop_reason: null, stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  }

  function chunk(c) {
    let out = "";
    if (c?.id && !messageId) messageId = c.id;
    if (c?.usage) usage = c.usage;
    const choice = c?.choices?.[0];
    if (!choice) return out;
    const d = choice.delta ?? {};

    const think = d.reasoning_content ?? d.reasoning;
    if (think && reasoning !== "drop") {
      if (reasoning === "think") {
        if (open !== "thinking") out += openBlock("thinking", { type: "thinking", thinking: "" });
        out += sse("content_block_delta", {
          type: "content_block_delta", index, delta: { type: "thinking_delta", thinking: think },
        });
      } else {
        if (open !== "text") out += openBlock("text", { type: "text", text: "" });
        out += sse("content_block_delta", {
          type: "content_block_delta", index, delta: { type: "text_delta", text: think },
        });
      }
    }

    if (d.content) {
      if (open !== "text") out += openBlock("text", { type: "text", text: "" });
      out += sse("content_block_delta", {
        type: "content_block_delta", index, delta: { type: "text_delta", text: d.content },
      });
    }

    for (const tc of d.tool_calls ?? []) {
      const slot = tc.index ?? 0;
      // A new slot means a new call. The id and name only appear on the first delta of each one;
      // everything after that is argument text, which Anthropic wants as raw JSON fragments.
      if (open !== "tool" || toolSlot !== slot) {
        out += openBlock("tool", {
          type: "tool_use", id: tc.id || rid("toolu"), name: tc.function?.name ?? "", input: {},
        });
        toolSlot = slot;
      }
      const args = tc.function?.arguments;
      if (args) {
        out += sse("content_block_delta", {
          type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: args },
        });
      }
    }

    if (choice.finish_reason) stopReason = STOP[choice.finish_reason] ?? "end_turn";
    return out;
  }

  return {
    /** Bytes in, Anthropic SSE text out. Handles a chunk boundary falling mid-event. */
    push(buffer) {
      buf += buffer.toString("utf8");
      let out = start();
      // Frames are separated by a blank line, but NVIDIA sends one `data:` line per frame, so
      // splitting on newlines and skipping the blanks handles both without a full frame parser.
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const raw of lines) {
        const line = raw.trim();
        if (!line || line.startsWith(":") || line.startsWith("event:")) continue;
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") { sawDone = true; continue; }
        let c;
        try { c = JSON.parse(payload); } catch { continue; }
        // Some gateways report a mid-stream failure in the body instead of via a status code.
        if (c?.error) {
          out += closeBlock();
          out += sse("error", {
            type: "error",
            error: { type: "api_error", message: `upstream: ${c.error.message ?? JSON.stringify(c.error)}` },
          });
          sawDone = true;
          continue;
        }
        out += chunk(c);
      }
      return out;
    },

    /** The closing bracket: whatever block is still open, then the message tail. */
    end() {
      let out = start();
      out += closeBlock();
      out += sse("message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: usage?.completion_tokens ?? 0 },
      });
      out += sse("message_stop", { type: "message_stop" });
      return out;
    },

    /** True once a well-formed stream has ended - used to tell a truncated stream from a clean one. */
    get complete() { return sawDone; },
  };
}

/**
 * Claude Code asks the API to count a prompt's tokens before sending it. There is no OpenAI
 * equivalent, and a wrong answer here only skews a progress meter, so the request is answered
 * locally from a characters-per-token estimate rather than forwarded to something that would 404.
 */
export function estimateTokens(body) {
  let chars = 0;
  const walk = (v) => {
    if (typeof v === "string") chars += v.length;
    else if (Array.isArray(v)) v.forEach(walk);
    else if (v && typeof v === "object") Object.values(v).forEach(walk);
  };
  walk(body?.system);
  walk(body?.messages);
  walk(body?.tools);
  return Math.max(1, Math.round(chars / 4));
}
