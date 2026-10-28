// Anthropic Messages <-> Codex Responses. Claude Code owns tool execution.
import { toOpenAIRequest } from "./openai.mjs";
import { CODEX_API, codexHeaders } from "./codex-subscription.mjs";

export function toResponsesRequest(body, { model, extra } = {}) {
  const chat = toOpenAIRequest(body, { model });
  const input = [];
  let instructions = "";
  for (const m of chat.messages) {
    if (m.role === "system") { instructions += m.content + "\n"; continue; }
    if (m.role === "tool") {
      input.push({ type: "function_call_output", call_id: m.tool_call_id, output: m.content });
      continue;
    }
    const parts = typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content ?? [];
    const content = parts.map(p => p.type === "image_url"
      ? { type: "input_image", image_url: p.image_url.url }
      : { type: m.role === "assistant" ? "output_text" : "input_text", text: p.text });
    if (content.length) input.push({ type: "message", role: m.role, content });
    for (const call of m.tool_calls ?? []) input.push({ type: "function_call", call_id: call.id,
      name: call.function.name, arguments: call.function.arguments });
  }
  const out = { model, instructions: instructions.trim() || "You are a helpful coding assistant.",
    input, store: false, stream: true };
  if (chat.tools?.length) {
    out.tools = chat.tools.map(t => ({ type: "function", ...t.function, strict: false }));
    out.tool_choice = typeof chat.tool_choice === "object"
      ? { type: "function", name: chat.tool_choice.function.name } : chat.tool_choice;
    out.parallel_tool_calls = !body.tool_choice?.disable_parallel_tool_use;
  }
  // Only allow supported overrides; credentials, storage and transport are fixed.
  const effort = extra?.reasoning?.effort ?? extra?.reasoning_effort;
  if (effort) out.reasoning = { effort };
  return out;
}

const usageOf = (u) => {
  const cached = u?.input_tokens_details?.cached_tokens ?? 0;
  return { input_tokens: Math.max(0, (u?.input_tokens ?? 0) - cached), output_tokens: u?.output_tokens ?? 0,
    cache_creation_input_tokens: 0, cache_read_input_tokens: cached };
};
const sse = (event, data) => `event: ${event}\ndata: ${JSON.stringify({ type: event, ...data })}\n\n`;

export function toAnthropicMessage(response, model) {
  const content = [];
  for (const item of response.output ?? []) {
    if (item.type === "message") for (const p of item.content ?? []) {
      if (p.type === "output_text") content.push({ type: "text", text: p.text });
      if (p.type === "refusal") content.push({ type: "text", text: p.refusal });
    }
    if (item.type === "function_call") content.push({ type: "tool_use", id: item.call_id,
      name: item.name, input: JSON.parse(item.arguments || "{}") });
  }
  return { id: response.id, type: "message", role: "assistant", model,
    content: content.length ? content : [{ type: "text", text: "" }],
    stop_reason: response.status === "incomplete" ? "max_tokens" : content.some(c => c.type === "tool_use") ? "tool_use" : "end_turn",
    stop_sequence: null, usage: usageOf(response.usage) };
}

// Preserve UTF-8 across arbitrary byte boundaries. An interrupted stream is never success.
export function responsesTranslator({ model, onChunk = () => {} } = {}) {
  const decoder = new TextDecoder();
  let buffer = "", data = [], started = false, completed = null, nextIndex = 0;
  const blocks = new Map();
  function emit(event, value) { onChunk(sse(event, value)); }
  function start(response = {}) {
    if (started) return;
    started = true;
    emit("message_start", { message: { id: response.id ?? `msg_${Date.now()}`, type: "message", role: "assistant",
      model, content: [], stop_reason: null, stop_sequence: null, usage: usageOf(null) } });
  }
  function block(key, content) {
    if (blocks.has(key)) return blocks.get(key);
    start();
    const b = { index: nextIndex++, text: "", closed: false };
    blocks.set(key, b);
    emit("content_block_start", { index: b.index, content_block: content });
    return b;
  }
  function delta(b, text, tool = false) {
    if (!text) return;
    b.text += text;
    emit("content_block_delta", { index: b.index, delta: tool
      ? { type: "input_json_delta", partial_json: text } : { type: "text_delta", text } });
  }
  function close(b) {
    if (b && !b.closed) { b.closed = true; emit("content_block_stop", { index: b.index }); }
  }
  function event(e) {
    if (completed) return;
    const key = `${e.output_index}:${e.content_index ?? 0}`;
    if (e.type === "error" || e.type === "response.failed") {
      throw new Error(e.response?.error?.message ?? e.error?.message ?? e.message ?? "Codex request failed");
    }
    if (e.type === "response.created") start(e.response);
    if (e.type === "response.output_item.added" && e.item?.type === "function_call") {
      block(`tool:${e.output_index}`, { type: "tool_use", id: e.item.call_id, name: e.item.name, input: {} });
    }
    if (e.type === "response.output_text.delta" || e.type === "response.refusal.delta") {
      delta(block(key, { type: "text", text: "" }), e.delta);
    }
    if (e.type === "response.output_text.done" || e.type === "response.refusal.done") {
      const b = block(key, { type: "text", text: "" });
      if (!b.text) delta(b, e.text ?? e.refusal);
      close(b);
    }
    if (e.type === "response.function_call_arguments.delta") {
      const b = blocks.get(`tool:${e.output_index}`);
      if (!b) throw new Error("Codex sent tool arguments without a tool call");
      delta(b, e.delta, true);
    }
    if (e.type === "response.output_item.done" && e.item?.type === "function_call") {
      const b = block(`tool:${e.output_index}`, { type: "tool_use", id: e.item.call_id, name: e.item.name, input: {} });
      if (!b.text) delta(b, e.item.arguments || "{}", true);
      JSON.parse(b.text);
      close(b);
    }
    if (e.type === "response.completed" || e.type === "response.incomplete") {
      if (e.type === "response.incomplete" && e.response?.incomplete_details?.reason !== "max_output_tokens") {
        throw new Error(`Codex response incomplete: ${e.response?.incomplete_details?.reason ?? "unknown"}`);
      }
      const message = toAnthropicMessage(e.response, model);
      start(e.response);
      for (const b of blocks.values()) close(b);
      emit("message_delta", { delta: { stop_reason: message.stop_reason, stop_sequence: null }, usage: message.usage });
      emit("message_stop", {});
      completed = message;
    }
  }
  function consume(text) {
    buffer += text;
    let end;
    while ((end = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, end).replace(/\r$/, "");
      buffer = buffer.slice(end + 1);
      if (!line) {
        if (data.length) { const raw = data.join("\n"); data = []; if (raw !== "[DONE]") event(JSON.parse(raw)); }
      } else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
  }
  return {
    push(chunk) { consume(typeof chunk === "string" ? chunk : decoder.decode(chunk, { stream: true })); },
    end() {
      consume(decoder.decode() + "\n\n");
      if (!completed) throw new Error("Codex stream ended before response.completed; retry the request");
      return completed;
    },
  };
}

export async function codexRequest(body, remote, { onChunk, signal } = {}) {
  const response = await fetch(`${CODEX_API}/responses`, {
    method: "POST", headers: codexHeaders(), redirect: "error",
    body: JSON.stringify(toResponsesRequest(body, remote)),
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(3600000)]) : AbortSignal.timeout(3600000),
  });
  if (!response.ok) {
    const raw = await response.text();
    let detail;
    try { const data = JSON.parse(raw); detail = data.error?.message ?? data.detail; } catch {}
    const error = new Error(`Codex HTTP ${response.status}: ${String(detail ?? response.statusText).slice(0, 500)}${response.status === 401 ? "; run 'codex login'" : ""}`);
    error.status = response.status;
    throw error;
  }
  const translator = responsesTranslator({ model: body.model ?? remote.model, onChunk });
  for await (const chunk of response.body) translator.push(chunk);
  return translator.end();
}
