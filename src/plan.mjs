// The planner: decides how much of the model lives in VRAM, how much stays in system RAM, what
// the KV cache is made of, and how big the context can be - all from the real tensor table plus
// live free VRAM/RAM.
//
// Three facts about this machine and this model drive every number below.
//
// 1. Qwen3.8-27B is DENSE. Every weight is read for every token, so there is no MoE-style win
//    from splitting by tensor family. What matters is only how many bytes sit on the fast side.
//
// 2. Its attention is HYBRID: 48 DeltaNet (linear/SSM) layers keep a small fixed recurrent state,
//    and only 16 of 64 layers keep a real KV cache. That is what makes 262144 tokens reachable at
//    all on 32 GB - a normal 27B dense model would want ~34 GB of KV for the same context.
//
// 3. Weights are mmap'd, so they are clean file-backed pages: Windows can drop them under pressure
//    and re-read them from disk. The KV cache is NOT - it is committed memory that can only go to
//    the pagefile. So the KV cache is the only part of the footprint that can actually freeze this
//    machine, and it is sized against a hard budget while the weights get a soft one.
import { readGGUF, archInfo } from "./gguf.mjs";
import { ram, cpu, gpu, GiB, MiB, fmtGB } from "./sys.mjs";

// Held back on top of the weights and KV we place. CUDA's context, cuBLAS workspaces and the graph
// allocator live here, and Windows grows the desktop's own VRAM use while we run. Too small and
// the driver spills into shared system memory, which on Windows looks like a frozen desktop rather
// than an error - so this is deliberately generous.
const VRAM_RESERVE = 768 * MiB;
const COMPUTE_PER_UBATCH = 1.4 * MiB;
// llama.cpp host buffers, the tokenizer and the HTTP server. Committed, unlike the mmap'd weights.
const RAM_OVERHEAD = 1.2 * GiB;

// bytes per KV element
const KV_ELEM = { f16: 2, bf16: 2, q8_0: 1.0625, q5_1: 0.75, q4_0: 0.5625 };
// Best quality first. The planner walks down this list until the KV cache fits the hard budget.
const KV_LADDER = ["f16", "q8_0", "q5_1", "q4_0"];
export const CTX_LADDER = [262144, 196608, 131072, 98304, 65536, 49152, 32768, 16384];

const layerOf = (n) => { const m = n.match(/^blk\.(\d+)\./); return m ? +m[1] : -1; };
const mmapWeightsFor = (placement, forcedCpuBytes) => Math.max(0, placement.weightsOnCpu - forcedCpuBytes);

// Everything about the file that does not depend on how much memory happens to be free.
export function inspect(modelPath) {
  const g = readGGUF(modelPath);
  const a = archInfo(g);
  const meta = (k, d) => g.meta[`${a.arch}.${k}`] ?? d;

  const perLayer = new Map();
  const attnLayers = new Set();
  let embd = 0, output = 0, other = 0;
  for (const t of g.tensors) {
    const l = layerOf(t.name);
    if (l < 0) {
      if (/^token_embd\./.test(t.name)) embd += t.bytes;
      else if (/^output\.weight$/.test(t.name)) output += t.bytes;
      else other += t.bytes;
      continue;
    }
    perLayer.set(l, (perLayer.get(l) ?? 0) + t.bytes);
    // A block carrying a key projection is a real attention layer and keeps a KV cache; recurrent
    // and short-convolution blocks carry ssm_*/shortconv_* instead and keep only a fixed-size
    // state. MLA models (DeepSeek, GLM) have no attn_k at all - they project into a single latent
    // cache through attn_kv_a_mqa - so matching attn_k alone reports them as having no KV
    // whatsoever, which is a 6 GB hole in the budget rather than a rounding error.
    if (/\.attn_(?:k|kv_a_mqa|k_b)\.weight$/.test(t.name)) attnLayers.add(l);
  }

  // The vocabulary size is not in this file's metadata, but token_embd is [n_embd, n_vocab] and it
  // drives the single largest host-side allocation llama.cpp makes (see runtimeBytes below).
  const embdTensor = g.tensors.find((t) => t.name === "token_embd.weight");
  const nVocab = embdTensor?.dims?.[1] ?? 0;

  const nBlocks = perLayer.size;
  const ssmInner = meta("ssm.inner_size", 0);
  const ssmState = meta("ssm.state_size", 0);
  const ssmConv = meta("ssm.conv_kernel", 0);

  // --- how much KV each individual layer keeps -------------------------------------------------
  // A single "n_head_kv x head_dim x 2" figure is only right for a plain dense transformer. Every
  // other family on this machine breaks it in a different direction, and each one is worth
  // gigabytes on a 32 GB box:
  //
  //   - LFM2 and Qwen3.8 are HYBRID: `head_count_kv` is a per-layer ARRAY and the zeros are
  //     conv/recurrent layers that keep no KV cache at all. Multiplying by a scalar here is how a
  //     16-layer model gets billed for 16 attention layers when it only has 6.
  //   - Gemma 3/4 are SLIDING-WINDOW: `sliding_window_pattern` marks layers whose cache never grows
  //     past `sliding_window` tokens (512 here) no matter how long the context is, and those layers
  //     use their own smaller `key_length_swa`. At 131072 tokens that is a 250x overestimate per
  //     SWA layer.
  //   - DeepSeek/GLM use MLA: one latent cache with `key_length` 576 and `value_length` 512 against
  //     a single KV head. Assuming k and v share a dimension gets it wrong in both directions.
  //
  // So KV is modelled per layer, from key_length and value_length separately, and clamped by each
  // layer's own attention window.
  const headCountKV = meta("attention.head_count_kv", null);
  const kvArr = Array.isArray(headCountKV) ? headCountKV : null;
  const kvScalar = kvArr ? null : (typeof headCountKV === "number" ? headCountKV : 1);
  const nHead = meta("attention.head_count", a.nHead || 1);
  const fallbackDim = nHead ? Math.round(a.nEmbd / nHead) : 128;
  const keyLen = meta("attention.key_length", fallbackDim);
  const valLen = meta("attention.value_length", keyLen);
  const swaPattern = meta("attention.sliding_window_pattern", null);
  const swaWindow = meta("attention.sliding_window", 0);
  const keyLenSwa = meta("attention.key_length_swa", keyLen);
  const valLenSwa = meta("attention.value_length_swa", valLen);
  // Gemma 3 declares a sliding window but no per-layer pattern, because its pattern is fixed in the
  // architecture rather than in the file: five local layers then one global, repeating. llama.cpp
  // hard-codes the same ratio. Without it a 4B model is billed 17 GB of KV at its native 131072
  // window and is written off as unloadable, when the real figure is under a gigabyte.
  //
  // Anywhere else, an unexplained sliding_window is left alone: overestimating KV only costs a
  // shorter context, while underestimating it is what makes a load die half-way through.
  const KNOWN_SWA_PERIOD = { gemma3: 6 };
  const period = KNOWN_SWA_PERIOD[a.arch];
  const swa = Array.isArray(swaPattern) && swaWindow > 0
    ? swaPattern
    : (period && swaWindow > 0
      ? Array.from({ length: nBlocks }, (_, l) => (l + 1) % period !== 0)
      : null);

  const kvLayers = [];
  for (const l of [...perLayer.keys()].sort((x, y) => x - y)) {
    const heads = kvArr ? (kvArr[l] ?? 0) : (attnLayers.has(l) ? kvScalar : 0);
    if (!heads) continue;
    const windowed = swa ? !!swa[l] : false;
    kvLayers.push({
      layer: l,
      elemsPerToken: heads * ((windowed ? keyLenSwa : keyLen) + (windowed ? valLenSwa : valLen)),
      window: windowed ? swaWindow : Infinity,
    });
  }

  const attn = kvLayers.map((k) => k.layer);
  const attnSet = new Set(attn);
  const ssm = [...perLayer.keys()].filter((l) => !attnSet.has(l)).sort((x, y) => x - y);

  const elemBytes = (kv) => KV_ELEM[kv] ?? 2;
  // bytes of KV cache each layer holds at this context length
  const kvPerLayer = (kv, ctx) => {
    const m = new Map();
    for (const k of kvLayers) m.set(k.layer, Math.round(k.elemsPerToken * elemBytes(kv) * Math.min(ctx, k.window)));
    return m;
  };
  const kvBytesAt = (kv, ctx) => {
    let s = 0;
    for (const k of kvLayers) s += k.elemsPerToken * elemBytes(kv) * Math.min(ctx, k.window);
    return Math.round(s);
  };

  return {
    path: modelPath,
    name: g.meta["general.name"] ?? a.arch,
    arch: a.arch,
    fileType: g.meta["general.file_type"],
    nBlocks,
    nextnLayers: meta("nextn_predict_layers", 0),
    attnLayers: attn, ssmLayers: ssm,
    nEmbd: a.nEmbd, nHead, nHeadKV: kvArr ?? kvScalar, keyLen, valLen, nVocab,
    swaWindow: swa ? swaWindow : 0,
    swaLayers: swa ? kvLayers.filter((k) => k.window !== Infinity).length : 0,
    nCtxTrain: a.nCtxTrain,
    hasMTP: a.hasMTP && meta("nextn_predict_layers", 0) > 0,
    perLayer,
    bytes: {
      total: g.tensors.reduce((s, t) => s + t.bytes, 0),
      embd, output, other,
      attnLayers: attn.reduce((s, l) => s + perLayer.get(l), 0),
      ssmLayers: ssm.reduce((s, l) => s + perLayer.get(l), 0),
    },
    // Fixed cost, independent of context length: conv window + recurrent state per recurrent layer.
    ssmStateBytes: ssm.length * (ssmInner * ssmConv + ssmInner * ssmState) * 4,
    kvPerLayer, kvBytesAt,
    // Average bytes per token across the whole cache. Only meaningful for display - with sliding
    // windows in play the cache does not actually grow linearly, so nothing budgets off this.
    kvPerToken: (kv, ctx = 4096) => kvBytesAt(kv, ctx) / ctx,
  };
}

// Place layers on the GPU under a VRAM budget.
//
// "layers"  - whole blocks from the top down. This is llama.cpp's native `-ngl N`, so the KV cache
//             for an offloaded layer lands in VRAM with it, and a token crosses the CPU/GPU
//             boundary exactly once. Correct everywhere, and the right default.
//
// "ssm-pack" - at very long context a full-attention layer drags a huge KV slice onto the card with
//             it (at 262144 that is ~280 MB per layer, against 168 MB for a whole DeltaNet layer's
//             weights), so VRAM buys far more model if it holds only DeltaNet layers and the KV
//             cache is left in RAM. Costs a boundary crossing per layer, and depends on llama.cpp
//             putting a layer's KV where `-ot` put its weights - which is why `claudelocal tune`
//             measures it instead of assuming it.
function place(m, { strategy, vramBudget, kvBytes, kvOf }) {
  // Per layer, not an average: a Gemma sliding-window layer carries a few MB of KV onto the card
  // while its full-attention neighbour carries hundreds, and spreading the total evenly across
  // them either wastes VRAM or overcommits it.
  const kvAt = kvOf ?? (() => (m.attnLayers.length ? kvBytes / m.attnLayers.length : 0));
  const gpu = new Set();
  let w = 0, kv = 0;

  if (strategy === "ssm-pack") {
    for (let i = m.ssmLayers.length - 1; i >= 0; i--) {
      const l = m.ssmLayers[i];
      const b = m.perLayer.get(l);
      if (w + b > vramBudget) break;
      w += b; gpu.add(l);
    }
  } else {
    for (let l = m.nBlocks - 1; l >= 0; l--) {
      const b = m.perLayer.get(l);
      const k = kvAt(l);
      if (w + kv + b + k > vramBudget) break;
      w += b; kv += k; gpu.add(l);
    }
  }

  const all = gpu.size === m.nBlocks;
  // The output head is only worth VRAM once every layer is already there; token_embd is a row
  // lookup rather than a matmul, so it never is.
  const outputOnGpu = all && w + kv + m.bytes.output <= vramBudget;
  if (outputOnGpu) w += m.bytes.output;

  const cpuLayers = [...m.perLayer.keys()].filter((l) => !gpu.has(l)).sort((x, y) => x - y);
  return {
    strategy, gpuLayers: gpu, cpuLayers, nGpuLayers: gpu.size, all, outputOnGpu,
    weightsOnGpu: w, kvOnGpu: kv, kvOnCpu: kvBytes - kv,
    weightsOnCpu: m.bytes.total - w,
  };
}

export function plan(modelOrPath, opts = {}) {
  const m = typeof modelOrPath === "string" ? inspect(modelOrPath) : modelOrPath;
  const c = cpu();
  const r = ram();
  const v = gpu();

  // ssm-pack only means anything on a hybrid model - on a plain transformer there are no recurrent
  // layers to pack, so it would put nothing on the card at all.
  const strategy = (opts.strategy === "ssm-pack" && m.ssmLayers.length) ? "ssm-pack" : "layers";
  const ubatch = opts.ubatch ?? 256;
  const batch = opts.batch ?? 1024;
  const ramMargin = (opts.ramMarginGB ?? 2) * GiB;
  const vramReserve = opts.vramReserve ?? VRAM_RESERVE;
  const computeBytes = Math.round(ubatch * COMPUTE_PER_UBATCH);
  const vramFree = v ? Math.max(0, v.free) : 0;
  // Default to the model's full native context. That is the whole point of this setup: the user
  // asked for all 27B parameters at the full 262144 window, and the hybrid attention makes it
  // reachable as long as the KV cache is quantized to suit.
  const ctx = Math.min(opts.ctx ?? m.nCtxTrain, m.nCtxTrain);

  // --- pick the KV cache type -----------------------------------------------------------------
  // The KV cache competes with the weights for the same RAM, so choosing it on the committed
  // budget alone is a trap: at 262144 tokens an f16 cache technically "fits" in free RAM, and then
  // there is no page cache left and 11 GB of weights get re-read from disk on every single token.
  // So prefer the best-quality cache that still leaves the whole CPU-side of the model resident,
  // and only if none does, fall back to the smallest cache that at least clears the hard budget.
  const vramBudget = Math.max(0, vramFree - vramReserve - computeBytes - m.ssmStateBytes);
  const hardBudget = Math.max(0, r.free - ramMargin - RAM_OVERHEAD);
  const consider = (opts.kvType && opts.kvType !== "auto") ? [opts.kvType] : KV_LADDER;

  const candidates = consider.map((t) => {
    const kvBytes = m.kvBytesAt(t, ctx);
    const perLayer = m.kvPerLayer(t, ctx);
    const cand = place(m, { strategy, vramBudget, kvBytes, kvOf: (l) => perLayer.get(l) ?? 0 });
    // Speculative decoding builds a second context against the same model at the same n_ctx, so
    // when MTP is on the KV cache is paid for twice. Ignoring that here is what let the picker
    // choose q8_0 and then hit the guard's commit cap part-way through the load.
    const kvCommitted = cand.kvOnCpu * (m.hasMTP ? 2 : 1);
    return {
      type: t, kvBytes, ...cand, kvCommitted,
      // What has to live in RAM: the committed caches plus every weight not on the card.
      ramWanted: kvCommitted + RAM_OVERHEAD + cand.weightsOnCpu,
      fitsHard: kvCommitted + RAM_OVERHEAD + ramMargin <= r.free,
    };
  });

  const p0 =
    // best quality that leaves everything resident
    candidates.find((k) => k.ramWanted + ramMargin <= r.free) ??
    // else the smallest cache that at least keeps committed memory safe
    [...candidates].reverse().find((k) => k.fitsHard) ??
    candidates[candidates.length - 1];

  const kvType = p0.type;
  const kvBytes = p0.kvBytes;
  const placement = p0;

  // --- RAM accounting -------------------------------------------------------------------------
  // hard: committed memory (KV in RAM + llama.cpp host buffers). Overrunning this is what makes
  //       Windows page out live allocations and lock up the desktop, so it is a hard failure.
  // soft: mmap'd weights. Overrunning this only means Windows evicts model pages and re-reads them
  //       from disk - slower per token, never a freeze. Worth a warning, not a refusal.
  // Speculative decoding builds a second context against the same model at the same n_ctx, so it
  // brings a second KV cache of its own. Budgeting for it here is what keeps the guard's commit cap
  // from cutting the load off part-way through.
  // The MTP draft context is built against the same model at the same n_ctx and gets no GPU layers
  // of its own, so it commits a WHOLE second KV cache in RAM - not just the share the main context
  // left on the CPU. Budgeting it as the CPU share understated the real footprint by over a
  // gigabyte at 262144 tokens.
  const draftKv = m.hasMTP ? kvBytes : 0;

  // Tensors forced to the CPU by -ot are not mmap'd: llama.cpp copies them into a committed host
  // buffer, which is what its own "tensor overrides to CPU are used with mmap enabled" warning is
  // about. token_embd and (unless it is on the card) output.weight are 1.64 GB here, and counting
  // them as evictable mmap'd pages both understated committed memory and overstated how much of
  // the model Windows can drop under pressure.
  const forcedCpuBytes = m.bytes.embd + (placement.outputOnGpu ? 0 : m.bytes.output);

  // llama.cpp's host-side output buffer is n_vocab * n_batch * 4 bytes. Qwen3.8's vocabulary is
  // 248320 entries, so at -b 1024 that one allocation is a full gigabyte - larger than everything
  // else in "runtime overhead" put together, and the main reason a flat 1.2 GB estimate came out
  // ~1.5 GB under what the guard actually measured. It scales with --batch, so compute it.
  const logitsBytes = m.nVocab * batch * 4;
  const runtimeBytes = RAM_OVERHEAD + logitsBytes;

  const baseCommitted = placement.kvOnCpu + draftKv + forcedCpuBytes + runtimeBytes;

  // llama-server keeps a RAM cache of previous prompts' KV state so a conversation that comes back
  // to a slot is restored instead of reprocessed. Two things about it matter here:
  //
  //   - It defaults to 8192 MiB of COMMITTED memory, and nothing above was budgeting for it. On a
  //     32 GB machine already holding a doubled 262144-token KV cache that is enough on its own to
  //     walk into the guard's commit cap part-way through a run, which looks like llama-server
  //     dying for no reason the moment a request finishes.
  //   - It is also the most valuable buffer in this whole setup. Claude Code interleaves small
  //     side-requests with the main conversation, and with -np 1 any one of them evicts the slot.
  //     Rebuilding a 25k-token prefix costs minutes at this machine's prompt-processing rate, so a
  //     cache that can hold the conversation across an interruption is worth more than it costs.
  //
  // So it is sized explicitly from what is actually left over once the committed caches are paid
  // for AND every CPU-side weight still fits in the page cache - never at the expense of either.
  const cacheSpare = r.free - baseCommitted - ramMargin - mmapWeightsFor(placement, forcedCpuBytes);
  const cacheWant = Math.round((m.kvBytesAt(kvType, Math.min(ctx, 65536)) + m.ssmStateBytes) * 2);
  const cacheRam = Math.max(0, Math.min(cacheWant, Math.floor(cacheSpare)));

  const committed = baseCommitted + cacheRam;
  const committedHeadroom = r.free - committed - ramMargin;
  const pageCacheAfter = r.free - committed;
  // Only the genuinely mmap'd weights belong in the soft budget - the -ot-forced ones are already
  // paid for above as committed memory.
  const mmapWeights = Math.max(0, placement.weightsOnCpu - forcedCpuBytes);
  const weightsResident = Math.min(mmapWeights, Math.max(0, pageCacheAfter));
  const weightsStreamed = mmapWeights - weightsResident;

  const threads = Math.max(2, Math.min(c.physical - 2, 8));
  const threadsBatch = Math.max(threads, Math.min(c.physical, 8));

  const warnings = [];
  if (!v) warnings.push("no NVIDIA GPU visible - this will run CPU-only and be very slow");
  if (committedHeadroom <= 0) {
    warnings.push(`the KV cache for ${ctx} tokens needs ${fmtGB(placement.kvOnCpu)} of committed RAM and only ${fmtGB(r.free)} is free (margin ${fmtGB(ramMargin)}). Lower --ctx or free some RAM.`);
  }
  if (weightsStreamed > 64 * MiB) {
    warnings.push(`${fmtGB(weightsStreamed)} of weights will not stay in the page cache and will be re-read from disk each token - safe, but expect it to be slow. Closing other apps or lowering --ctx buys it back.`);
  }
  if (v && placement.nGpuLayers === 0) warnings.push("nothing fits in VRAM at this context length");

  const est = {
    totalWeights: m.bytes.total,
    onGpu: placement.weightsOnGpu, onCpuWeights: placement.weightsOnCpu,
    kvBytes, kvOnGpu: placement.kvOnGpu, kvOnCpu: placement.kvOnCpu,
    ssmStateBytes: m.ssmStateBytes, computeBytes, vramReserve,
    vramFree, vramUsed: placement.weightsOnGpu + placement.kvOnGpu + vramReserve + computeBytes + m.ssmStateBytes,
    ramFree: r.free, ramMargin,
    committed, committedHeadroom, weightsResident, weightsStreamed, draftKv, cacheRam, cacheWant,
    forcedCpuBytes, mmapWeights, logitsBytes, runtimeBytes,
    ramTotalTouched: committed + mmapWeights,
  };

  return {
    model: m, strategy, ctx, kvType, ubatch, batch, threads, threadsBatch,
    ngl: placement.all ? 999 : (strategy === "ssm-pack" ? 999 : placement.nGpuLayers),
    nGpuLayers: placement.nGpuLayers,
    cpuLayers: placement.cpuLayers,
    outputOnGpu: placement.outputOnGpu,
    // token_embd is a row lookup, not a matmul, so VRAM buys it almost nothing - keeping it in RAM
    // frees ~0.67 GB, which is three or four more transformer layers on the card.
    cpuTensors: buildOverrides(m, placement, strategy),
    mtp: m.hasMTP,
    est,
    ok: committedHeadroom > 0 && !!v,
    warnings,
  };
}

function buildOverrides(m, placement, strategy) {
  const pats = ["token_embd\\..*"];
  if (!placement.outputOnGpu) pats.push("output\\.weight");
  // Under "layers", `-ngl N` already expresses the split and no per-tensor override is needed.
  if (strategy === "ssm-pack" && placement.cpuLayers.length) {
    pats.push(`blk\\.(${placement.cpuLayers.join("|")})\\..*`);
  }
  return pats;
}

// Turn a plan into the llama-server command line.
export function serverArgs(p, { port, mmproj, alias = "local" }) {
  const args = [
    "-m", p.model.path,
    "--host", "127.0.0.1", "--port", String(port),
    "-c", String(p.ctx),
    "-b", String(p.batch), "-ub", String(p.ubatch),
    "-t", String(p.threads), "-tb", String(p.threadsBatch),
    "-ngl", String(p.ngl),
    // Flash attention is what makes a quantized KV cache legal, and it removes the attention
    // scratch buffer that would otherwise eat the VRAM reserve at long context.
    "-fa", "on",
    "-ctk", p.kvType, "-ctv", p.kvType,
    // mmap, never mlock. Memory-mapped weights are clean file-backed pages, so under memory
    // pressure Windows drops them and re-reads from disk instead of writing them to the pagefile.
    // That is the difference between "gets slower" and "the desktop stops responding", and it is
    // the single most important flag in this list.
    "--load-mode", "mmap",
    // Generation threads run below normal priority so the desktop always wins a scheduling fight.
    "--prio", "-1",
    // One slot keeps the whole context in a single KV cache and is required for MTP.
    "-np", "1",
    "--jinja",
    "--no-webui",
    "--alias", alias,
    // Sized by the planner rather than left at its 8192 MiB default - see the note in plan().
    "--cache-ram", String(Math.floor(p.est.cacheRam / MiB)),
    // Claude Code edits the MIDDLE of its prompt constantly: system-reminders appear and vanish,
    // tool results get trimmed, the todo list is rewritten in place. Plain prefix matching gives up
    // at the first changed byte and reprocesses everything after it, which at this machine's
    // prompt-processing rate is minutes per turn. --cache-reuse lets the server KV-shift the
    // unchanged chunks after the edit back into place instead of recomputing them.
    "--cache-reuse", "256",
  ];
  // One -ot flag with comma-separated patterns. Repeating the flag looks like it should work and
  // does not: llama.cpp warns "specified multiple times ... only last value will be used", so
  // every override but the last is silently dropped.
  if (p.cpuTensors.length) args.push("-ot", p.cpuTensors.map((pat) => `${pat}=CPU`).join(","));
  // The vision projector is ~0.9 GB in F16 and would otherwise land in VRAM, where it is worth far
  // less than the three transformer layers it displaces - it only runs when an image is actually
  // attached. Keeping it on the CPU leaves vision available at no cost to text generation.
  if (mmproj) args.push("--mmproj", mmproj, "--no-mmproj-offload");
  // The MTP head ships inside the GGUF as blk.<last>.nextn.*. Speculative decoding is worth more
  // here than on a big card: a verification pass costs one sweep over ~12 GB of RAM whether it
  // confirms one token or three, so every accepted draft token is close to free.
  if (p.mtp) {
    args.push("--spec-type", "draft-mtp", "--spec-draft-n-max", "2");
    // The MTP draft context is built against the same model and sized at the same n_ctx, so it
    // allocates a SECOND KV cache - and its cache type defaults to f16 regardless of what the main
    // one uses. At 262144 tokens that difference is about 12 GB of committed memory, so the draft
    // cache is pinned to the same quantization as the main one.
    args.push("-ctkd", p.kvType, "-ctvd", p.kvType);
  }
  return args;
}

export function explain(p) {
  const m = p.model, e = p.est;
  const pct = (b) => `${((b / e.totalWeights) * 100).toFixed(0)}%`;
  const L = [];
  const mix = m.ssmLayers.length
    ? `${m.attnLayers.length} attention + ${m.ssmLayers.length} recurrent/conv`
    : `${m.attnLayers.length} attention`;
  const swa = m.swaLayers ? `, ${m.swaLayers} of them windowed at ${m.swaWindow}` : "";
  L.push(`model      ${m.name}   ${m.arch}, ${m.nBlocks} blocks (${mix}${swa}), ${fmtGB(e.totalWeights)} of weights`);
  L.push(`context    ${p.ctx} tokens (native max ${m.nCtxTrain})   KV ${p.kvType} = ${fmtGB(e.kvBytes)} at ${(e.kvBytes / p.ctx / 1024).toFixed(1)} KB/token`);
  L.push("");
  L.push(`VRAM       ${p.nGpuLayers} of ${m.nBlocks} layers on the GPU  ->  ${fmtGB(e.onGpu)} weights (${pct(e.onGpu)})`);
  L.push(`           + ${fmtGB(e.kvOnGpu)} KV + ${fmtGB(e.ssmStateBytes + e.computeBytes)} recurrent state/compute + ${fmtGB(e.vramReserve)} reserved`);
  L.push(`           = ${fmtGB(e.vramUsed)} of ${fmtGB(e.vramFree)} free  [strategy: ${p.strategy}]`);
  L.push("");
  L.push(`RAM  hard  ${fmtGB(e.kvOnCpu)} KV${e.draftKv ? ` + ${fmtGB(e.draftKv)} MTP draft KV` : ""} + ${fmtGB(e.forcedCpuBytes)} pinned tensors + ${fmtGB(e.runtimeBytes)} runtime${e.cacheRam ? ` + ${fmtGB(e.cacheRam)} prompt cache` : ""} = ${fmtGB(e.committed)} committed`);
  L.push(`           ${fmtGB(e.ramFree)} free, ${fmtGB(e.committedHeadroom)} spare after a ${fmtGB(e.ramMargin)} margin`);
  L.push(`RAM  soft  ${fmtGB(e.mmapWeights)} of weights (${pct(e.mmapWeights)}) mmap'd - evictable, never paged out`);
  L.push(`           ${fmtGB(e.weightsResident)} stays resident, ${fmtGB(e.weightsStreamed)} re-read from disk`);
  L.push("");
  L.push(`CPU        ${p.threads} generation / ${p.threadsBatch} batch threads at below-normal priority (${cpu().physical} cores, ${cpu().logical} threads)`);
  L.push(`extras     ${p.mtp ? "MTP speculative decoding on" : "no MTP head"}${p.outputOnGpu ? ", output head on GPU" : ", output head + embeddings in RAM"}`);
  L.push(`cache      ${e.cacheRam
    ? `${fmtGB(e.cacheRam)} prompt cache (of ${fmtGB(e.cacheWant)} wanted) - survives a side-request evicting the slot`
    : "no prompt cache - nothing left after the KV cache, so an evicted conversation is reprocessed from scratch"}`);
  for (const w of p.warnings) L.push(`WARNING    ${w}`);
  return L.join("\n");
}

// What the other context lengths would cost. Printed by `claudelocal plan` so the trade is visible.
export function ladder(m, opts = {}) {
  return CTX_LADDER.filter((ctx) => ctx <= m.nCtxTrain).map((ctx) => {
    const p = plan(m, { ...opts, ctx });
    return {
      ctx, kvType: p.kvType, ngl: p.nGpuLayers,
      gpuWeights: p.est.onGpu, kv: p.est.kvBytes,
      committed: p.est.committed, streamed: p.est.weightsStreamed,
      cacheRam: p.est.cacheRam, ok: p.ok,
    };
  });
}
