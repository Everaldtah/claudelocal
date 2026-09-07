// Minimal GGUF reader: metadata + the tensor table with exact byte sizes.
//
// The planner needs real per-tensor byte counts, not estimates, because on a 6 GB card the
// difference between "fits" and "the driver spills into shared system memory" is a few hundred MB.
// Only the header is read (a few hundred KB), so this also works on a file that is still
// downloading - the header is written first.
import fs from "node:fs";

// id -> [blockSize, bytesPerBlock]
const GGML = {
  0: [1, 4], 1: [1, 2], 2: [32, 18], 3: [32, 20], 6: [32, 22], 7: [32, 24],
  8: [32, 34], 9: [32, 36], 10: [256, 84], 11: [256, 110], 12: [256, 144],
  13: [256, 176], 14: [256, 210], 15: [256, 292], 16: [256, 66], 17: [256, 74],
  18: [256, 98], 19: [256, 50], 20: [32, 18], 21: [256, 110], 22: [256, 82],
  23: [256, 136], 24: [1, 1], 25: [1, 2], 26: [1, 4], 27: [1, 8], 28: [1, 8],
  29: [256, 56], 30: [1, 2],
  // Ternary and micro-scaling formats. MXFP4 (39) is what OpenAI ships gpt-oss in, and without it
  // the whole file is unreadable rather than merely mis-sized - a block is one 8-bit shared
  // exponent plus 32 4-bit elements, so 1 + 16 bytes.
  34: [256, 54], 35: [256, 66], 39: [32, 17],
};
const TYPE_NAME = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
  10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS",
  17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
  22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16",
  34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
};

class Cur {
  constructor(fd) { this.fd = fd; this.pos = 0; this.buf = Buffer.alloc(0); this.base = 0; }
  // Keep a sliding window over the file so a 15 GB model costs a couple of reads.
  need(n) {
    const end = this.pos + n;
    if (this.pos >= this.base && end <= this.base + this.buf.length) return;
    const size = Math.max(1 << 22, n);
    const buf = Buffer.alloc(size);
    const got = fs.readSync(this.fd, buf, 0, size, this.pos);
    if (got < n) throw new Error("gguf: truncated header (file still downloading?)");
    this.buf = buf.subarray(0, got);
    this.base = this.pos;
  }
  take(n) { this.need(n); const off = this.pos - this.base; this.pos += n; return this.buf.subarray(off, off + n); }
  u8() { return this.take(1).readUInt8(0); }
  i8() { return this.take(1).readInt8(0); }
  u16() { return this.take(2).readUInt16LE(0); }
  i16() { return this.take(2).readInt16LE(0); }
  u32() { return this.take(4).readUInt32LE(0); }
  i32() { return this.take(4).readInt32LE(0); }
  f32() { return this.take(4).readFloatLE(0); }
  f64() { return this.take(8).readDoubleLE(0); }
  u64() { return Number(this.take(8).readBigUInt64LE(0)); }
  i64() { return Number(this.take(8).readBigInt64LE(0)); }
  str() { const n = this.u64(); return this.take(n).toString("utf8"); }
  val(t) {
    switch (t) {
      case 0: return this.u8();  case 1: return this.i8();
      case 2: return this.u16(); case 3: return this.i16();
      case 4: return this.u32(); case 5: return this.i32();
      case 6: return this.f32(); case 7: return this.u8() !== 0;
      case 8: return this.str();
      case 9: { const et = this.u32(), n = this.u64(); const out = [];
                // Long arrays are token vocabularies - the planner never needs them, so skip the
                // contents and keep only the length. Reading 151k strings costs seconds.
                const keep = n <= 4096;
                for (let i = 0; i < n; i++) { const v = this.val(et); if (keep) out.push(v); }
                return keep ? out : { skipped: n }; }
      case 10: return this.u64(); case 11: return this.i64(); case 12: return this.f64();
      default: throw new Error(`gguf: unknown value type ${t}`);
    }
  }
}

export function tensorBytes(type, dims) {
  const spec = GGML[type];
  if (!spec) throw new Error(`gguf: unknown tensor type ${type}`);
  const [block, size] = spec;
  const n = dims.reduce((a, b) => a * b, 1);
  return Math.ceil(n / block) * size;
}

export function readGGUF(path) {
  const fd = fs.openSync(path, "r");
  try {
    const c = new Cur(fd);
    if (c.take(4).toString("ascii") !== "GGUF") throw new Error("gguf: bad magic");
    const version = c.u32();
    const nTensors = c.u64();
    const nKV = c.u64();
    const meta = {};
    for (let i = 0; i < nKV; i++) { const k = c.str(); meta[k] = c.val(c.u32()); }
    const tensors = [];
    for (let i = 0; i < nTensors; i++) {
      const name = c.str();
      const nd = c.u32();
      const dims = []; for (let d = 0; d < nd; d++) dims.push(c.u64());
      const type = c.u32();
      c.u64(); // data offset - not needed
      tensors.push({ name, dims, type, typeName: TYPE_NAME[type] ?? `t${type}`, bytes: tensorBytes(type, dims) });
    }
    return { version, meta, tensors, headerBytes: c.pos, fileBytes: fs.fstatSync(fd).size };
  } finally { fs.closeSync(fd); }
}

// Pull the architecture numbers the planner needs, whatever the arch prefix is called.
export function archInfo(g) {
  const arch = g.meta["general.architecture"] ?? "unknown";
  const k = (suffix, dflt) => g.meta[`${arch}.${suffix}`] ?? dflt;
  return {
    arch,
    name: g.meta["general.name"] ?? arch,
    nLayer: k("block_count", 0),
    nEmbd: k("embedding_length", 0),
    nHead: k("attention.head_count", 0),
    nHeadKV: k("attention.head_count_kv", 0),
    nCtxTrain: k("context_length", 0),
    // Present on hybrid models: these mark which layers keep a real KV cache.
    recurrent: k("recurrent_layer_arr", null),
    hasMTP: g.tensors.some((t) => /\bnextn\b/.test(t.name)),
  };
}
