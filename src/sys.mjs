// Live machine facts. Everything the planner decides is derived from these, never hard-coded,
// so the same profile behaves sanely whether the desktop is idle or Overwatch is running.
import os from "node:os";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export const GiB = 1024 ** 3;
export const MiB = 1024 ** 2;
export const ROOT = path.resolve(new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));

export const fmtGB = (b) => `${(b / GiB).toFixed(2)} GB`;

export function ram() {
  return { total: os.totalmem(), free: os.freemem(), used: os.totalmem() - os.freemem() };
}

let cpuCache = null;
export function cpu() {
  if (cpuCache) return cpuCache;
  const logical = os.cpus().length;
  let physical = Math.max(1, logical >> 1);
  try {
    const out = execFileSync("powershell", ["-NoLogo", "-NoProfile", "-Command",
      "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
      { encoding: "utf8", timeout: 15000 }).trim();
    if (/^\d+$/.test(out)) physical = parseInt(out, 10);
  } catch { /* keep the halved-logical guess */ }
  cpuCache = { logical, physical, model: os.cpus()[0]?.model?.trim() ?? "unknown" };
  return cpuCache;
}

// nvidia-smi is the only source that reports what the DESKTOP is already holding. Planning against
// `total` instead of `free` is exactly how you end up spilling into shared system memory, which on
// Windows looks like a hang rather than an error.
export function gpu() {
  try {
    const out = execFileSync("nvidia-smi",
      ["--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
      { encoding: "utf8", timeout: 15000 }).trim();
    if (!out) return null;
    const [name, total, used, free] = out.split("\n")[0].split(",").map((s) => s.trim());
    return { name, total: +total * MiB, used: +used * MiB, free: +free * MiB };
  } catch { return null; }
}

export function diskFree(dir) {
  try {
    const drive = path.resolve(dir).slice(0, 2); // "C:"
    const out = execFileSync("powershell", ["-NoLogo", "-NoProfile", "-Command",
      `(Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${drive}'").FreeSpace`],
      { encoding: "utf8", timeout: 15000 }).trim();
    return /^\d+$/.test(out) ? Number(out) : null;
  } catch { return null; }
}

export const paths = {
  root: ROOT,
  models: path.join(ROOT, "models"),
  runtime: path.join(ROOT, "runtime"),
  logs: path.join(ROOT, "logs"),
  state: path.join(ROOT, "state.json"),
};

// Windows PowerShell 5.1's `-Encoding utf8` means utf8-WITH-BOM, and JSON.parse throws on a
// leading U+FEFF. guard.ps1 writes guard.json, so every reader here has to tolerate one - a silent
// parse failure there costs the caller its only signal that llama-server died during a load.
export function readJSON(file, fallback = null) {
  try { return JSON.parse(fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "")); }
  catch { return fallback; }
}

export function readState() {
  return readJSON(paths.state, {}) ?? {};
}
export function writeState(patch) {
  const next = { ...readState(), ...patch };
  fs.mkdirSync(path.dirname(paths.state), { recursive: true });
  fs.writeFileSync(paths.state, JSON.stringify(next, null, 2));
  return next;
}

export async function httpJSON(url, { method = "GET", body, timeout = 5000, headers } = {}) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeout);
  try {
    const r = await fetch(url, { method, body, signal: ac.signal, headers: { "content-type": "application/json", ...headers } });
    return { ok: r.ok, status: r.status, json: await r.json().catch(() => null) };
  } catch (e) { return { ok: false, status: 0, error: e.message };
  } finally { clearTimeout(t); }
}

export async function portAlive(port, pathname = "/health", timeout = 1500) {
  const r = await httpJSON(`http://127.0.0.1:${port}${pathname}`, { timeout });
  return r.ok;
}
