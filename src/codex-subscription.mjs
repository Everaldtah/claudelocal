// The Codex subscription, as a model source.
//
// Codex uses OAuth (chatgpt auth mode) with tokens stored in ~/.codex/auth.json.
// The access token is read from that store at call time and never cached.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const CODEX_SUBSCRIPTION_AUTH = "codex-subscription";
export const CODEX_API = "https://chatgpt.com/backend-api/codex";

const CREDENTIALS = path.join(process.env.CODEX_HOME || path.join(os.homedir(), ".codex"), "auth.json");
export const CREDENTIALS_FILE = CREDENTIALS;

/** True when this entry authenticates with the Codex subscription rather than with a key in the file. */
export const isCodexSubscription = (r) => r?.auth === CODEX_SUBSCRIPTION_AUTH || r?.remote?.auth === CODEX_SUBSCRIPTION_AUTH;

/**
 * The current Codex subscription token, or a reason there is none.
 * Never throws: a missing or malformed credential store must not stop the local models loading.
 *
 * @returns {{token: string|null, reason: string|null, expiresAt: number|null, plan: string|null}}
 */
export function codexSubscriptionToken() {
  const none = (reason) => ({ token: null, reason, expiresAt: null, plan: null });

  if (!fs.existsSync(CREDENTIALS)) {
    return none(`no credential store at ${CREDENTIALS} - run 'codex' once and sign in`);
  }
  let cred;
  try {
    cred = JSON.parse(fs.readFileSync(CREDENTIALS, "utf8").replace(/^\uFEFF/, ""));
  } catch {
    return none(`${CREDENTIALS} is not valid JSON`);
  }
  if (cred.auth_mode && cred.auth_mode !== "chatgpt") return none("Codex is not using ChatGPT authentication - run 'codex login'");
  const tokens = cred?.tokens;
  if (!tokens || typeof tokens !== "object") return none("credential store has no Codex token object - sign in with 'codex'");
  if (!tokens.access_token || typeof tokens.access_token !== "string") return none("subscription session has no access token - sign in again with 'codex'");

  let claims;
  try { claims = JSON.parse(Buffer.from(tokens.access_token.split('.')[1], 'base64url').toString()); }
  catch { return none("invalid Codex access token - run 'codex login'"); }
  const auth = claims['https://api.openai.com/auth'] ?? {};
  const expiresAt = Number(claims.exp) * 1000;
  const plan = auth.chatgpt_plan_type ?? null;
  const accountId = tokens.account_id ?? auth.chatgpt_account_id;
  if (!accountId) return none("no ChatGPT account in Codex credentials - run 'codex login'");
  if (!Number.isFinite(expiresAt) || expiresAt < Date.now() + 60000) {
    return { ...none("Codex session expired - open Codex to refresh it, or run 'codex login'"), expiresAt, plan };
  }
  return { token: tokens.access_token, accountId, reason: null, expiresAt, plan };
}

/** Human-readable state of the Codex subscription, for `claudelocal remotes` and the doctor. */
export function codexSubscriptionStatus() {
  const s = codexSubscriptionToken();
  if (!s.token) return { ok: false, detail: s.reason, plan: s.plan };
  return { ok: true, detail: `ChatGPT ${s.plan ?? "subscription"}; expires ${new Date(s.expiresAt).toISOString()}`, plan: s.plan };
}

export function codexHeaders() {
  const s = codexSubscriptionToken();
  if (!s.token) throw new Error(s.reason);
  return { authorization: `Bearer ${s.token}`, 'ChatGPT-Account-Id': s.accountId,
    'content-type': 'application/json', accept: 'text/event-stream', originator: 'claudelocal' };
}

export async function codexModels() {
  const response = await fetch(`${CODEX_API}/models?client_version=0.165.0`, {
    headers: codexHeaders(), redirect: 'error', signal: AbortSignal.timeout(30000),
  });
  if (!response.ok) throw new Error(`Codex model discovery failed (HTTP ${response.status}); check 'codex login status'`);
  const data = await response.json();
  if (!Array.isArray(data.models)) throw new Error('Codex returned no model catalog');
  return data.models.filter(m => m.visibility === 'list');
}
