// The Claude subscription, as a model source.
//
// Every other entry in remotes.json authenticates with a key that sits in the file. The Claude
// subscription has no such key: Claude Code signs in over OAuth and keeps a short-lived access
// token, a refresh token and an expiry in its own credential store. So a subscription remote
// carries no secret at all - it names `"auth": "claude-subscription"` and the token is read from
// that store at call time.
//
// Read at call time, never cached, and never copied into remotes.json, for three reasons:
//   * the access token rotates - Claude Code refreshes it in the background, and a value captured
//     at startup would be stale within the hour;
//   * remotes.json is a config file a user may paste into an issue;
//   * a session that has been logged out should start failing immediately, not keep working off a
//     token this process happened to be holding.
//
// Nothing here refreshes the token. Refreshing means spending the refresh token, and if this
// process raced Claude Code's own refresh one of the two would be left holding an invalidated
// credential - the failure mode is being logged out of the editor, which is far worse than the
// error message this returns instead. `claude` refreshes on its own next launch.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const SUBSCRIPTION_AUTH = "claude-subscription";
export const ANTHROPIC_API = "https://api.anthropic.com";

// The beta header Anthropic's API requires when the caller presents a subscription OAuth token
// rather than an API key. Claude Code sends it on every request; a request that reached the proxy
// without one still has to be valid upstream.
export const OAUTH_BETA = "oauth-2025-04-20";

const CREDENTIALS = path.join(os.homedir(), ".claude", ".credentials.json");
export const CREDENTIALS_FILE = CREDENTIALS;

/** True when this entry authenticates with the subscription rather than with a key in the file. */
export const isSubscription = (r) => r?.auth === SUBSCRIPTION_AUTH || r?.remote?.auth === SUBSCRIPTION_AUTH;

/**
 * The current subscription token, or a reason there is none.
 * Never throws: a missing or malformed credential store must not stop the local models loading.
 *
 * @returns {{token: string|null, reason: string|null, expiresAt: number|null, plan: string|null}}
 */
export function subscriptionToken() {
  const none = (reason) => ({ token: null, reason, expiresAt: null, plan: null });

  if (!fs.existsSync(CREDENTIALS)) {
    // macOS keeps these in the Keychain instead of on disk, so "no file" is not always "logged
    // out" - say where it looked rather than guessing at the cause.
    return none(`no credential store at ${CREDENTIALS} - run 'claude' once and sign in`);
  }
  let cred;
  try {
    cred = JSON.parse(fs.readFileSync(CREDENTIALS, "utf8"));
  } catch {
    return none(`${CREDENTIALS} is not valid JSON`);
  }
  const o = cred?.claudeAiOauth;
  if (!o || typeof o !== "object") return none("credential store has no Claude subscription session - sign in with 'claude'");
  if (!o.accessToken || typeof o.accessToken !== "string") return none("subscription session has no access token - sign in again with 'claude'");

  const expiresAt = typeof o.expiresAt === "number" ? o.expiresAt : null;
  const plan = typeof o.subscriptionType === "string" ? o.subscriptionType : null;

  // A minute of slack: a token that expires while the request is in flight is as useless as one
  // that expired a minute ago, and the error is clearer before the send than after a 401.
  if (expiresAt && expiresAt < Date.now() + 60_000) {
    return {
      token: null, expiresAt, plan,
      reason: `subscription token expired ${new Date(expiresAt).toISOString()} - run 'claude' once to refresh it`,
    };
  }
  return { token: o.accessToken, reason: null, expiresAt, plan };
}

/** Human-readable state of the subscription, for `claudelocal remotes` and the doctor. */
export function subscriptionStatus() {
  const s = subscriptionToken();
  if (!s.token) return { ok: false, detail: s.reason, plan: s.plan };
  const mins = s.expiresAt ? Math.round((s.expiresAt - Date.now()) / 60000) : null;
  const left = mins === null ? "no expiry recorded"
             : mins > 120 ? `${Math.round(mins / 60)} h left`
             : `${mins} min left`;
  return { ok: true, detail: `${s.plan ?? "subscription"}, ${left}`, plan: s.plan };
}
