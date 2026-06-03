// Persistence of the post-login "return to" path across the OAuth redirect round-trip.
//
// The auth -> IdP -> callback flow is a sequence of full-page navigations, so React Router's
// in-memory `location.state.path` is lost. We stash the intended return path in sessionStorage
// (same-origin, survives navigations) and restore it on the callback page.

/// sessionStorage key holding the sanitized post-OAuth return path.
export const OAUTH_RETURN_KEY = 'THORIUM_OAUTH_RETURN';

/**
 * Sanitize a candidate return path to a safe same-origin relative path.
 *
 * Rejects anything that could drive an open redirect: absolute URLs, protocol-relative
 * (`//host`) paths, and backslash variants browsers may normalize to `//`. Falls back to `/`.
 *
 * @param path - The candidate path (e.g. from router state or sessionStorage).
 * @returns A path guaranteed to start with a single `/`, or `/`.
 */
export function sanitizeReturnPath(path: string | null | undefined): string {
  if (!path || !path.startsWith('/') || path.startsWith('//') || path.startsWith('/\\')) {
    return '/';
  }
  return path;
}

/**
 * Stash the (sanitized) return path before navigating to the external IdP.
 *
 * @param path - The path to return to after sign-in completes.
 */
export function stashOAuthReturn(path: string | null | undefined): void {
  if (typeof window === 'undefined') return;
  window.sessionStorage.setItem(OAUTH_RETURN_KEY, sanitizeReturnPath(path));
}

/**
 * Read and clear the stashed return path on the callback page.
 *
 * @returns The sanitized return path, or `/` if none was stashed.
 */
export function consumeOAuthReturn(): string {
  if (typeof window === 'undefined') return '/';
  const raw = window.sessionStorage.getItem(OAUTH_RETURN_KEY);
  window.sessionStorage.removeItem(OAUTH_RETURN_KEY);
  return sanitizeReturnPath(raw);
}
