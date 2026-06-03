import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// project imports
import { buildCookie, buildRevokeCookie } from './auth';

// buildCookie/buildRevokeCookie read location.hostname at call time; the node test
// environment has no global location, so stub it for these tests.
const HOSTNAME = 'thorium.example.com';

// Parse a cookie string ("name=value; attr=val; flag") into name/value plus a
// case-insensitive map of its attributes so we can assert on them directly.
function parseCookie(cookie: string) {
  const [pair, ...attrs] = cookie.split(';').map((part) => part.trim());
  const eq = pair.indexOf('=');
  const attrMap: Record<string, string | true> = {};
  for (const attr of attrs) {
    const idx = attr.indexOf('=');
    if (idx === -1) {
      attrMap[attr.toLowerCase()] = true;
    } else {
      attrMap[attr.slice(0, idx).toLowerCase()] = attr.slice(idx + 1);
    }
  }
  return { name: pair.slice(0, eq), value: pair.slice(eq + 1), attrs: attrMap };
}

describe('THORIUM_TOKEN cookie contract', () => {
  beforeEach(() => {
    vi.stubGlobal('location', { hostname: HOSTNAME });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('buildCookie writes the token with an explicit domain and path', () => {
    const { name, value, attrs } = parseCookie(buildCookie('tok-123', 'Wed, 01 Jan 2030 00:00:00 GMT'));
    expect(name).toBe('THORIUM_TOKEN');
    expect(value).toBe('tok-123');
    expect(attrs.domain).toBe(HOSTNAME);
    expect(attrs.path).toBe('/');
    expect(attrs.secure).toBe(true);
  });

  it('buildRevokeCookie expires the token and clears the value', () => {
    const { name, value, attrs } = parseCookie(buildRevokeCookie());
    expect(name).toBe('THORIUM_TOKEN');
    expect(value).toBe('');
    expect(attrs['max-age']).toBe('0');
  });

  // Regression: a cookie written with an explicit Domain can only be deleted by a write
  // carrying the SAME domain + path. If buildRevokeCookie drops domain (or path), the
  // browser keeps the real cookie and logout silently re-authenticates on reload.
  it('buildRevokeCookie mirrors buildCookie domain and path so logout actually clears it', () => {
    const set = parseCookie(buildCookie('tok-123', 'Wed, 01 Jan 2030 00:00:00 GMT')).attrs;
    const revoke = parseCookie(buildRevokeCookie()).attrs;
    expect(revoke.domain).toBe(set.domain);
    expect(revoke.path).toBe(set.path);
  });
});
