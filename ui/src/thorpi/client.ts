import { ScopedToken } from '@models/users';
import axios, { AxiosResponse } from 'axios';
import JSONbig from 'json-bigint';

interface ClientConfig {
  headers?: Record<string, string>;
}

// Import GUI config file
const CONFIG: ClientConfig = import.meta.glob('./config.json', { eager: true, import: 'default' })['./config.json'] ?? {};

/**
 * Read a cookie value by name from `document.cookie`.
 *
 * Guarded with a `typeof document` check so the module remains importable in non-browser
 * environments (e.g. Vitest's node environment).
 *
 * @param cname - The name of the cookie to read.
 * @returns The decoded cookie value, or an empty string if not present or outside a browser.
 */
function getCookie(cname: string): string {
  if (typeof document === 'undefined') return '';
  const name = cname + '=';
  const decodedCookie = decodeURIComponent(document.cookie);
  const ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return '';
}

/**
 * Resolve the base URL (origin) the API client should target.
 *
 * In local development (`localhost`/`127.0.0.1`) it honors the `THORIUM_API_URL` build-time env
 * var so the dev server can point at a separately running API; otherwise it derives the origin
 * from the current page location. Guarded for non-browser environments.
 *
 * @returns The resolved origin (without a trailing slash or `/api` suffix), or an empty string outside a browser.
 */
function resolveBaseURL(): string {
  if (typeof window === 'undefined') return '';
  const host = window.location.hostname;
  if (host === 'localhost' || host === '127.0.0.1') {
    const apiUrl = import.meta.env.THORIUM_API_URL as string | undefined;
    if (apiUrl && apiUrl !== '') {
      return apiUrl.replace(/\/+$/, '');
    }
  }
  return `${window.location.protocol}//${window.location.hostname}`;
}

const apiURL = `${resolveBaseURL()}/api`;
const client = axios.create({
  baseURL: apiURL,
});

export function getScopedFromSession(): ScopedToken | undefined {
  try {
    const scoped = sessionStorage.getItem('THORIUM_SCOPED_TOKEN');
    if (!scoped) return undefined;
    const parsed = JSON.parse(scoped) as ScopedToken;
    return parsed;
  } catch {
    return undefined;
  }
}

export function setScopedSession(token: ScopedToken) {
  sessionStorage.setItem('THORIUM_SCOPED_TOKEN', JSON.stringify(token));
}

export function clearScopedSession() {
  sessionStorage.removeItem('THORIUM_SCOPED_TOKEN');
}

function buildAuthHeader(): string {
  const scoped = getScopedFromSession();
  const activeToken = scoped ? scoped.token : getCookie('THORIUM_TOKEN');
  return 'token ' + btoa(activeToken);
}

function buildCookieAuthHeader(): string {
  const activeToken = getCookie('THORIUM_TOKEN');
  return 'token ' + btoa(activeToken);
}

//override for scoped token routes
export function thoriumCookieAuthConfig() {
  return {
    headers: {
      Authorization: buildCookieAuthHeader(),
    },
  };
}

client.interceptors.request.use((config) => {
  if (CONFIG.headers) {
    for (const header in CONFIG.headers) {
      if (Object.hasOwn(CONFIG.headers, header)) {
        config.headers[header] = CONFIG.headers[header];
      }
    }
  }
  if (typeof config.headers.Authorization === 'undefined') {
    config.headers.Authorization = buildAuthHeader();
  }
  return config;
});

const bigIntClient = axios.create({
  baseURL: apiURL,
  transformResponse: [
    function (data: string): unknown {
      try {
        return JSONbig.parse(data) as unknown;
      } catch {
        return data;
      }
    },
  ],
});

bigIntClient.interceptors.request.use((config) => {
  if (CONFIG.headers) {
    for (const header in CONFIG.headers) {
      if (Object.hasOwn(CONFIG.headers, header)) {
        config.headers[header] = CONFIG.headers[header];
      }
    }
  }
  if (typeof config.headers.Authorization === 'undefined') {
    config.headers.Authorization = buildAuthHeader();
  }
  return config;
});

/**
 * Format an unknown request error into a human-readable message and pass it to an error handler.
 *
 * Shared by every thorpi client function to produce consistent error messages. It distinguishes
 * the axios failure modes — server response (using the response body/`error` field and appending
 * any `trace`, with 401 rendered as "Permission Denied"), no response received, request setup
 * failure, and non-axios errors.
 *
 * @param error - The caught error (typically an `AxiosError`, but any value is accepted).
 * @param errorHandler - Callback invoked with the formatted, user-facing error message.
 * @param requestType - A short label for the failed operation (e.g. `'Create Image'`) used in the message.
 */
function parseRequestError(error: unknown, errorHandler: (error: string) => void, requestType: string) {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      const data = error.response.data as Record<string, unknown> | string | undefined;
      const traceVal = typeof data === 'object' && data?.trace ? data.trace : null;
      const trace = traceVal ? ` trace: ${typeof traceVal === 'string' ? traceVal : JSON.stringify(traceVal)}` : '';
      const errorStatus = error.response.status == 401 ? 'Permission Denied' : error.response.status;
      if (typeof data === 'string' && data) {
        errorHandler(`${requestType}: ${data}${trace}`);
      } else if (typeof data === 'object' && data?.error) {
        const errMsg = typeof data.error === 'string' ? data.error : JSON.stringify(data.error);
        errorHandler(`${requestType}: ${errMsg}${trace}`);
      } else {
        errorHandler(`Failed to ${requestType}: ${errorStatus}${trace}`);
      }
    } else if (error.request) {
      errorHandler(`Failed to receive a ${requestType} request response: "${String(error.request)}`);
    } else {
      errorHandler(`Failed to setup ${requestType} request: "${error.message}"`);
    }
  } else {
    errorHandler(`Unexpected error: ${String(error)}`);
  }
}

export { client as default, bigIntClient, parseRequestError };
export type { AxiosResponse };
