import axios, { AxiosInstance } from 'axios';
import type { Page } from '@playwright/test';
import fs from 'fs';
import path from 'path';

export const TEST_USER = process.env.THORIUM_USER || 'test';
export const TEST_PASS = process.env.THORIUM_PASS || 'INSECURE_DEV_PASSWORD';

export const MOCK_USER = {
  username: 'test',
  role: 'Admin',
  email: 'test@thorium.dev',
  groups: ['system'],
  token: 'mock-token-for-visual-test',
  token_expiration: '2099-01-01T00:00:00Z',
  settings: { theme: 'Dark' },
  local: true,
  verified: true,
};

export async function loginViaUI(page: Page) {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.locator('input[placeholder="username"]').fill(TEST_USER);
  await page.locator('input[placeholder="password"]').fill(TEST_PASS);
  await page.locator('button:has-text("Login")').click();
  await page.waitForURL((url) => !url.pathname.includes('/auth'), { timeout: 15000 });
}

export async function setupMockAuth(page: Page) {
  await page.route('**/api/users/whoami', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) }),
  );
  await page.route('**/api/**', (route) => {
    const url = route.request().url();
    if (url.includes('/users/whoami')) return route.fallback();
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.context().addCookies([
    {
      name: 'THORIUM_TOKEN',
      value: MOCK_USER.token,
      domain: 'localhost',
      path: '/',
    },
  ]);
}

export async function waitForEditor(page: Page) {
  await page.waitForSelector('.cm-editor', { timeout: 10000 });
  await page.waitForTimeout(500);
}

export async function waitForLinter(page: Page) {
  await page.waitForTimeout(600);
}

export async function setEditorContent(page: Page, text: string) {
  await page.evaluate((content) => {
    const container = document.querySelector('.cm-editor')?.parentElement as HTMLElement & {
      _cmView?: { state: { doc: { length: number } }; dispatch: (spec: unknown) => void };
    };
    const view = container?._cmView;
    if (view) {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: content },
      });
    }
  }, text);
}

const API_URL = process.env.THORIUM_API_URL || 'http://localhost:8080';

export function buildClient(token?: string): AxiosInstance {
  const client = axios.create({ baseURL: `${API_URL}/api` });
  if (token) {
    const encoded = Buffer.from(token).toString('base64');
    client.defaults.headers.common['Authorization'] = `token ${encoded}`;
  }
  return client;
}

export async function healthCheck(): Promise<boolean> {
  const client = buildClient();
  const res = await client.get('/health');
  return res.status === 200;
}

export async function authenticate(username: string, password: string): Promise<string> {
  const client = buildClient();
  const encoded = Buffer.from(`${username}:${password}`).toString('base64');
  const res = await client.post(
    '/users/auth',
    {},
    {
      headers: { Authorization: `basic ${encoded}` },
    },
  );
  // The API returns a tagged auth response (`{ Authed: { token } }`); older builds returned a flat
  // `{ token }`. Accept both so seeding works regardless of which API version is deployed.
  return res.data?.Authed?.token ?? res.data?.token;
}

export async function createEntity(token: string, name: string, kind: string, groups: string[]): Promise<string> {
  const client = buildClient(token);
  const form = new FormData();
  form.set('name', name);
  form.set('kind', kind);
  for (const group of groups) {
    form.append('groups', group);
  }
  const res = await client.post('/entities/', form);
  return res.data.id;
}

export async function createEntityWithImage(
  token: string,
  name: string,
  kind: string,
  groups: string[],
  image: { data: Buffer; filename: string; contentType: string },
): Promise<string> {
  const client = buildClient(token);
  const form = new FormData();
  form.set('name', name);
  form.set('kind', kind);
  for (const group of groups) {
    form.append('groups', group);
  }
  form.set('image', new Blob([image.data], { type: image.contentType }), image.filename);
  const res = await client.post('/entities/', form);
  return res.data.id;
}

/**
 * A single `metadata[...]` multipart field to attach when seeding a typed entity.
 *
 * The API's entity-create route parses metadata from bracketed multipart keys (mirroring the UI's
 * `buildCreateEntityForm`), e.g. `metadata[suspicion]`, `metadata[functions][]`. Scalars use `set`;
 * list fields (`key` ending in `[]`) are appended so repeated values accumulate.
 */
export interface MetadataField {
  /** The full bracketed multipart key, e.g. `metadata[reasoning]` or `metadata[functions][]`. */
  key: string;
  /** The value for this field (numbers/booleans are stringified). */
  value: string | number | boolean;
}

/**
 * Create an entity of a metadata-carrying kind (Flag, Incident, CompiledFunction, DecompiledFunction,
 * PeSection, PeImport, ...) via `POST /entities/`.
 *
 * Mirrors the multipart contract of `buildCreateEntityForm` (`ui/src/components/entities/utilities.ts`):
 * `name`/`kind`/`groups` plus one bracketed `metadata[...]` field per entry. Keys ending in `[]` are
 * appended (list fields); all others are set (scalars). This lets details-page E2E specs seed entities
 * whose metadata renders on the page without duplicating per-kind seeding logic.
 *
 * @param token - Auth token for the seeding user.
 * @param name - The entity name (rendered in the details header and, for PE kinds, carrying the section/library name).
 * @param kind - The entity kind (an `Entities` variant string, e.g. `"Flag"`).
 * @param groups - The groups to create the entity in.
 * @param metadata - The bracketed metadata fields to attach.
 * @returns The created entity's UUID.
 */
export async function createEntityWithMetadata(
  token: string,
  name: string,
  kind: string,
  groups: string[],
  metadata: MetadataField[],
): Promise<string> {
  const client = buildClient(token);
  const form = new FormData();
  form.set('name', name);
  form.set('kind', kind);
  for (const group of groups) {
    form.append('groups', group);
  }
  for (const { key, value } of metadata) {
    const str = typeof value === 'string' ? value : String(value);
    // list fields (bracketed `[]` suffix) accumulate; scalars overwrite
    if (key.endsWith('[]')) {
      form.append(key, str);
    } else {
      form.set(key, str);
    }
  }
  const res = await client.post('/entities/', form);
  return res.data.id;
}

export async function deleteEntity(token: string, id: string): Promise<void> {
  const client = buildClient(token);
  await client.delete(`/entities/${id}`);
}

export async function uploadFile(
  token: string,
  content: Buffer | Blob,
  filename: string,
  groups: string[],
): Promise<{ sha256: string; id: string }> {
  const client = buildClient(token);
  const form = new FormData();
  form.set('data', new Blob([content]), filename);
  for (const group of groups) {
    form.append('groups', group);
  }
  try {
    const res = await client.post('/files/', form);
    return { sha256: res.data.sha256, id: res.data.id };
  } catch (err: any) {
    if (err.response?.status === 409) {
      const { createHash } = await import('crypto');
      const sha256 = createHash('sha256').update(content).digest('hex');
      return { sha256, id: '' };
    }
    throw err;
  }
}

export interface ResultFileSeed {
  name: string;
  content: Buffer;
  contentType?: string;
}

export interface UploadResultsOptions {
  tool: string;
  result: string;
  groups: string[];
  displayType?: string;
  version?: string;
  cmd?: string;
  files?: ResultFileSeed[];
}

/**
 * Seed a tool result on a file via `POST /files/results/{sha256}` (the route thorctl/agents use).
 *
 * Sends the metadata text fields (`tool`, `result`, `display_type`, `groups`, ...) followed by one
 * multipart `files` entry per result file. Lets E2E tests exercise the tool-result tile without a
 * running analysis agent.
 *
 * @returns The created result's id.
 */
export async function uploadResults(token: string, sha256: string, opts: UploadResultsOptions): Promise<string> {
  const client = buildClient(token);
  const form = new FormData();
  for (const group of opts.groups) {
    form.append('groups', group);
  }
  form.set('tool', opts.tool);
  if (opts.version) form.set('tool_version', opts.version);
  if (opts.cmd) form.set('cmd', opts.cmd);
  form.set('result', opts.result);
  // NOTE: the multipart field uses the FromStr spelling ("Json"), not the serialized "JSON"
  form.set('display_type', opts.displayType ?? 'Json');
  for (const file of opts.files ?? []) {
    form.append('files', new Blob([file.content], { type: file.contentType ?? 'application/octet-stream' }), file.name);
  }
  const res = await client.post(`/files/results/${sha256}`, form);
  return res.data.id;
}

export interface AssociationTarget {
  Entity?: { id: string; name: string };
  File?: string;
  Repo?: string;
}

export interface AssociationRequest {
  kind: string;
  source: AssociationTarget;
  targets: AssociationTarget[];
  groups: string[];
  is_bidirectional: boolean;
}

export async function buildTree(
  token: string,
  seed: Record<string, unknown>,
): Promise<{ id: string; initial: string[]; growable: string[] }> {
  const client = buildClient(token);
  const res = await client.post('/trees/', seed, {
    params: { filter_childless: true, limit: 1 },
  });
  return res.data;
}

export async function createAssociation(token: string, request: AssociationRequest): Promise<void> {
  const client = buildClient(token);
  await client.post('/associations/', request);
}

export async function createImageViaAPI(token: string, imageRequest: Record<string, unknown>): Promise<boolean> {
  const client = buildClient(token);
  const res = await client.post('/images/', imageRequest);
  return res.status === 200 || res.status === 204;
}

export async function getImageViaAPI(token: string, group: string, name: string): Promise<Record<string, unknown> | null> {
  const client = buildClient(token);
  try {
    const res = await client.get(`/images/data/${group}/${name}`);
    return res.data;
  } catch {
    return null;
  }
}

export async function updateImageViaAPI(token: string, group: string, name: string, update: Record<string, unknown>): Promise<boolean> {
  const client = buildClient(token);
  const res = await client.patch(`/images/${group}/${name}`, update);
  return res.status === 200 || res.status === 204;
}

export async function deleteImageViaAPI(token: string, group: string, name: string): Promise<boolean> {
  const client = buildClient(token);
  try {
    const res = await client.delete(`/images/${group}/${name}`);
    return res.status === 200 || res.status === 204;
  } catch {
    return false;
  }
}

export async function createPipelineViaAPI(token: string, pipelineRequest: Record<string, unknown>): Promise<boolean> {
  const client = buildClient(token);
  const res = await client.post('/pipelines/', pipelineRequest);
  return res.status === 200 || res.status === 204;
}

export async function getPipelineViaAPI(token: string, group: string, name: string): Promise<Record<string, unknown> | null> {
  const client = buildClient(token);
  try {
    const res = await client.get(`/pipelines/data/${group}/${name}`);
    return res.data;
  } catch {
    return null;
  }
}

export async function updatePipelineViaAPI(token: string, group: string, name: string, update: Record<string, unknown>): Promise<boolean> {
  const client = buildClient(token);
  const res = await client.patch(`/pipelines/${group}/${name}`, update);
  return res.status === 200 || res.status === 204;
}

export async function deletePipelineViaAPI(token: string, group: string, name: string): Promise<boolean> {
  const client = buildClient(token);
  try {
    const res = await client.delete(`/pipelines/${group}/${name}`);
    return res.status === 200 || res.status === 204;
  } catch {
    return false;
  }
}

/**
 * Capture a paired screenshot + HTML snapshot of the current page state.
 *
 * Writes two files with the same base name:
 *   <dir>/<name>.png   — full-page screenshot
 *   <dir>/<name>.html  — raw page HTML at the moment of capture
 *
 * Use the HTML file to inspect DOM structure, CSS classes, aria attributes,
 * and element ordering when a screenshot alone isn't enough to diagnose a
 * test failure.
 */
export async function snapshot(page: Page, dir: string, name: string): Promise<void> {
  const pngPath = path.join(dir, `${name}.png`);
  const htmlPath = path.join(dir, `${name}.html`);
  const [, html] = await Promise.all([page.screenshot({ path: pngPath, fullPage: true }), page.content()]);
  fs.writeFileSync(htmlPath, html, 'utf-8');
}
