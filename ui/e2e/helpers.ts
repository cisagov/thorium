import axios, { AxiosInstance } from 'axios';
import type { Page } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const API_URL = process.env.THORIUM_API_URL || 'http://localhost:8080';

function buildClient(token?: string): AxiosInstance {
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
  const res = await client.post('/users/auth', {}, {
    headers: { Authorization: `basic ${encoded}` },
  });
  return res.data.token;
}

export async function createEntity(
  token: string,
  name: string,
  kind: string,
  groups: string[],
): Promise<string> {
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

export async function createAssociation(
  token: string,
  request: AssociationRequest,
): Promise<void> {
  const client = buildClient(token);
  await client.post('/associations/', request);
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
  const [, html] = await Promise.all([
    page.screenshot({ path: pngPath, fullPage: true }),
    page.content(),
  ]);
  fs.writeFileSync(htmlPath, html, 'utf-8');
}
