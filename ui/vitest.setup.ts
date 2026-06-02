import { vi } from 'vitest';

// Auto-mock @thorpi/client to prevent browser-global side effects in node.
// The real client.ts guards window/document access, but this mock provides
// a safety net and gives tests a stub axios instance by default.
vi.mock('@thorpi/client', () => {
  const stubClient = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    patch: vi.fn(),
    interceptors: { request: { use: vi.fn() } },
  };
  return {
    default: stubClient,
    bigIntClient: { ...stubClient, interceptors: { request: { use: vi.fn() } } },
    parseRequestError: vi.fn(),
  };
});
