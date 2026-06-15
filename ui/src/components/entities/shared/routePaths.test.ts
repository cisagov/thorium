import { describe, it, expect } from 'vitest';

// project imports
import { buildPathByType, normalizeRoutePath } from './routePaths';
import { Entities } from '@models/entities';

describe('normalizeRoutePath', () => {
  it('strips a trailing /:param segment', () => {
    expect(normalizeRoutePath('/device/:entityID')).toBe('/device');
  });
  it('strips a trailing /* wildcard', () => {
    expect(normalizeRoutePath('/repos/*')).toBe('/repos');
  });
  it('leaves a plain path unchanged', () => {
    expect(normalizeRoutePath('/files')).toBe('/files');
  });
  it('only strips a param when it is the final segment', () => {
    expect(normalizeRoutePath('/pe/section/:entityID')).toBe('/pe/section');
  });
});

describe('buildPathByType', () => {
  it('maps each entity type to its normalized base path', () => {
    const routes = {
      '/device/:entityID': Entities.Device,
      '/vendor/:entityID': Entities.Vendor,
    };
    const result = buildPathByType(routes, (type) => type);
    expect(result[Entities.Device]).toBe('/device');
    expect(result[Entities.Vendor]).toBe('/vendor');
  });

  it('prefers a non-wildcard route over a wildcard one when the wildcard comes first', () => {
    const routes = {
      '/repos/*': Entities.Repo,
      '/repos': Entities.Repo,
    };
    const result = buildPathByType(routes, (type) => type);
    expect(result[Entities.Repo]).toBe('/repos');
  });

  it('prefers a non-wildcard route over a wildcard one when the wildcard comes second', () => {
    const routes = {
      '/repos': Entities.Repo,
      '/repos/*': Entities.Repo,
    };
    const result = buildPathByType(routes, (type) => type);
    expect(result[Entities.Repo]).toBe('/repos');
  });

  it('keeps a wildcard route when it is the only one for a type', () => {
    const routes = {
      '/repo/*': Entities.Repo,
    };
    const result = buildPathByType(routes, (type) => type);
    expect(result[Entities.Repo]).toBe('/repo');
  });

  it('extracts the type via the accessor from a wrapper value', () => {
    const routes = {
      '/flag/:entityID': { type: Entities.Flag },
    };
    const result = buildPathByType(routes, (value) => value.type);
    expect(result[Entities.Flag]).toBe('/flag');
  });
});
