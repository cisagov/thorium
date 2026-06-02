import { describe, test, expect } from 'vitest';

// project imports
import { imageToEditorObject, editorObjectToImageCreate, editorObjectToImageUpdate } from './image';

describe('imageToEditorObject', () => {
  test('strips read-only fields', () => {
    const image = {
      group: 'analysis',
      name: 'scanner',
      creator: 'admin',
      runtime: { count: 5 },
      used_by: ['pipeline-a'],
      bans: [],
      image: 'thorium/scanner:latest',
    };
    const result = imageToEditorObject(image);
    expect(result).not.toHaveProperty('creator');
    expect(result).not.toHaveProperty('runtime');
    expect(result).not.toHaveProperty('used_by');
    expect(result).not.toHaveProperty('bans');
  });

  test('preserves editable fields', () => {
    const image = {
      group: 'analysis',
      name: 'scanner',
      image: 'thorium/scanner:latest',
      timeout: 300,
      scaler: 'K8s',
      resources: { cpu: 1000 },
      env: { PATH: '/usr/bin' },
      volumes: [{ name: 'data', hostPath: '/mnt/data' }],
    };
    const result = imageToEditorObject(image);
    expect(result.group).toBe('analysis');
    expect(result.name).toBe('scanner');
    expect(result.image).toBe('thorium/scanner:latest');
    expect(result.timeout).toBe(300);
    expect(result.resources).toEqual({ cpu: 1000 });
    expect(result.env).toEqual({ PATH: '/usr/bin' });
    expect(result.volumes).toEqual([{ name: 'data', hostPath: '/mnt/data' }]);
  });

  test('returns empty object for image with only read-only fields', () => {
    const image = { creator: 'admin', runtime: {}, used_by: [], bans: [] };
    const result = imageToEditorObject(image);
    expect(Object.keys(result)).toHaveLength(0);
  });

  test('converts null optional string fields to empty strings', () => {
    const image = { group: 'analysis', name: 'test', image: null, modifiers: null, description: null };
    const result = imageToEditorObject(image);
    expect(result.image).toBe('');
    expect(result.modifiers).toBe('');
    expect(result.description).toBe('');
  });

  test('preserves null for non-string fields', () => {
    const image = { group: 'analysis', name: 'test', clean_up: null, kvm: null, lifetime: null };
    const result = imageToEditorObject(image);
    expect(result.clean_up).toBeNull();
    expect(result.kvm).toBeNull();
    expect(result.lifetime).toBeNull();
  });
});

describe('editorObjectToImageCreate', () => {
  test('returns null when group is missing', () => {
    expect(editorObjectToImageCreate({ name: 'test' })).toBeNull();
  });

  test('returns null when name is missing', () => {
    expect(editorObjectToImageCreate({ group: 'analysis' })).toBeNull();
  });

  test('returns null when both group and name are missing', () => {
    expect(editorObjectToImageCreate({})).toBeNull();
  });

  test('strips null and undefined values', () => {
    const obj = { group: 'analysis', name: 'test', description: null, version: undefined, image: 'img:latest' };
    const result = editorObjectToImageCreate(obj);
    expect(result).not.toHaveProperty('description');
    expect(result).not.toHaveProperty('version');
    expect(result).toHaveProperty('image', 'img:latest');
  });

  test('strips empty objects', () => {
    const obj = { group: 'analysis', name: 'test', resources: {}, env: {} };
    const result = editorObjectToImageCreate(obj);
    expect(result).not.toHaveProperty('resources');
    expect(result).not.toHaveProperty('env');
  });

  test('strips empty arrays', () => {
    const obj = { group: 'analysis', name: 'test', volumes: [], network_policies: [] };
    const result = editorObjectToImageCreate(obj);
    expect(result).not.toHaveProperty('volumes');
    expect(result).not.toHaveProperty('network_policies');
  });

  test('preserves non-empty objects and arrays', () => {
    const obj = {
      group: 'analysis',
      name: 'test',
      resources: { cpu: 1000 },
      volumes: [{ name: 'v1' }],
    };
    const result = editorObjectToImageCreate(obj)!;
    expect(result.resources).toEqual({ cpu: 1000 });
    expect(result.volumes).toEqual([{ name: 'v1' }]);
  });

  test('preserves scalar values including zero and false', () => {
    const obj = { group: 'analysis', name: 'test', timeout: 0, collect_logs: false, spawn_limit: 0 };
    const result = editorObjectToImageCreate(obj)!;
    expect(result.timeout).toBe(0);
    expect(result.collect_logs).toBe(false);
    expect(result.spawn_limit).toBe(0);
  });

  test('preserves string values including empty string', () => {
    const obj = { group: 'analysis', name: 'test', description: '', image: 'img:latest' };
    const result = editorObjectToImageCreate(obj)!;
    expect(result.description).toBe('');
    expect(result.image).toBe('img:latest');
  });
});

describe('editorObjectToImageUpdate', () => {
  const ORIGINAL = {
    group: 'analysis',
    name: 'scanner',
    image: 'thorium/scanner:v1',
    timeout: 300,
    scaler: 'K8s',
  };

  test('returns null when original has no group', () => {
    expect(editorObjectToImageUpdate({}, { name: 'test' })).toBeNull();
  });

  test('returns null when original has no name', () => {
    expect(editorObjectToImageUpdate({}, { group: 'analysis' })).toBeNull();
  });

  test('returns group and name from original image', () => {
    const result = editorObjectToImageUpdate({ image: 'new:latest' }, ORIGINAL)!;
    expect(result.group).toBe('analysis');
    expect(result.name).toBe('scanner');
  });

  test('passes through simple update fields', () => {
    const obj = { image: 'new:latest', timeout: 600, scaler: 'BareMetal', collect_logs: true };
    const result = editorObjectToImageUpdate(obj, ORIGINAL)!;
    expect(result.data.image).toBe('new:latest');
    expect(result.data.timeout).toBe(600);
    expect(result.data.scaler).toBe('BareMetal');
    expect(result.data.collect_logs).toBe(true);
  });

  test('omits undefined simple fields', () => {
    const result = editorObjectToImageUpdate({}, ORIGINAL)!;
    expect(result.data).not.toHaveProperty('image');
    expect(result.data).not.toHaveProperty('timeout');
  });

  describe('description handling', () => {
    test('sets description when present and non-empty', () => {
      const result = editorObjectToImageUpdate({ description: 'A scanner' }, ORIGINAL)!;
      expect(result.data.description).toBe('A scanner');
      expect(result.data).not.toHaveProperty('clear_description');
    });

    test('sets clear_description when description removed and original had one', () => {
      const original = { ...ORIGINAL, description: 'Old desc' };
      const result = editorObjectToImageUpdate({}, original)!;
      expect(result.data.clear_description).toBe(true);
      expect(result.data).not.toHaveProperty('description');
    });

    test('does not set clear_description when original had no description', () => {
      const result = editorObjectToImageUpdate({}, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('clear_description');
    });

    test('whitespace-only description triggers clear_description', () => {
      const original = { ...ORIGINAL, description: 'Old desc' };
      const result = editorObjectToImageUpdate({ description: '   ' }, original)!;
      expect(result.data.clear_description).toBe(true);
      expect(result.data).not.toHaveProperty('description');
    });
  });

  describe('volume diffs', () => {
    test('detects added volumes', () => {
      const obj = { volumes: [{ name: 'data', hostPath: '/mnt' }] };
      const original = { ...ORIGINAL, volumes: [] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.add_volumes).toEqual([{ name: 'data', hostPath: '/mnt' }]);
      expect(result.data).not.toHaveProperty('remove_volumes');
    });

    test('detects removed volumes', () => {
      const obj = { volumes: [] };
      const original = { ...ORIGINAL, volumes: [{ name: 'data', hostPath: '/mnt' }] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.remove_volumes).toEqual(['data']);
      expect(result.data).not.toHaveProperty('add_volumes');
    });

    test('detects modified volumes as add + remove', () => {
      const obj = { volumes: [{ name: 'data', hostPath: '/mnt/new' }] };
      const original = { ...ORIGINAL, volumes: [{ name: 'data', hostPath: '/mnt/old' }] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.add_volumes).toEqual([{ name: 'data', hostPath: '/mnt/new' }]);
      expect(result.data.remove_volumes).toEqual(['data']);
    });

    test('no volume diffs when volumes unchanged', () => {
      const volumes = [{ name: 'data', hostPath: '/mnt' }];
      const obj = { volumes: [...volumes] };
      const original = { ...ORIGINAL, volumes: [...volumes] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data).not.toHaveProperty('add_volumes');
      expect(result.data).not.toHaveProperty('remove_volumes');
    });

    test('raw volumes key is removed from data', () => {
      const result = editorObjectToImageUpdate({ volumes: [] }, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('volumes');
    });
  });

  describe('env diffs', () => {
    test('detects added env vars', () => {
      const obj = { env: { PATH: '/usr/bin', HOME: '/root' } };
      const original = { ...ORIGINAL, env: { PATH: '/usr/bin' } };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.add_env).toEqual({ HOME: '/root' });
      expect(result.data).not.toHaveProperty('remove_env');
    });

    test('detects removed env vars', () => {
      const obj = { env: {} };
      const original = { ...ORIGINAL, env: { PATH: '/usr/bin' } };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.remove_env).toEqual(['PATH']);
      expect(result.data).not.toHaveProperty('add_env');
    });

    test('detects changed env var values', () => {
      const obj = { env: { PATH: '/opt/bin' } };
      const original = { ...ORIGINAL, env: { PATH: '/usr/bin' } };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.add_env).toEqual({ PATH: '/opt/bin' });
    });

    test('no env diffs when env unchanged', () => {
      const obj = { env: { PATH: '/usr/bin' } };
      const original = { ...ORIGINAL, env: { PATH: '/usr/bin' } };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data).not.toHaveProperty('add_env');
      expect(result.data).not.toHaveProperty('remove_env');
    });

    test('raw env key is removed from data', () => {
      const result = editorObjectToImageUpdate({ env: {} }, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('env');
    });
  });

  describe('network policy diffs', () => {
    test('detects added policies', () => {
      const obj = { network_policies: ['allow-dns', 'allow-api'] };
      const original = { ...ORIGINAL, network_policies: ['allow-dns'] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.network_policies).toEqual({
        policies_added: ['allow-api'],
        policies_removed: [],
      });
    });

    test('detects removed policies', () => {
      const obj = { network_policies: [] };
      const original = { ...ORIGINAL, network_policies: ['allow-dns'] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.network_policies).toEqual({
        policies_added: [],
        policies_removed: ['allow-dns'],
      });
    });

    test('no network policy key when policies unchanged', () => {
      const obj = { network_policies: ['allow-dns'] };
      const original = { ...ORIGINAL, network_policies: ['allow-dns'] };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data).not.toHaveProperty('network_policies');
    });
  });

  describe('combined update', () => {
    test('handles all diff types in single update', () => {
      const obj = {
        image: 'new:latest',
        timeout: 600,
        description: 'Updated',
        volumes: [{ name: 'new-vol', hostPath: '/new' }],
        env: { NEW_VAR: 'value' },
        network_policies: ['new-policy'],
      };
      const original = {
        ...ORIGINAL,
        description: 'Old desc',
        volumes: [{ name: 'old-vol', hostPath: '/old' }],
        env: { OLD_VAR: 'old' },
        network_policies: ['old-policy'],
      };
      const result = editorObjectToImageUpdate(obj, original)!;
      expect(result.data.image).toBe('new:latest');
      expect(result.data.timeout).toBe(600);
      expect(result.data.description).toBe('Updated');
      expect(result.data.add_volumes).toEqual([{ name: 'new-vol', hostPath: '/new' }]);
      expect(result.data.remove_volumes).toEqual(['old-vol']);
      expect(result.data.add_env).toEqual({ NEW_VAR: 'value' });
      expect(result.data.remove_env).toEqual(['OLD_VAR']);
      expect(result.data.network_policies).toEqual({
        policies_added: ['new-policy'],
        policies_removed: ['old-policy'],
      });
    });
  });
});
