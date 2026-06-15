import { describe, it, expect, vi } from 'vitest';
import JSZip from 'jszip';

// project imports
import { buildResultsZip, resultsBundleName, type ResultsMetadata } from './zip';
import { OutputDisplayType, type Output } from '@models/results';

function makeOutput(patch: Partial<Output> = {}): Output {
  return {
    id: 'result-id',
    groups: ['group'],
    uploaded: '2026-06-15T00:00:00Z',
    result: { hello: 'world', nested: { a: 1 } },
    files: [],
    display_type: OutputDisplayType.Json,
    children: {},
    ...patch,
  };
}

const textOf = (bytes: string) => new TextEncoder().encode(bytes).buffer;

/** Read and parse the result-metadata.json manifest from a built bundle. */
async function metadataOf(blob: Blob): Promise<ResultsMetadata | undefined> {
  const zip = await JSZip.loadAsync(await blob.arrayBuffer());
  const json = await zip.file('result-metadata.json')?.async('string');
  return json ? (JSON.parse(json) as ResultsMetadata) : undefined;
}

describe('buildResultsZip', () => {
  it('writes the structured result as pretty-printed result.json', async () => {
    const output = makeOutput();
    const blob = await buildResultsZip('tool', output, () => Promise.resolve(null));

    const zip = await JSZip.loadAsync(await blob.arrayBuffer());
    const json = await zip.file('result.json')?.async('string');
    expect(json).toBe(JSON.stringify(output.result, null, 2));
  });

  it('fetches and includes each result file', async () => {
    const output = makeOutput({ files: ['a.txt', 'b.bin'] });
    const fetchFile = vi.fn((name: string) => Promise.resolve(textOf(`bytes-of-${name}`)));

    const blob = await buildResultsZip('tool', output, fetchFile);
    const zip = await JSZip.loadAsync(await blob.arrayBuffer());

    expect(fetchFile).toHaveBeenCalledTimes(2);
    expect(fetchFile).toHaveBeenCalledWith('a.txt');
    expect(fetchFile).toHaveBeenCalledWith('b.bin');
    expect(await zip.file('a.txt')?.async('string')).toBe('bytes-of-a.txt');
    expect(await zip.file('b.bin')?.async('string')).toBe('bytes-of-b.bin');
  });

  it('skips files that fail to download', async () => {
    const output = makeOutput({ files: ['ok.txt', 'missing.txt'] });
    const fetchFile = vi.fn((name: string) => Promise.resolve(name === 'ok.txt' ? textOf('ok') : null));

    const blob = await buildResultsZip('tool', output, fetchFile);
    const zip = await JSZip.loadAsync(await blob.arrayBuffer());

    expect(zip.file('ok.txt')).not.toBeNull();
    expect(zip.file('missing.txt')).toBeNull();
  });

  it('handles a result with no files', async () => {
    const output = makeOutput({ files: undefined });
    const fetchFile = vi.fn(() => Promise.resolve(null));

    const blob = await buildResultsZip('tool', output, fetchFile);
    const zip = await JSZip.loadAsync(await blob.arrayBuffer());

    expect(fetchFile).not.toHaveBeenCalled();
    expect(zip.file('result.json')).not.toBeNull();
  });

  describe('result-metadata.json manifest', () => {
    it('records the tool, version, upload time, and downloaded artifacts', async () => {
      const output = makeOutput({
        files: ['a.txt', 'b.bin'],
        tool_version: { SemVer: { major: 1, minor: 2, patch: 3, pre: '', build: '' } },
      });
      const fetchFile = vi.fn((name: string) => Promise.resolve(textOf(`bytes-of-${name}`)));

      const metadata = await metadataOf(await buildResultsZip('unpacker', output, fetchFile));

      expect(metadata).toEqual({
        tool: 'unpacker',
        tool_version: 'v1.2.3',
        uploaded: '2026-06-15T00:00:00Z',
        artifacts: ['a.txt', 'b.bin'],
      });
    });

    it('only lists artifacts that were successfully downloaded', async () => {
      const output = makeOutput({ files: ['ok.txt', 'missing.txt'] });
      const fetchFile = vi.fn((name: string) => Promise.resolve(name === 'ok.txt' ? textOf('ok') : null));

      const metadata = await metadataOf(await buildResultsZip('unpacker', output, fetchFile));

      expect(metadata?.artifacts).toEqual(['ok.txt']);
    });

    it('omits tool_version when the result has no version', async () => {
      const metadata = await metadataOf(await buildResultsZip('unpacker', makeOutput(), () => Promise.resolve(null)));

      expect(metadata?.tool_version).toBeUndefined();
    });

    it('leaves a result-supplied result-metadata.json untouched', async () => {
      const output = makeOutput({ files: ['result-metadata.json'] });
      const fetchFile = vi.fn(() => Promise.resolve(textOf('{"from":"tool"}')));

      const zip = await JSZip.loadAsync(await (await buildResultsZip('unpacker', output, fetchFile)).arrayBuffer());
      const json = await zip.file('result-metadata.json')?.async('string');

      // the tool's own file is preserved rather than overwritten with a generated manifest
      expect(json).toBe('{"from":"tool"}');
    });

    it('still writes a generated manifest when the tool-supplied one fails to download', async () => {
      const output = makeOutput({ files: ['result-metadata.json'] });
      // the tool declares its own manifest but the fetch fails, so the bundle would have none
      const fetchFile = vi.fn(() => Promise.resolve(null));

      const metadata = await metadataOf(await buildResultsZip('unpacker', output, fetchFile));

      // gating on downloaded artifacts (not the declared list) keeps the bundle self-describing
      expect(metadata).toBeDefined();
      expect(metadata?.artifacts).toEqual([]);
    });
  });

  describe('result-file name sanitization', () => {
    it('strips traversal segments so entries cannot escape the archive', async () => {
      const output = makeOutput({ files: ['../../etc/passwd'] });
      const fetchFile = vi.fn(() => Promise.resolve(textOf('pwned')));

      const zip = await JSZip.loadAsync(await (await buildResultsZip('tool', output, fetchFile)).arrayBuffer());

      // the '..' components are dropped, leaving a safe nested path
      expect(zip.file('etc/passwd')).not.toBeNull();
      expect(zip.file('../../etc/passwd')).toBeNull();
    });

    it('strips a leading slash from absolute-looking names', async () => {
      const output = makeOutput({ files: ['/abs/name.txt'] });
      const fetchFile = vi.fn(() => Promise.resolve(textOf('x')));

      const zip = await JSZip.loadAsync(await (await buildResultsZip('tool', output, fetchFile)).arrayBuffer());

      expect(zip.file('abs/name.txt')).not.toBeNull();
    });

    it('preserves legitimate nested paths', async () => {
      const output = makeOutput({ files: ['sub/dir/file.txt'] });
      const fetchFile = vi.fn(() => Promise.resolve(textOf('x')));

      const zip = await JSZip.loadAsync(await (await buildResultsZip('tool', output, fetchFile)).arrayBuffer());

      expect(zip.file('sub/dir/file.txt')).not.toBeNull();
    });
  });
});

describe('resultsBundleName', () => {
  it('combines tool, sha256, and upload timestamp', () => {
    const output = makeOutput();
    expect(resultsBundleName('abc123', 'unpacker', output)).toBe('unpacker_abc123_2026-06-15T00-00-00Z.zip');
  });

  it('sanitizes filesystem-unsafe characters in each part', () => {
    const output = makeOutput({ uploaded: '2026-06-15T00:00:00Z' });
    // path separators and spaces in the tool name collapse to single dashes; the timestamp's colons too
    expect(resultsBundleName('sha', 'my/tool name', output)).toBe('my-tool-name_sha_2026-06-15T00-00-00Z.zip');
  });

  it('omits the timestamp when the result has no upload time', () => {
    const output = makeOutput({ uploaded: '' });
    expect(resultsBundleName('abc123', 'unpacker', output)).toBe('unpacker_abc123.zip');
  });
});
