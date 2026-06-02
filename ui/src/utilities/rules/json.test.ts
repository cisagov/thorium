import { describe, test, expect } from 'vitest';
import { parseJson } from './json';

describe('parseJson', () => {
  test('empty string returns no diagnostics', () => {
    const result = parseJson('');
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value).toBeNull();
    expect(result.doc).toBeNull();
  });

  test('valid JSON parses successfully', () => {
    const result = parseJson('{"group": "test", "name": "minimal"}');
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value).toEqual({ group: 'test', name: 'minimal' });
    expect(result.doc).not.toBeNull();
  });

  test('invalid JSON returns syntax error', () => {
    const result = parseJson('{"key": }');
    expect(result.diagnostics.length).toBeGreaterThan(0);
    expect(result.diagnostics[0].severity).toBe('error');
    expect(result.value).toBeNull();
    expect(result.doc).toBeNull();
  });

  test('trailing comma returns error', () => {
    const result = parseJson('{"key": "value",}');
    expect(result.diagnostics.length).toBeGreaterThan(0);
    expect(result.diagnostics[0].severity).toBe('error');
  });

  test('unquoted key returns error', () => {
    const result = parseJson('{key: "value"}');
    expect(result.diagnostics.length).toBeGreaterThan(0);
    expect(result.diagnostics[0].severity).toBe('error');
  });

  test('syntax error has line and column', () => {
    const result = parseJson('{\n  "key": \n}');
    expect(result.diagnostics.length).toBeGreaterThan(0);
    const diag = result.diagnostics[0];
    expect(diag.line).toBeGreaterThanOrEqual(1);
    expect(diag.column).toBeGreaterThanOrEqual(1);
  });

  test('duplicate keys are detected', () => {
    const result = parseJson('{"key": 1, "key": 2}');
    const dupes = result.diagnostics.filter((d) => d.message.includes('Duplicate key'));
    expect(dupes.length).toBeGreaterThan(0);
  });

  test('nested duplicate keys are detected', () => {
    const result = parseJson('{"outer": {"inner": 1, "inner": 2}}');
    const dupes = result.diagnostics.filter((d) => d.message.includes('Duplicate key'));
    expect(dupes.length).toBeGreaterThan(0);
    expect(dupes.some((d) => d.message.includes("'inner'"))).toBe(true);
  });

  test('duplicate keys at different nesting levels do not conflict', () => {
    const result = parseJson('{"key": {"key": 1}}');
    const dupes = result.diagnostics.filter((d) => d.message.includes('Duplicate key'));
    expect(dupes).toHaveLength(0);
  });

  test('valid nested JSON has no errors', () => {
    const result = parseJson('{"resources": {"cpu": 1000, "memory": 4096}, "group": "test"}');
    expect(result.diagnostics).toHaveLength(0);
    expect(result.value).toEqual({ resources: { cpu: 1000, memory: 4096 }, group: 'test' });
  });
});
