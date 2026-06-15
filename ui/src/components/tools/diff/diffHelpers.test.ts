import { describe, it, expect } from 'vitest';
import { DiffMethod } from 'react-diff-viewer-continued';

// project imports
import { diffForFile, diffForValue, jsonPretty, prettyJsonText, resultToYaml } from './diffHelpers';

describe('jsonPretty', () => {
  it('serializes an object with 2-space indentation', () => {
    expect(jsonPretty({ a: 1 })).toBe('{\n  "a": 1\n}');
  });

  it('falls back to String() on a circular reference instead of throwing', () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(() => jsonPretty(circular)).not.toThrow();
    expect(jsonPretty(circular)).toBe('[object Object]');
  });
});

describe('prettyJsonText', () => {
  it('re-indents valid JSON text', () => {
    expect(prettyJsonText('{"a":1}')).toBe('{\n  "a": 1\n}');
  });

  it('returns the original string verbatim for non-JSON input', () => {
    expect(prettyJsonText('not json')).toBe('not json');
  });
});

describe('diffForValue', () => {
  it('diffs two objects with the JSON method, pretty-printing both', () => {
    const { oldValue, newValue, method } = diffForValue({ a: 1 }, { a: 2 });
    expect(method).toBe(DiffMethod.JSON);
    expect(oldValue).toBe('{\n  "a": 1\n}');
    expect(newValue).toBe('{\n  "a": 2\n}');
  });

  it('diffs two strings by lines, passing them through verbatim', () => {
    const { oldValue, newValue, method } = diffForValue('one', 'two');
    expect(method).toBe(DiffMethod.LINES);
    expect(oldValue).toBe('one');
    expect(newValue).toBe('two');
  });

  it('serializes null/undefined to empty strings', () => {
    const { oldValue, newValue, method } = diffForValue(null, undefined);
    expect(method).toBe(DiffMethod.LINES);
    expect(oldValue).toBe('');
    expect(newValue).toBe('');
  });

  it('serializes a scalar via jsonPretty on the line path', () => {
    const { oldValue, method } = diffForValue(42, 'x');
    expect(method).toBe(DiffMethod.LINES);
    expect(oldValue).toBe('42');
  });
});

describe('resultToYaml', () => {
  it('returns a string value verbatim', () => {
    expect(resultToYaml('hello')).toBe('hello');
  });

  it('returns an empty string for null/undefined', () => {
    expect(resultToYaml(null)).toBe('');
    expect(resultToYaml(undefined)).toBe('');
  });

  it('renders an object as YAML text', () => {
    expect(resultToYaml({ a: 1 })).toContain('a: 1');
  });
});

describe('diffForFile', () => {
  it('uses the YAML method for .yaml and .yml', () => {
    expect(diffForFile('a.yaml', 'x', 'y').method).toBe(DiffMethod.YAML);
    expect(diffForFile('a.yml', 'x', 'y').method).toBe(DiffMethod.YAML);
  });

  it('lowercases the extension before matching', () => {
    expect(diffForFile('a.YAML', 'x', 'y').method).toBe(DiffMethod.YAML);
  });

  it('uses the JSON method for .json and pretty-prints inputs', () => {
    const { oldValue, method } = diffForFile('a.json', '{"a":1}', '{"a":2}');
    expect(method).toBe(DiffMethod.JSON);
    expect(oldValue).toBe('{\n  "a": 1\n}');
  });

  it('uses JSON when both sides parse as JSON despite no extension', () => {
    expect(diffForFile('noext', '{"a":1}', '[1,2]').method).toBe(DiffMethod.JSON);
  });

  it('falls back to the LINES method otherwise', () => {
    expect(diffForFile('a.txt', 'plain', 'text').method).toBe(DiffMethod.LINES);
  });
});
