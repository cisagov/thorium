import { describe, it, expect, vi } from 'vitest';

// project imports
import { formatResultBody, getAlerts, normalizeResultText } from './alerts';
import { Value } from '@models/results';

// Capture the values passed to getAlerts's setter callbacks so tests can assert on them
// without rendering any component.
function callGetAlerts(result: Value, ignoreJsonError = false) {
  const setResults = vi.fn<(r: Value) => void>();
  const setWarnings = vi.fn<(w: string[]) => void>();
  const setErrors = vi.fn<(e: string[]) => void>();
  const setIsJson = vi.fn<(j: boolean) => void>();
  getAlerts(result, setResults, setWarnings, setErrors, setIsJson, ignoreJsonError);
  return {
    results: setResults.mock.calls.at(-1)?.[0],
    warnings: setWarnings.mock.calls.at(-1)?.[0] ?? [],
    errors: setErrors.mock.calls.at(-1)?.[0] ?? [],
    isJson: setIsJson.mock.calls.at(-1)?.[0],
  };
}

describe('getAlerts', () => {
  it('extracts array errors and warnings into the setter callbacks', () => {
    const { errors, warnings } = callGetAlerts({ errors: ['boom'], warnings: ['heads up'] });
    expect(errors).toContain('boom');
    expect(warnings).toContain('heads up');
  });

  it('extracts capitalized and singular error/warning variants', () => {
    const { errors, warnings } = callGetAlerts({
      Errors: ['cap-errs'],
      error: 'single-err',
      Error: 'single-cap-err',
      Warnings: ['cap-warns'],
      warning: 'single-warn',
      Warning: 'single-cap-warn',
    });
    expect(errors).toEqual(expect.arrayContaining(['cap-errs', 'single-err', 'single-cap-err']));
    expect(warnings).toEqual(expect.arrayContaining(['cap-warns', 'single-warn', 'single-cap-warn']));
  });

  it('does not mutate the input object (errors/warnings keys stay intact)', () => {
    const input = { errors: ['boom'], warnings: ['heads up'], data: 42 };
    callGetAlerts(input);
    // the shared result object must be left untouched so isEmptyResult/diff read stable data
    expect(input).toEqual({ errors: ['boom'], warnings: ['heads up'], data: 42 });
  });

  it('strips errors/warnings from the object handed to the display body', () => {
    const { results } = callGetAlerts({ errors: ['boom'], warnings: ['heads up'], data: 42 });
    expect(results).toEqual({ data: 42 });
  });

  it('leaves a pure-error object non-empty in the input but empty in the display body', () => {
    const input = { errors: ['only an error'] };
    const { results } = callGetAlerts(input);
    // input still has its key, so isEmptyResult(input) stays false and the error banner shows
    expect(Object.keys(input)).toHaveLength(1);
    // the displayed body is emptied so the error isn't duplicated in the json dump
    expect(results).toEqual({});
  });

  it('classifies a plain string as non-json and warns it cannot render as json', () => {
    const { isJson, warnings } = callGetAlerts('just text', false);
    expect(isJson).toBe(false);
    expect(warnings.join(' ')).toMatch(/Could not display result as json/);
  });

  it('suppresses the json-parse warning when ignoreJsonError is set', () => {
    const { isJson, warnings } = callGetAlerts('just text', true);
    expect(isJson).toBe(false);
    expect(warnings).toHaveLength(0);
  });

  it('flags an empty result string as json with an output warning', () => {
    const { isJson, warnings } = callGetAlerts('{}', true);
    expect(isJson).toBe(true);
    expect(warnings.join(' ')).toMatch(/Tool did not produce output/);
  });

  it('flags the empty string result with an output warning (not short-circuited)', () => {
    const { isJson, warnings } = callGetAlerts('', true);
    expect(isJson).toBe(true);
    expect(warnings.join(' ')).toMatch(/Tool did not produce output/);
  });

  it('flags an empty json array result with an output warning', () => {
    const { isJson, warnings } = callGetAlerts('[]', true);
    expect(isJson).toBe(true);
    expect(warnings.join(' ')).toMatch(/Tool did not produce output/);
  });
});

describe('formatResultBody', () => {
  // the renderer (String/Tables/Markdown/XML) only consults resultsJson when isJson is true,
  // so a placeholder is fine for the non-json cases
  const NO_JSON: Value = {};

  it('returns an empty string for a non-json non-string result', () => {
    expect(formatResultBody(null, false, NO_JSON)).toBe('');
    expect(formatResultBody(undefined as never, false, NO_JSON)).toBe('');
    expect(formatResultBody({ a: 1 }, false, NO_JSON)).toBe('');
  });

  it('unescapes newlines and strips quotes from a non-json string', () => {
    expect(formatResultBody('line1\\nline2', false, NO_JSON)).toBe('line1\nline2');
    expect(formatResultBody('"quoted"', false, NO_JSON)).toBe('quoted');
  });

  it('passes a plain non-json string through unchanged', () => {
    expect(formatResultBody('plain text', false, NO_JSON)).toBe('plain text');
  });

  it('returns an empty string for an empty json object', () => {
    expect(formatResultBody('{}', true, {})).toBe('');
  });

  it('stringifies filled json content', () => {
    expect(formatResultBody('', true, { a: 1 })).toBe('{"a":1}');
  });

  it('stringifies array json content (not collapsed to empty)', () => {
    expect(formatResultBody('', true, [])).toBe('[]');
    expect(formatResultBody('', true, [1, 2])).toBe('[1,2]');
  });
});

describe('normalizeResultText', () => {
  it('turns escaped newline sequences into real newlines', () => {
    expect(normalizeResultText('line1\\nline2')).toBe('line1\nline2');
  });

  it('strips every double-quote character', () => {
    expect(normalizeResultText('"a" and "b"')).toBe('a and b');
  });

  it('applies both transforms together', () => {
    expect(normalizeResultText('"line1"\\n"line2"')).toBe('line1\nline2');
  });

  it('passes text with neither escape nor quotes through unchanged', () => {
    expect(normalizeResultText('plain text')).toBe('plain text');
  });
});
