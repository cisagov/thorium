import { Value } from '@models/results';

/**
 * Extract error/warning messages from a raw tool-result value and classify it as JSON or string.
 *
 * Recognizes errors/Errors/error/Error and warnings/Warnings/warning/Warning keys, stripping them
 * from a shallow clone of the JSON body (never mutating the shared result object). Empty results
 * ('', '{}', '[]') are reported as a warning.
 *
 * @param result - The raw `Output.result` value.
 * @param setResults - Receives the display JSON body (called only when the result is JSON).
 * @param setWarnings - Receives extracted warning strings.
 * @param setErrors - Receives extracted error strings.
 * @param setIsJson - Receives whether the result was classified as JSON.
 * @param ignoreJsonError - When true, suppresses the "could not display as json" warning for strings.
 */
const getAlerts = (
  result: Value,
  setResults: (results: Value) => void,
  setWarnings: (warnings: string[]) => void,
  setErrors: (errors: string[]) => void,
  setIsJson: (isJson: boolean) => void,
  ignoreJsonError: boolean,
) => {
  const jsonFormattedResults = result;
  let isJson = false;
  const errors: string[] = [];
  const warnings: string[] = [];

  // check if an empty result was returned by tool run; the empty string must be tested without a
  // truthiness guard (`'' && ...` short-circuits to false), otherwise '' would never warn
  if (result === '' || result == '{}' || result == '[]') {
    isJson = true;
    warnings.push(...['Tool did not produce output, check tool logs for more info.']);
  }

  // check is string, if not treat as json
  if (typeof jsonFormattedResults == 'string') {
    if (!ignoreJsonError) {
      warnings.push(
        ...[
          `Could not display result as json,
          displaying as string instead`,
        ],
      );
    }
  } else {
    isJson = true;
  }

  // the object handed to the display body; only differs from the input when we strip
  // errors/warnings keys out of a json object below
  let displayJson = jsonFormattedResults;
  // Narrow to record type for bracket access on the json object
  if (isJson && typeof jsonFormattedResults === 'object' && jsonFormattedResults !== null && !Array.isArray(jsonFormattedResults)) {
    // shallow clone so the de-dup deletes below strip errors/warnings from the *displayed*
    // body only, never from the shared result object (which isEmptyResult and the diff view
    // also read); mutating it there would retroactively flip results to "empty" on re-render
    const obj = { ...(jsonFormattedResults as Record<string, Value>) };
    displayJson = obj;

    if (obj['errors'] && Array.isArray(obj['errors'])) {
      errors.push(...(obj['errors'] as string[]));
      delete obj['errors'];
    }

    if (obj['Errors'] && Array.isArray(obj['Errors'])) {
      errors.push(...(obj['Errors'] as string[]));
      delete obj['Errors'];
    }

    if (obj['error'] && typeof obj['error'] === 'string') {
      errors.push(obj['error']);
      delete obj['error'];
    }

    if (obj['Error'] && typeof obj['Error'] === 'string') {
      errors.push(obj['Error']);
      delete obj['Error'];
    }

    if (obj['warnings'] && Array.isArray(obj['warnings'])) {
      warnings.push(...(obj['warnings'] as string[]));
      delete obj['warnings'];
    }

    if (obj['Warnings'] && Array.isArray(obj['Warnings'])) {
      warnings.push(...(obj['Warnings'] as string[]));
      delete obj['Warnings'];
    }

    if (obj['warning'] && typeof obj['warning'] === 'string') {
      warnings.push(obj['warning']);
      delete obj['warning'];
    }

    if (obj['Warning'] && typeof obj['Warning'] === 'string') {
      warnings.push(obj['Warning']);
      delete obj['Warning'];
    }
  }

  setWarnings(warnings);
  setErrors(errors);
  setIsJson(isJson);
  if (isJson) {
    setResults(displayJson);
  }
};

/**
 * Normalize a raw result string for text display: turn escaped `\n` sequences into real newlines
 * and remove every double-quote character.
 *
 * @param text - The raw result string.
 * @returns The normalized display text.
 */
const normalizeResultText = (text: string): string => text.replace(/\\n/g, '\n').replace(/["]+/g, '');

/**
 * Format an alert-processed tool result into the string a text-based renderer
 * (`String`/`Tables`/`Markdown`/`XML`) displays in its body.
 *
 * When the result isn't json, the raw string is normalized for display via {@link normalizeResultText};
 * non-strings yield an empty string.
 * When it is json, an empty object (`{}`) yields an empty string so nothing is shown, otherwise
 * the json value is stringified.
 *
 * @param result - The raw result value (the `Output.result` field).
 * @param isJson - Whether `getAlerts` classified the result as json.
 * @param resultsJson - The json value produced by `getAlerts` (consulted only when `isJson`).
 * @returns The display string for the renderer body.
 */
const formatResultBody = (result: Value, isJson: boolean, resultsJson: Value): string => {
  if (!isJson) {
    return typeof result === 'string' ? normalizeResultText(result) : '';
  }
  // an empty object carries nothing to show; anything else is dumped as json text
  if (JSON.stringify(resultsJson) === '{}') {
    return '';
  }
  return JSON.stringify(resultsJson);
};

export { getAlerts, formatResultBody, normalizeResultText };
