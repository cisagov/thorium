import { Value } from '@models/results';

// set warnings and errors from results
const getAlerts = (
  result: Value,
  setResults: (results: Value) => void,
  setWarnings: (warnings: string[]) => void,
  setErrors: (errors: string[]) => void,
  setIsJson: (isJson: boolean) => void,
  ignoreJsonError: boolean,
) => {
  // handle empty result
  const jsonFormattedResults = result;
  let isJson = false;
  const errors: string[] = [];
  const warnings: string[] = [];

  // check if an empty result was returned by tool run
  if (result && (result == '' || result == '{}' || result == '[]')) {
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

  // Narrow to record type for bracket access on the json object
  if (isJson && typeof jsonFormattedResults === 'object' && jsonFormattedResults !== null && !Array.isArray(jsonFormattedResults)) {
    const obj = jsonFormattedResults as Record<string, Value>;

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
    setResults(jsonFormattedResults);
  }
};

export { getAlerts };
