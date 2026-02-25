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

  if (isJson && jsonFormattedResults && jsonFormattedResults['errors']) {
    errors.push(...jsonFormattedResults['errors']);
    delete jsonFormattedResults['errors'];
  }

  if (isJson && jsonFormattedResults && jsonFormattedResults['Errors']) {
    errors.push(...jsonFormattedResults['Errors']);
    delete jsonFormattedResults['Errors'];
  }

  if (isJson && jsonFormattedResults && jsonFormattedResults['error']) {
    errors.push(jsonFormattedResults['error']);
    delete jsonFormattedResults['error'];
  }

  if (isJson && jsonFormattedResults && jsonFormattedResults['Error']) {
    errors.push(jsonFormattedResults['Error']);
    delete jsonFormattedResults['Error'];
  }

  if (isJson && jsonFormattedResults && jsonFormattedResults['warnings']) {
    warnings.push(...jsonFormattedResults['warnings']);
    delete jsonFormattedResults['warnings'];
  }

  if (isJson && jsonFormattedResults && jsonFormattedResults['Warnings']) {
    warnings.push(...jsonFormattedResults['Warnings']);
    delete jsonFormattedResults['Warnings'];
  }

  if (isJson && jsonFormattedResults && jsonFormattedResults['warning']) {
    warnings.push(jsonFormattedResults['warning']);
    delete jsonFormattedResults['warning'];
  }

  if (jsonFormattedResults && jsonFormattedResults['Warning']) {
    warnings.push(jsonFormattedResults['Warning']);
    delete jsonFormattedResults['Warning'];
  }

  setWarnings(warnings);
  setErrors(errors);
  setIsJson(isJson);
  if (isJson) {
    setResults(jsonFormattedResults);
  }
};

export { getAlerts };
