import { useEffect, useState } from 'react';

// project imports
import { getAlerts } from '../alerts';
import { Value } from '@models/results';

/** The alert/JSON state a display component needs from {@link getAlerts}. */
export interface ResultAlertsState {
  /** Error strings extracted from the result. */
  errors: string[];
  /** Warning strings extracted from the result. */
  warnings: string[];
  /** The display JSON body (only meaningful when `isJson`). */
  resultsJson: Value;
  /** Whether the result was classified as JSON. */
  isJson: boolean;
}

/**
 * Run {@link getAlerts} against a result value and expose its outputs as reactive state.
 *
 * Factors out the identical alert-extraction wiring shared by the text/JSON display components so
 * each display no longer repeats the same four `useState` hooks and effect.
 *
 * @param result - The raw `Output.result` value to classify and extract alerts from.
 * @param ignoreJsonError - When true, suppresses the "could not display as json" warning for
 *   non-JSON string results (used by string-oriented displays that expect raw text).
 * @param initialJson - Initial value for the display JSON body; defaults to an empty object.
 * @returns The extracted errors, warnings, display JSON body, and JSON classification flag.
 */
export function useResultAlerts(result: Value, ignoreJsonError: boolean, initialJson: Value = {}): ResultAlertsState {
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [resultsJson, setResultsJson] = useState<Value>(initialJson);
  const [isJson, setIsJson] = useState(true);
  useEffect(() => {
    getAlerts(result, setResultsJson, setWarnings, setErrors, setIsJson, ignoreJsonError);
  }, [result, ignoreJsonError]);
  return { errors, warnings, resultsJson, isJson };
}
