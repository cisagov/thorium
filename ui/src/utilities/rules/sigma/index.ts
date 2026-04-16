// project imports
import { RuleChecker, CheckResult, FormatType, Severity } from '../types';
import { parseYaml } from '../yaml';
import { validateSigmaRule } from './validate';
import { generateSuggestions } from './suggestions';

export class SigmaRuleChecker implements RuleChecker {
  format = FormatType.YAML;

  check(text: string): CheckResult {
    const { doc, value, diagnostics: syntaxDiagnostics } = parseYaml(text);

    if (syntaxDiagnostics.some((d) => d.severity === Severity.Error) || !doc || !value) {
      return { diagnostics: syntaxDiagnostics, suggestions: [] };
    }

    if (typeof value !== 'object' || Array.isArray(value)) {
      return {
        diagnostics: [
          ...syntaxDiagnostics,
          {
            line: 1,
            column: 1,
            severity: Severity.Error,
            message: 'Sigma rule must be a YAML mapping (key-value pairs), not a list or scalar',
          },
        ],
        suggestions: [],
      };
    }

    const parsed = value as Record<string, unknown>;
    const ruleDiagnostics = validateSigmaRule(doc, text, parsed);
    const suggestions = generateSuggestions(doc, text, parsed);

    return {
      diagnostics: [...syntaxDiagnostics, ...ruleDiagnostics],
      suggestions,
    };
  }
}
