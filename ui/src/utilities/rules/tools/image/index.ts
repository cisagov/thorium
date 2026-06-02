import { type RuleChecker, type CheckResult, FormatType, Severity } from '../../types';
import { parseYaml } from '../../yaml';
import { parseJson } from '../../json';
import { validateImageRequest } from './validate';
import { generateImageSuggestions } from './suggestions';

export class ImageChecker implements RuleChecker {
  format = FormatType.YAML;

  check(text: string): CheckResult {
    const { doc, value, diagnostics: syntaxDiagnostics } = this.format === FormatType.JSON ? parseJson(text) : parseYaml(text);

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
            message: 'Image request must be a YAML mapping (key-value pairs), not a list or scalar',
          },
        ],
        suggestions: [],
      };
    }

    const parsed = value as Record<string, unknown>;
    const ruleDiagnostics = validateImageRequest(doc, text, parsed);
    const suggestions = generateImageSuggestions(doc, text, parsed);

    return {
      diagnostics: [...syntaxDiagnostics, ...ruleDiagnostics],
      suggestions,
    };
  }
}
