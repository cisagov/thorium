import { type RuleChecker, type CheckResult, FormatType, Severity } from '../../types';
import { parseYaml } from '../../yaml';
import { parseJson } from '../../json';
import { validatePipelineRequest } from './validate';
import { generatePipelineSuggestions } from './suggestions';

export class PipelineChecker implements RuleChecker {
  format = FormatType.YAML;
  private imageGroup: string | null = null;
  private validImageNames: Set<string> | null = null;

  setValidImageNames(group: string, names: string[]) {
    this.imageGroup = group;
    this.validImageNames = new Set(names);
  }

  clearValidImageNames() {
    this.imageGroup = null;
    this.validImageNames = null;
  }

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
            message: 'Pipeline request must be a YAML mapping (key-value pairs), not a list or scalar',
          },
        ],
        suggestions: [],
      };
    }

    const parsed = value as Record<string, unknown>;
    const parsedGroup = typeof parsed.group === 'string' ? parsed.group : null;
    const imageNames = parsedGroup && parsedGroup === this.imageGroup ? this.validImageNames : null;
    const ruleDiagnostics = validatePipelineRequest(doc, text, parsed, imageNames);
    const suggestions = generatePipelineSuggestions(doc, text, parsed, imageNames);

    return {
      diagnostics: [...syntaxDiagnostics, ...ruleDiagnostics],
      suggestions,
    };
  }
}
