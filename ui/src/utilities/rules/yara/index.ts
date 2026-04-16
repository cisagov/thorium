// project imports
import type { RuleChecker, CheckResult } from '../types';
import { FormatType, Severity } from '../types';
import { parseYaraText } from './parse';
import { validateYaraRules } from './validate';
import { generateYaraSuggestions } from './suggestions';

export class YaraRuleChecker implements RuleChecker {
  format = FormatType.YARA;

  check(text: string): CheckResult {
    if (!text.trim()) {
      return { diagnostics: [], suggestions: [] };
    }

    const parseResult = parseYaraText(text);

    const parseDiagnostics = parseResult.errors.map((e) => ({
      line: e.line,
      severity: Severity.Error,
      message: e.message,
    }));

    const ruleDiagnostics = validateYaraRules(parseResult);
    const suggestions = generateYaraSuggestions(parseResult);

    return {
      diagnostics: [...parseDiagnostics, ...ruleDiagnostics],
      suggestions,
    };
  }
}
