// project imports
import type { Suggestion } from '../types';
import type { YaraParseResult, YaraRuleInfo } from './parse';
import { COMMON_META_KEYS, KNOWN_MODULES, STRING_MODIFIERS } from './schema';

function getSectionRange(rule: YaraRuleInfo, section: 'meta' | 'strings' | 'condition'): { start: number; end: number } | null {
  const sectionLine = section === 'meta' ? rule.metaLine : section === 'strings' ? rule.stringsLine : rule.conditionLine;
  if (sectionLine == null) return null;

  const sectionLines = [rule.metaLine, rule.stringsLine, rule.conditionLine].filter((l): l is number => l != null).sort((a, b) => a - b);

  const idx = sectionLines.indexOf(sectionLine);
  const nextSectionLine = idx < sectionLines.length - 1 ? sectionLines[idx + 1] : null;

  const end = nextSectionLine != null ? nextSectionLine - 1 : Infinity;
  return { start: sectionLine, end };
}

function getRuleBodyRange(rule: YaraRuleInfo): { start: number; end: number } {
  const start = rule.bodyStartLine ?? rule.nameLine + 1;
  const end = rule.bodyEndLine ?? Infinity;
  return { start, end };
}

export function generateYaraSuggestions(result: YaraParseResult): Suggestion[] {
  const suggestions: Suggestion[] = [];

  for (const rule of result.rules) {
    const metaRange = getSectionRange(rule, 'meta');
    const stringsRange = getSectionRange(rule, 'strings');
    const bodyRange = getRuleBodyRange(rule);

    if (!rule.hasMeta) {
      suggestions.push({
        line: bodyRange.start,
        lineEnd: bodyRange.end,
        field: 'section.meta',
        message: 'Consider adding a meta: section with description, author, and date',
        values: ['meta'],
      });
    } else {
      const missing = COMMON_META_KEYS.filter((key) => !rule.metaKeys.includes(key)).slice(0, 5);
      if (missing.length > 0) {
        suggestions.push({
          line: metaRange?.start ?? rule.nameLine,
          lineEnd: metaRange?.end,
          field: 'meta',
          message: 'Consider adding common meta fields',
          values: missing,
        });
      }
    }

    if (!rule.hasStrings) {
      suggestions.push({
        line: bodyRange.start,
        lineEnd: bodyRange.end,
        field: 'section.strings',
        message: 'Consider adding a strings: section to define match patterns',
        values: ['strings'],
      });
    }

    if (!rule.hasCondition) {
      suggestions.push({
        line: bodyRange.start,
        lineEnd: bodyRange.end,
        field: 'section.condition',
        message: 'A condition: section is required',
        values: ['condition'],
      });
    }

    if (rule.hasStrings && rule.stringDefs.length > 0) {
      suggestions.push({
        line: stringsRange?.start ?? rule.nameLine,
        lineEnd: stringsRange?.end,
        field: 'strings.modifiers',
        message: 'Common string modifiers',
        values: STRING_MODIFIERS,
      });
    }
  }

  if (result.rules.length > 0 && result.imports.length === 0) {
    suggestions.push({
      line: 1,
      lineEnd: result.rules[0].nameLine - 1,
      field: 'import',
      message: 'Consider importing modules for richer conditions',
      values: KNOWN_MODULES.slice(0, 6),
    });
  }

  return suggestions;
}
