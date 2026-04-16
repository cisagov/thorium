// project imports
import { Severity, includes, type Diagnostic } from '../types';
import type { YaraParseResult } from './parse';
import { KNOWN_MODULES, RULE_NAME_PATTERN, TEXT_STRING_MODIFIERS, REGEX_STRING_MODIFIERS, HEX_STRING_MODIFIERS } from './schema';

const MODIFIER_SETS = {
  text: TEXT_STRING_MODIFIERS,
  regex: REGEX_STRING_MODIFIERS,
  hex: HEX_STRING_MODIFIERS,
} as const;

const TYPE_LABELS = {
  text: 'text string',
  regex: 'regular expression',
  hex: 'hex string',
} as const;

export function validateYaraRules(result: YaraParseResult): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];

  for (const imp of result.imports) {
    if (!includes(KNOWN_MODULES, imp.module)) {
      diagnostics.push({
        line: imp.line,
        column: imp.moduleColumn,
        endColumn: imp.moduleColumn + imp.module.length,
        severity: Severity.Warning,
        message: `Unknown YARA module: '${imp.module}'. Known modules: ${KNOWN_MODULES.join(', ')}`,
      });
    }
  }

  const ruleNames = new Map<string, { line: number; column: number; flagged: boolean }>();
  for (const rule of result.rules) {
    if (!RULE_NAME_PATTERN.test(rule.name)) {
      diagnostics.push({
        line: rule.nameLine,
        column: rule.nameColumn,
        endColumn: rule.nameColumn + rule.name.length,
        severity: Severity.Error,
        message: `Invalid rule name: '${rule.name}'. Must start with a letter or underscore and contain only alphanumerics and underscores`,
      });
    }

    const prevRule = ruleNames.get(rule.name);
    if (prevRule) {
      if (!prevRule.flagged) {
        diagnostics.push({
          line: prevRule.line,
          column: prevRule.column,
          endColumn: prevRule.column + rule.name.length,
          severity: Severity.Warning,
          message: `Duplicate rule name: '${rule.name}' (also defined on line ${rule.nameLine})`,
        });
        prevRule.flagged = true;
      }
      diagnostics.push({
        line: rule.nameLine,
        column: rule.nameColumn,
        endColumn: rule.nameColumn + rule.name.length,
        severity: Severity.Warning,
        message: `Duplicate rule name: '${rule.name}' (previously defined on line ${prevRule.line})`,
      });
    } else {
      ruleNames.set(rule.name, { line: rule.nameLine, column: rule.nameColumn, flagged: false });
    }

    const seenTags = new Map<string, { line: number; column: number; flagged: boolean }>();
    for (const tp of rule.tagPositions) {
      const prev = seenTags.get(tp.tag);
      if (prev) {
        if (!prev.flagged) {
          diagnostics.push({
            line: prev.line,
            column: prev.column,
            endColumn: prev.column + tp.tag.length,
            severity: Severity.Error,
            message: `Duplicate tag: '${tp.tag}' (also specified on line ${tp.line})`,
          });
          prev.flagged = true;
        }
        diagnostics.push({
          line: tp.line,
          column: tp.column,
          endColumn: tp.column + tp.tag.length,
          severity: Severity.Error,
          message: `Duplicate tag: '${tp.tag}' (already specified on line ${prev.line})`,
        });
      } else {
        seenTags.set(tp.tag, { line: tp.line, column: tp.column, flagged: false });
      }
    }

    for (const us of rule.unknownSections) {
      diagnostics.push({
        line: us.line,
        column: us.column,
        endColumn: us.column + us.name.length,
        severity: Severity.Error,
        message: `Unknown section '${us.name}:'. Valid sections are: meta, strings, condition`,
      });
    }

    const flaggedSections = new Set<string>();
    for (const ds of rule.duplicateSections) {
      if (!flaggedSections.has(ds.name)) {
        diagnostics.push({
          line: ds.prevLine,
          column: ds.prevColumn,
          endColumn: ds.prevColumn + ds.name.length,
          severity: Severity.Error,
          message: `Duplicate section '${ds.name}:' (also defined on line ${ds.line})`,
        });
        flaggedSections.add(ds.name);
      }
      diagnostics.push({
        line: ds.line,
        column: ds.column,
        endColumn: ds.column + ds.name.length,
        severity: Severity.Error,
        message: `Duplicate section '${ds.name}:' (previously defined on line ${ds.prevLine})`,
      });
    }

    if (!rule.hasCondition) {
      diagnostics.push({
        line: rule.nameLine,
        column: rule.nameColumn,
        endColumn: rule.nameColumn + rule.name.length,
        severity: Severity.Error,
        message: `Rule '${rule.name}' is missing required 'condition:' section`,
      });
    }

    if (rule.hasCondition && !rule.conditionText) {
      diagnostics.push({
        line: rule.conditionLine ?? rule.nameLine,
        column: 1,
        severity: Severity.Error,
        message: `Rule '${rule.name}' has an empty condition`,
      });
    }

    const seenMetaKeys = new Map<string, { line: number; column: number; flagged: boolean }>();
    for (const mk of rule.metaKeyPositions) {
      const prev = seenMetaKeys.get(mk.key);
      if (prev) {
        if (!prev.flagged) {
          diagnostics.push({
            line: prev.line,
            column: prev.column,
            endColumn: prev.column + mk.key.length,
            severity: Severity.Error,
            message: `Duplicate meta key '${mk.key}' (also defined on line ${mk.line})`,
          });
          prev.flagged = true;
        }
        diagnostics.push({
          line: mk.line,
          column: mk.column,
          endColumn: mk.column + mk.key.length,
          severity: Severity.Error,
          message: `Duplicate meta key '${mk.key}' (previously defined on line ${prev.line})`,
        });
      } else {
        seenMetaKeys.set(mk.key, { line: mk.line, column: mk.column, flagged: false });
      }
    }

    const seenStringIds = new Map<string, { line: number; column: number; flagged: boolean }>();
    for (const sd of rule.stringDefs) {
      const prev = seenStringIds.get(sd.id);
      if (prev) {
        if (!prev.flagged) {
          diagnostics.push({
            line: prev.line,
            column: prev.column,
            endColumn: prev.column + sd.id.length,
            severity: Severity.Error,
            message: `Duplicate string identifier '${sd.id}' (also defined on line ${sd.line})`,
          });
          prev.flagged = true;
        }
        diagnostics.push({
          line: sd.line,
          column: sd.column,
          endColumn: sd.column + sd.id.length,
          severity: Severity.Error,
          message: `Duplicate string identifier '${sd.id}' (previously defined on line ${prev.line})`,
        });
      } else {
        seenStringIds.set(sd.id, { line: sd.line, column: sd.column, flagged: false });
      }
    }

    if (rule.hasStrings && rule.stringDefs.length === 0) {
      diagnostics.push({
        line: rule.stringsLine ?? rule.nameLine,
        column: 1,
        severity: Severity.Warning,
        message: `Rule '${rule.name}' has a 'strings:' section with no string definitions`,
      });
    }

    let stringsHasError = false;
    for (const sd of rule.stringDefs) {
      const allowed = MODIFIER_SETS[sd.type];
      const seenModifiers = new Set<string>();
      for (const mp of sd.modifierPositions) {
        if (seenModifiers.has(mp.modifier)) {
          stringsHasError = true;
          diagnostics.push({
            line: sd.line,
            column: mp.column,
            endColumn: mp.column + mp.modifier.length,
            severity: Severity.Error,
            message: `Duplicate modifier '${mp.modifier}' on string '${sd.id}'`,
          });
        }
        seenModifiers.add(mp.modifier);

        if (!allowed.has(mp.modifier)) {
          stringsHasError = true;
          diagnostics.push({
            line: sd.line,
            column: mp.column,
            endColumn: mp.column + mp.modifier.length,
            severity: Severity.Error,
            message: `Modifier '${mp.modifier}' is not valid for ${TYPE_LABELS[sd.type]}s (${sd.id}). Allowed: ${[...allowed].join(', ')}`,
          });
        }
      }
    }

    if (stringsHasError && rule.stringsLine != null) {
      const col = rule.stringsColumn ?? 1;
      diagnostics.push({
        line: rule.stringsLine,
        column: col,
        endColumn: col + 'strings'.length,
        severity: Severity.Error,
        message: `Section contains errors`,
      });
    }

    let conditionHasError = false;
    if (rule.hasCondition && rule.conditionText) {
      const usesWildcard = rule.conditionRefs.some((r) => r.ref.endsWith('*'));
      const usesThem = /\bthem\b/.test(rule.conditionText);
      const usesEntrypoint = /\bentrypoint\b/.test(rule.conditionText);

      if (usesEntrypoint) {
        for (const cl of rule.conditionLines) {
          const epRe = /\bentrypoint\b/g;
          let m;
          while ((m = epRe.exec(cl.text)) !== null) {
            diagnostics.push({
              line: cl.line,
              column: m.index + 1,
              endColumn: m.index + 1 + 'entrypoint'.length,
              severity: Severity.Warning,
              message: `'entrypoint' is deprecated. Use 'pe.entry_point' or 'elf.entry_point' instead`,
            });
          }
        }
      }

      for (const ref of rule.conditionRefs) {
        if (ref.ref.endsWith('*')) continue;
        const normalizedRef = '$' + ref.ref.slice(1);
        if (rule.hasStrings && !rule.stringIds.includes(normalizedRef)) {
          conditionHasError = true;
          diagnostics.push({
            line: ref.line,
            column: ref.column,
            endColumn: ref.column + ref.ref.length,
            severity: Severity.Error,
            message: `String '${ref.ref}' referenced in condition but not defined in strings section`,
          });
        }
      }

      if (!usesWildcard && !usesThem && rule.hasStrings) {
        const referencedIds = new Set(rule.conditionRefs.filter((r) => !r.ref.endsWith('*')).map((r) => '$' + r.ref.slice(1)));
        for (const sd of rule.stringDefs) {
          if (!referencedIds.has(sd.id)) {
            diagnostics.push({
              line: sd.line,
              column: sd.column,
              endColumn: sd.column + sd.id.length,
              severity: Severity.Warning,
              message: `String '${sd.id}' is defined but never referenced in condition`,
            });
          }
        }
      }

      const condStringRefs = rule.conditionRefs.filter((r) => r.ref.startsWith('$'));
      if (condStringRefs.length > 0 && !rule.hasStrings) {
        conditionHasError = true;
        for (const ref of condStringRefs) {
          diagnostics.push({
            line: ref.line,
            column: ref.column,
            endColumn: ref.column + ref.ref.length,
            severity: Severity.Error,
            message: `Condition references '${ref.ref}' but rule '${rule.name}' has no 'strings:' section`,
          });
        }
      }
    }

    if (conditionHasError && rule.conditionLine != null) {
      const col = rule.conditionColumn ?? 1;
      diagnostics.push({
        line: rule.conditionLine,
        column: col,
        endColumn: col + 'condition'.length,
        severity: Severity.Error,
        message: `Section contains errors`,
      });
    }
  }

  return diagnostics;
}
