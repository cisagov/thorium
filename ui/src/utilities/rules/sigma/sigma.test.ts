import { describe, test, expect } from 'vitest';
import { SigmaRuleChecker } from './index';

const VALID_OKTA_RULE = `title: Okta User Account Locked Out
id: 14701da0-4b0f-4ee6-9c95-2ffb4e73bb9a
status: test
description: Detects when a user account is locked out.
references:
    - https://developer.okta.com/docs/reference/api/system-log/
    - https://developer.okta.com/docs/reference/api/event-types/
author: Austin Songer @austinsonger
date: 2021-09-12
modified: 2022-10-09
tags:
    - attack.impact
logsource:
    product: okta
    service: okta
detection:
    selection:
        displaymessage: Max sign in attempts exceeded
    condition: selection
falsepositives:
    - Unknown
level: medium`;

const checker = new SigmaRuleChecker();

function errors(text: string) {
  return checker.check(text).diagnostics.filter((d) => d.severity === 'error');
}

function warnings(text: string) {
  return checker.check(text).diagnostics.filter((d) => d.severity === 'warning');
}

function suggestions(text: string) {
  return checker.check(text).suggestions;
}

function removeLine(text: string, prefix: string): string {
  return text
    .split('\n')
    .filter((line) => !line.trimStart().startsWith(prefix))
    .join('\n');
}

function removeBlock(text: string, key: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let skipping = false;
  for (const line of lines) {
    if (line.startsWith(`${key}:`)) {
      skipping = true;
      continue;
    }
    if (skipping && (line.startsWith('    ') || line.startsWith('\t') || line.trim() === '')) {
      continue;
    }
    skipping = false;
    result.push(line);
  }
  return result.join('\n');
}

function replaceLine(text: string, prefix: string, replacement: string): string {
  return text
    .split('\n')
    .map((line) => (line.trimStart().startsWith(prefix) ? replacement : line))
    .join('\n');
}

describe('SigmaRuleChecker', () => {
  describe('valid rule', () => {
    test('produces no errors or warnings for valid Okta rule', () => {
      const result = checker.check(VALID_OKTA_RULE);
      const errs = result.diagnostics.filter((d) => d.severity === 'error');
      const warns = result.diagnostics.filter((d) => d.severity === 'warning');
      expect(errs).toHaveLength(0);
      expect(warns).toHaveLength(0);
    });

    test('produces suggestions for missing optional fields', () => {
      const s = suggestions(VALID_OKTA_RULE);
      expect(s.length).toBeGreaterThan(0);
      const fields = s.map((sg) => sg.field);
      expect(fields).toContain('logsource.category');
      expect(fields).not.toContain('title');
      expect(fields).not.toContain('tags');
    });
  });

  describe('empty and invalid input', () => {
    test('empty string returns no diagnostics', () => {
      const result = checker.check('');
      expect(result.diagnostics).toHaveLength(0);
      expect(result.suggestions).toHaveLength(0);
    });

    test('whitespace-only string returns no diagnostics', () => {
      const result = checker.check('   \n\n  ');
      expect(result.diagnostics).toHaveLength(0);
    });

    test('invalid YAML syntax returns syntax error', () => {
      const result = checker.check('title: [unclosed');
      const errs = result.diagnostics.filter((d) => d.severity === 'error');
      expect(errs.length).toBeGreaterThan(0);
    });

    test('non-mapping YAML returns error', () => {
      const result = checker.check('- item1\n- item2');
      const errs = result.diagnostics.filter((d) => d.severity === 'error');
      expect(errs.some((e) => e.message.includes('mapping'))).toBe(true);
    });
  });

  describe('required field errors', () => {
    test('missing title', () => {
      const text = removeLine(VALID_OKTA_RULE, 'title:');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'title'"))).toBe(true);
    });

    test('missing logsource', () => {
      const text = removeBlock(VALID_OKTA_RULE, 'logsource');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'logsource'"))).toBe(true);
    });

    test('missing detection', () => {
      const text = removeBlock(VALID_OKTA_RULE, 'detection');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'detection'"))).toBe(true);
    });

    test('missing condition in detection', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'condition:', '');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("detection is missing required field: 'condition'"))).toBe(true);
    });
  });

  describe('enum validation', () => {
    test('invalid status value', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'status:', 'status: invalid');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Invalid status value: 'invalid'"))).toBe(true);
    });

    test('invalid level value', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'level:', 'level: extreme');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Invalid level value: 'extreme'"))).toBe(true);
    });
  });

  describe('format validation', () => {
    test('invalid date format', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'date:', "date: 'not-a-date'");
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes('Invalid date format'))).toBe(true);
    });

    test('valid date passes', () => {
      const errs = errors(VALID_OKTA_RULE);
      expect(errs.filter((e) => e.message.includes('date')).length).toBe(0);
    });

    test('invalid id (not UUID v4)', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'id:', "id: 'not-a-uuid'");
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes('UUID v4'))).toBe(true);
    });
  });

  describe('unknown fields and tags', () => {
    test('unknown top-level field', () => {
      const text = VALID_OKTA_RULE + '\nfoobar: baz';
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes("Unknown Sigma field: 'foobar'"))).toBe(true);
    });

    test('invalid tag pattern (uppercase)', () => {
      const text = replaceLine(VALID_OKTA_RULE, '- attack.impact', '    - ATTACK.IMPACT');
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes('does not match expected pattern'))).toBe(true);
    });
  });

  describe('detection validation', () => {
    test('unknown value modifier', () => {
      const text = VALID_OKTA_RULE.replace('displaymessage: Max sign in attempts exceeded', 'Image|badmod: foo');
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Unknown value modifier 'badmod'"))).toBe(true);
    });

    test('condition references undefined identifier', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'condition:', '    condition: nonexistent');
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes("'nonexistent'") && w.message.includes('not defined'))).toBe(true);
    });
  });

  describe('related entry validation', () => {
    test('related entry missing id and type', () => {
      const text = VALID_OKTA_RULE + '\nrelated:\n    - foo: bar';
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("related entry is missing required field: 'id'"))).toBe(true);
      expect(errs.some((e) => e.message.includes("related entry is missing required field: 'type'"))).toBe(true);
    });

    test('related entry with invalid type', () => {
      const text = VALID_OKTA_RULE + '\nrelated:\n    - id: abc\n      type: badtype';
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Invalid related type: 'badtype'"))).toBe(true);
    });
  });

  describe('suggestions', () => {
    test('empty status suggests valid values', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'status:', 'status:');
      const s = suggestions(text);
      const statusSugg = s.find((sg) => sg.field === 'status');
      expect(statusSugg).toBeDefined();
      expect(statusSugg!.values).toContain('stable');
      expect(statusSugg!.values).toContain('test');
    });

    test('empty level suggests valid values', () => {
      const text = replaceLine(VALID_OKTA_RULE, 'level:', 'level:');
      const s = suggestions(text);
      const levelSugg = s.find((sg) => sg.field === 'level');
      expect(levelSugg).toBeDefined();
      expect(levelSugg!.values).toContain('high');
      expect(levelSugg!.values).toContain('critical');
    });

    test('missing logsource.category suggests common categories', () => {
      const s = suggestions(VALID_OKTA_RULE);
      const catSugg = s.find((sg) => sg.field === 'logsource.category');
      expect(catSugg).toBeDefined();
      expect(catSugg!.values).toContain('process_creation');
    });

    test('missing optional fields are suggested', () => {
      const ruleNoOptionals = `title: Test
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;
      const s = suggestions(ruleNoOptionals);
      const fields = s.map((sg) => sg.field);
      expect(fields).toContain('id');
      expect(fields).toContain('description');
      expect(fields).toContain('author');
      expect(fields).toContain('date');
      expect(fields).toContain('level');
      expect(fields).toContain('status');
      expect(fields).toContain('tags');
      expect(fields).toContain('falsepositives');
      expect(fields).toContain('references');
    });

    test('missing level suggestion includes valid values', () => {
      const ruleNoLevel = removeLine(VALID_OKTA_RULE, 'level:');
      const s = suggestions(ruleNoLevel);
      const levelSugg = s.find((sg) => sg.field === 'level');
      expect(levelSugg).toBeDefined();
      expect(levelSugg!.values).toContain('informational');
      expect(levelSugg!.values).toContain('critical');
    });

    test('missing status suggestion includes valid values', () => {
      const ruleNoStatus = removeLine(VALID_OKTA_RULE, 'status:');
      const s = suggestions(ruleNoStatus);
      const statusSugg = s.find((sg) => sg.field === 'status');
      expect(statusSugg).toBeDefined();
      expect(statusSugg!.values).toContain('stable');
      expect(statusSugg!.values).toContain('experimental');
    });
  });

  describe('duplicate key detection', () => {
    test('duplicate top-level key flags both occurrences', () => {
      const text = `title: First Title
title: Second Title
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes("Duplicate key 'title'"));
      expect(dupErrs).toHaveLength(2);
      expect(dupErrs[0].line).toBe(1);
      expect(dupErrs[0].endColumn).toBe(1 + 'title'.length);
      expect(dupErrs[1].line).toBe(2);
      expect(dupErrs[1].endColumn).toBe(1 + 'title'.length);
    });

    test('duplicate key in nested map flags both occurrences', () => {
      const text = `title: Test
logsource:
    product: windows
    product: linux
detection:
    sel:
        Image: test
    condition: sel`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes("Duplicate key 'product'"));
      expect(dupErrs).toHaveLength(2);
    });

    test('duplicate key in detection search identifier flags both occurrences', () => {
      const text = `title: Test
logsource:
    product: windows
detection:
    sel:
        Image: foo
        Image: bar
    condition: sel`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes("Duplicate key 'Image'"));
      expect(dupErrs).toHaveLength(2);
    });

    test('same key in different list entries is not a duplicate', () => {
      const text = `title: Test
logsource:
    product: windows
detection:
    sel:
        Image: test
    condition: sel
related:
    - id: abc
      type: derived
    - id: def
      type: similar`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes('Duplicate key'));
      expect(dupErrs).toHaveLength(0);
    });
  });
});
