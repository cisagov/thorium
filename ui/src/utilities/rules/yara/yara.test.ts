import { describe, test, expect } from 'vitest';

// project imports
import { YaraRuleChecker } from './index';

const VALID_RULE = `import "pe"

rule DetectUPX : packer compression
{
    meta:
        description = "Detects UPX packed executables"
        author = "Test Author"
        date = "2024-01-15"

    strings:
        $upx0 = "UPX0" ascii
        $upx1 = "UPX1" ascii
        $hex = { 60 E8 00 00 00 00 58 }

    condition:
        uint16(0) == 0x5A4D and ($upx0 or $upx1) and $hex
}`;

const checker = new YaraRuleChecker();

function errors(text: string) {
  return checker.check(text).diagnostics.filter((d) => d.severity === 'error');
}

function warnings(text: string) {
  return checker.check(text).diagnostics.filter((d) => d.severity === 'warning');
}

function suggestions(text: string) {
  return checker.check(text).suggestions;
}

describe('YaraRuleChecker', () => {
  describe('valid rule', () => {
    test('produces no errors for valid rule', () => {
      const errs = errors(VALID_RULE);
      expect(errs).toHaveLength(0);
    });

    test('produces no warnings for valid rule with known module', () => {
      const warns = warnings(VALID_RULE);
      expect(warns).toHaveLength(0);
    });

    test('produces suggestions for common meta keys', () => {
      const s = suggestions(VALID_RULE);
      expect(s.length).toBeGreaterThan(0);
    });
  });

  describe('empty and whitespace input', () => {
    test('empty string returns no diagnostics', () => {
      const result = checker.check('');
      expect(result.diagnostics).toHaveLength(0);
      expect(result.suggestions).toHaveLength(0);
    });

    test('whitespace-only string returns no diagnostics', () => {
      const result = checker.check('   \n\n  ');
      expect(result.diagnostics).toHaveLength(0);
    });
  });

  describe('missing condition section', () => {
    test('rule without condition produces error', () => {
      const text = `rule NoCondition
{
    strings:
        $a = "test"
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("missing required 'condition:'"))).toBe(true);
    });
  });

  describe('empty condition', () => {
    test('rule with empty condition produces error', () => {
      const text = `rule EmptyCondition
{
    condition:
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes('empty condition'))).toBe(true);
    });
  });

  describe('undefined string reference', () => {
    test('condition referencing undefined string produces error', () => {
      const text = `rule UndefinedRef
{
    strings:
        $a = "test"
    condition:
        $a and $b
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("'$b'") && e.message.includes('not defined'))).toBe(true);
    });
  });

  describe('condition references strings without strings section', () => {
    test('produces error when condition uses strings but no strings section', () => {
      const text = `rule NoStringsSection
{
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("no 'strings:' section"))).toBe(true);
    });
  });

  describe('unknown import module', () => {
    test('unknown module produces warning', () => {
      const text = `import "badmodule"

rule Test
{
    condition:
        true
}`;
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes("Unknown YARA module: 'badmodule'"))).toBe(true);
    });
  });

  describe('duplicate rule name', () => {
    test('duplicate names produce warning', () => {
      const text = `rule Duplicate
{
    condition:
        true
}

rule Duplicate
{
    condition:
        false
}`;
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes("Duplicate rule name: 'Duplicate'"))).toBe(true);
    });
  });

  describe('unused string definition', () => {
    test('unused string produces warning', () => {
      const text = `rule UnusedString
{
    strings:
        $used = "used"
        $unused = "unused"
    condition:
        $used
}`;
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes("'$unused'") && w.message.includes('never referenced'))).toBe(true);
    });

    test('no warning when "them" is used in condition', () => {
      const text = `rule UsesAll
{
    strings:
        $a = "one"
        $b = "two"
    condition:
        any of them
}`;
      const warns = warnings(text);
      expect(warns.filter((w) => w.message.includes('never referenced'))).toHaveLength(0);
    });
  });

  describe('multi-rule file', () => {
    test('validates each rule independently', () => {
      const text = `rule Valid
{
    condition:
        true
}

rule Invalid
{
    strings:
        $a = "test"
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("'Invalid'") && e.message.includes('condition'))).toBe(true);
      expect(errs.filter((e) => e.message.includes("'Valid'"))).toHaveLength(0);
    });
  });

  describe('comment handling', () => {
    test('block comments do not interfere with parsing', () => {
      const text = `/* this is a comment */
rule Commented
{
    /* meta section commented out */
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });

    test('line comments do not interfere with parsing', () => {
      const text = `// import "pe"
rule Commented
{
    condition:
        true // always match
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });
  });

  describe('private and global rules', () => {
    test('private rule parses correctly', () => {
      const text = `private rule PrivateRule
{
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });

    test('global rule parses correctly', () => {
      const text = `global rule GlobalRule
{
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });
  });

  describe('suggestions', () => {
    test('missing meta section suggests adding one', () => {
      const text = `rule NoMeta
{
    condition:
        true
}`;
      const s = suggestions(text);
      expect(s.some((sg) => sg.field === 'section.meta')).toBe(true);
    });

    test('missing strings section suggests adding one', () => {
      const text = `rule NoStrings
{
    condition:
        true
}`;
      const s = suggestions(text);
      expect(s.some((sg) => sg.field === 'section.strings')).toBe(true);
    });

    test('missing condition section suggests adding one', () => {
      const text = `rule NoCondition
{
    meta:
        author = "test"
}`;
      const s = suggestions(text);
      expect(s.some((sg) => sg.field === 'section.condition')).toBe(true);
    });

    test('no section suggestions when all sections present', () => {
      const text = `rule AllSections
{
    meta:
        author = "test"
    strings:
        $a = "test"
    condition:
        $a
}`;
      const s = suggestions(text);
      expect(s.filter((sg) => sg.field.startsWith('section.'))).toHaveLength(0);
    });

    test('meta section with missing keys suggests common keys', () => {
      const text = `rule PartialMeta
{
    meta:
        description = "test"
    condition:
        true
}`;
      const s = suggestions(text);
      const metaSugg = s.find((sg) => sg.field === 'meta');
      expect(metaSugg).toBeDefined();
      expect(metaSugg!.values).toContain('author');
    });

    test('no imports suggests modules', () => {
      const text = `rule NoImports
{
    condition:
        true
}`;
      const s = suggestions(text);
      expect(s.some((sg) => sg.field === 'import')).toBe(true);
    });

    test('string modifiers suggested when strings section exists', () => {
      const text = `rule WithStrings
{
    strings:
        $a = "test"
    condition:
        $a
}`;
      const s = suggestions(text);
      expect(s.some((sg) => sg.field === 'strings.modifiers')).toBe(true);
    });
  });

  describe('same-line opening brace', () => {
    test('rule with brace on same line parses correctly', () => {
      const text = `rule InlineBrace {
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });

    test('rule with brace on same line and tags parses correctly', () => {
      const text = `rule Tagged : mytag {
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });
  });

  describe('unknown section keyword', () => {
    test('unknown section produces error', () => {
      const text = `rule BadSection
{
    metadata:
        description = "test"
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Unknown section 'metadata:'"))).toBe(true);
    });

    test('valid sections produce no unknown section error', () => {
      const text = `rule AllSections
{
    meta:
        author = "test"
    strings:
        $a = "test"
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs.filter((e) => e.message.includes('Unknown section'))).toHaveLength(0);
    });
  });

  describe('hex string braces do not break parser', () => {
    test('hex strings with braces inside strings section parse correctly', () => {
      const text = `rule HexTest
{
    strings:
        $h1 = { 60 E8 00 00 00 00 58 }
        $h2 = { AA BB CC }
    condition:
        $h1 or $h2
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });

    test('multiple hex strings do not cause unclosed brace error', () => {
      const text = `rule MultiHex {
    strings:
        $a = { 00 11 22 }
        $b = { FF EE DD }
        $c = "text string"
    condition:
        any of them
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });
  });

  describe('duplicate tag detection', () => {
    test('duplicate tags produce error', () => {
      const text = `rule DupTag : foo bar foo
{
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Duplicate tag: 'foo'"))).toBe(true);
    });

    test('unique tags produce no error', () => {
      const text = `rule UniqueTags : alpha beta gamma
{
    condition:
        true
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });
  });

  describe('duplicate modifier detection', () => {
    test('duplicate modifier on text string produces error', () => {
      const text = `rule DupMod
{
    strings:
        $a = "test" ascii ascii
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("Duplicate modifier 'ascii'"))).toBe(true);
    });
  });

  describe('modifier-per-string-type validation', () => {
    test('nocase on regex is allowed', () => {
      const text = `rule RegexNocase
{
    strings:
        $a = /test[0-9]+/ nocase
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });

    test('xor on regex produces error', () => {
      const text = `rule RegexXor
{
    strings:
        $a = /test/ xor
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("'xor'") && e.message.includes('not valid for regular expression'))).toBe(true);
    });

    test('ascii on hex string produces error', () => {
      const text = `rule HexAscii
{
    strings:
        $a = { AA BB } ascii
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs.some((e) => e.message.includes("'ascii'") && e.message.includes('not valid for hex string'))).toBe(true);
    });

    test('private on hex string is allowed', () => {
      const text = `rule HexPrivate
{
    strings:
        $a = { AA BB } private
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });

    test('private on text string is allowed', () => {
      const text = `rule TextPrivate
{
    strings:
        $a = "test" private
    condition:
        $a
}`;
      const errs = errors(text);
      expect(errs).toHaveLength(0);
    });
  });

  describe('entrypoint deprecation', () => {
    test('entrypoint in condition produces deprecation warning', () => {
      const text = `rule UsesEntrypoint
{
    condition:
        entrypoint == 0x1000
}`;
      const warns = warnings(text);
      expect(warns.some((w) => w.message.includes("'entrypoint' is deprecated"))).toBe(true);
    });
  });

  describe('precise error positioning', () => {
    test('undefined string ref error points to the ref in condition', () => {
      const text = `rule PreciseRef
{
    strings:
        $a = "test"
    condition:
        $a and $b
}`;
      const errs = errors(text);
      const refErr = errs.find((e) => e.message.includes("'$b'"));
      expect(refErr).toBeDefined();
      expect(refErr!.line).toBe(6);
      expect(refErr!.column).toBeGreaterThan(1);
    });

    test('unused string warning points to the string definition', () => {
      const text = `rule PreciseUnused
{
    strings:
        $used = "used"
        $unused = "unused"
    condition:
        $used
}`;
      const warns = warnings(text);
      const unusedWarn = warns.find((w) => w.message.includes("'$unused'"));
      expect(unusedWarn).toBeDefined();
      expect(unusedWarn!.line).toBe(5);
      expect(unusedWarn!.column).toBeGreaterThan(1);
    });

    test('duplicate tag error points to the duplicate tag', () => {
      const text = `rule DupTagPos : alpha alpha
{
    condition:
        true
}`;
      const errs = errors(text);
      const dupErr = errs.find((e) => e.message.includes("Duplicate tag: 'alpha'"));
      expect(dupErr).toBeDefined();
      expect(dupErr!.line).toBe(1);
      expect(dupErr!.column).toBeGreaterThan(1);
    });

    test('unknown module warning points to the module name', () => {
      const text = `import "badmodule"

rule Test
{
    condition:
        true
}`;
      const warns = warnings(text);
      const modWarn = warns.find((w) => w.message.includes("'badmodule'"));
      expect(modWarn).toBeDefined();
      expect(modWarn!.column).toBeGreaterThan(1);
    });

    test('no strings section error points to the ref in condition', () => {
      const text = `rule NoStrings
{
    condition:
        $a and $b
}`;
      const errs = errors(text);
      const refErr = errs.find((e) => e.message.includes("'$a'") && e.message.includes("no 'strings:' section"));
      expect(refErr).toBeDefined();
      expect(refErr!.line).toBe(4);
    });
  });

  describe('duplicate key detection', () => {
    test('duplicate meta keys flag both occurrences', () => {
      const text = `rule DupMeta
{
    meta:
        description = "First"
        author = "Test"
        description = "Second"

    condition:
        true
}`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes("Duplicate meta key 'description'"));
      expect(dupErrs).toHaveLength(2);
      expect(dupErrs[0].line).toBe(4);
      expect(dupErrs[1].line).toBe(6);
    });

    test('duplicate string IDs flag both occurrences', () => {
      const text = `rule DupStrings
{
    strings:
        $a = "first"
        $b = "second"
        $a = "third"

    condition:
        $a or $b
}`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes("Duplicate string identifier '$a'"));
      expect(dupErrs).toHaveLength(2);
      expect(dupErrs[0].line).toBe(4);
      expect(dupErrs[1].line).toBe(6);
    });

    test('duplicate sections produce error', () => {
      const text = `rule DupSections
{
    strings:
        $a = "first"

    strings:
        $b = "second"

    condition:
        $a or $b
}`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes("Duplicate section 'strings:'"));
      expect(dupErrs).toHaveLength(2);
      expect(dupErrs[0].line).toBe(3);
      expect(dupErrs[0].message).toContain('also defined on line 6');
      expect(dupErrs[1].line).toBe(6);
      expect(dupErrs[1].message).toContain('previously defined on line 3');
    });

    test('duplicate condition section produces error', () => {
      const text = `rule DupCond
{
    condition:
        true

    condition:
        false
}`;
      const errs = errors(text);
      const dupErr = errs.find((e) => e.message.includes("Duplicate section 'condition:'"));
      expect(dupErr).toBeDefined();
    });

    test('same key names in different rules are not duplicates', () => {
      const text = `rule Rule1
{
    meta:
        description = "First rule"

    condition:
        true
}

rule Rule2
{
    meta:
        description = "Second rule"

    condition:
        true
}`;
      const errs = errors(text);
      const dupErrs = errs.filter((e) => e.message.includes('Duplicate'));
      expect(dupErrs).toHaveLength(0);
    });
  });
});
