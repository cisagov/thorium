// project imports
import { RULE_NAME_PATTERN } from './schema';

const META_RE = /^meta\s*:/;
const STRINGS_RE = /^strings\s*:/;
const CONDITION_RE = /^condition\s*:/;
const IMPORT_RE = /^import\s+"([^"]+)"/;
const RULE_RE = /^(private\s+)?(global\s+)?rule\s+([a-zA-Z_]\w*)(?:\s*:\s*(.+?))?\s*(\{)?\s*$/;
const SECTION_RE = /^([a-zA-Z_]\w*)\s*:/;
const META_ENTRY_RE = /^([a-zA-Z_]\w*)\s*=/;
const STRING_DEF_RE = /^(\$[a-zA-Z_]\w*)\s*=\s*(.*)/;

export interface YaraImport {
  module: string;
  line: number;
  moduleColumn: number;
}

export enum YaraStringType {
  Text = 'text',
  Regex = 'regex',
  Hex = 'hex',
}

export interface YaraStringDef {
  id: string;
  line: number;
  column: number;
  type: YaraStringType;
  modifiers: string[];
  modifierPositions: Array<{ modifier: string; column: number }>;
}

export interface YaraConditionRef {
  ref: string;
  line: number;
  column: number;
}

export interface YaraRuleInfo {
  name: string;
  nameLine: number;
  nameColumn: number;
  tags: string[];
  tagPositions: Array<{ tag: string; column: number; line: number }>;
  isPrivate: boolean;
  isGlobal: boolean;
  hasMeta: boolean;
  metaLine: number | null;
  metaColumn: number | null;
  metaKeys: string[];
  metaKeyPositions: Array<{ key: string; line: number; column: number }>;
  hasStrings: boolean;
  stringsLine: number | null;
  stringsColumn: number | null;
  stringDefs: YaraStringDef[];
  stringIds: string[];
  hasCondition: boolean;
  conditionLine: number | null;
  conditionColumn: number | null;
  conditionText: string;
  conditionRefs: YaraConditionRef[];
  conditionLines: Array<{ text: string; line: number }>;
  unknownSections: Array<{ name: string; line: number; column: number }>;
  duplicateSections: Array<{ name: string; line: number; column: number; prevLine: number; prevColumn: number }>;
  bodyStartLine: number | null;
  bodyEndLine: number | null;
}

export interface YaraParseResult {
  imports: YaraImport[];
  rules: YaraRuleInfo[];
  errors: Array<{ line: number; message: string }>;
}

function stripBlockComments(text: string): string {
  let result = '';
  let i = 0;
  let inBlock = false;
  while (i < text.length) {
    if (inBlock) {
      if (text[i] === '*' && text[i + 1] === '/') {
        inBlock = false;
        result += '  ';
        i += 2;
      } else {
        result += text[i] === '\n' ? '\n' : ' ';
        i++;
      }
    } else {
      if (text[i] === '/' && text[i + 1] === '*') {
        inBlock = true;
        result += '  ';
        i += 2;
      } else {
        result += text[i];
        i++;
      }
    }
  }
  return result;
}

function stripLineComments(line: string): string {
  let inStr = false;
  for (let i = 0; i < line.length; i++) {
    if (inStr) {
      if (line[i] === '\\') {
        i++;
      } else if (line[i] === '"') {
        inStr = false;
      }
    } else {
      if (line[i] === '"') {
        inStr = true;
      } else if (line[i] === '/' && line[i + 1] === '/') {
        return line.slice(0, i);
      }
    }
  }
  return line;
}

function detectStringType(valuePart: string): YaraStringType {
  const v = valuePart.trim();
  if (v.startsWith('{')) return YaraStringType.Hex;
  if (v.startsWith('/')) return YaraStringType.Regex;
  return YaraStringType.Text;
}

function parseModifiers(afterValue: string, lineColumn: number): Array<{ modifier: string; column: number }> {
  const results: Array<{ modifier: string; column: number }> = [];
  const modRe = /\b(ascii|wide|xor|base64wide|base64|fullword|nocase|private)\b/g;
  let m;
  while ((m = modRe.exec(afterValue)) !== null) {
    results.push({ modifier: m[1], column: lineColumn + m.index });
  }
  return results;
}

function extractConditionRefs(condLines: Array<{ text: string; line: number }>): YaraConditionRef[] {
  const refs: YaraConditionRef[] = [];
  const refRe = /[$#@!][a-zA-Z_]\w*\*?/g;
  for (const cl of condLines) {
    let m;
    while ((m = refRe.exec(cl.text)) !== null) {
      refs.push({ ref: m[0], line: cl.line, column: m.index + 1 });
    }
  }
  return refs;
}

export function parseYaraText(text: string): YaraParseResult {
  const imports: YaraImport[] = [];
  const rules: YaraRuleInfo[] = [];
  const errors: Array<{ line: number; message: string }> = [];

  const cleaned = stripBlockComments(text);
  const lines = cleaned.split('\n');

  let currentRule: YaraRuleInfo | null = null;
  let braceDepth = 0;
  let inHexString = false;
  let currentSection: 'none' | 'meta' | 'strings' | 'condition' = 'none';
  let conditionLines: Array<{ text: string; line: number }> = [];

  for (let i = 0; i < lines.length; i++) {
    const lineNum = i + 1;
    const raw = stripLineComments(lines[i]);
    const trimmed = raw.trim();

    if (!trimmed) continue;

    const importMatch = trimmed.match(IMPORT_RE);
    if (importMatch && braceDepth === 0) {
      const moduleCol = raw.indexOf('"') + 2;
      imports.push({ module: importMatch[1], line: lineNum, moduleColumn: moduleCol });
      continue;
    }

    if (braceDepth === 0) {
      const ruleMatch = trimmed.match(RULE_RE);
      if (ruleMatch) {
        if (!RULE_NAME_PATTERN.test(ruleMatch[3])) {
          errors.push({ line: lineNum, message: `Invalid rule name: '${ruleMatch[3]}'` });
        }

        const nameCol = raw.indexOf(ruleMatch[3]) + 1;
        const tagPositions: Array<{ tag: string; column: number; line: number }> = [];
        if (ruleMatch[4]) {
          const tagsStr = ruleMatch[4].trim();
          const colonIdx = raw.indexOf(':');
          const tagsBase = colonIdx + 1;
          const tagRe = /\S+/g;
          let tm;
          while ((tm = tagRe.exec(tagsStr)) !== null) {
            const absCol = raw.indexOf(tm[0], tagsBase) + 1;
            tagPositions.push({ tag: tm[0], column: absCol, line: lineNum });
          }
        }

        currentRule = {
          name: ruleMatch[3],
          nameLine: lineNum,
          nameColumn: nameCol,
          tags: ruleMatch[4] ? ruleMatch[4].trim().split(/\s+/) : [],
          tagPositions,
          isPrivate: !!ruleMatch[1],
          isGlobal: !!ruleMatch[2],
          hasMeta: false,
          metaLine: null,
          metaColumn: null,
          metaKeys: [],
          metaKeyPositions: [],
          hasStrings: false,
          stringsLine: null,
          stringsColumn: null,
          stringDefs: [],
          stringIds: [],
          hasCondition: false,
          conditionLine: null,
          conditionColumn: null,
          conditionText: '',
          conditionRefs: [],
          conditionLines: [],
          unknownSections: [],
          duplicateSections: [],
          bodyStartLine: null,
          bodyEndLine: null,
        };

        if (ruleMatch[5] === '{') {
          braceDepth = 1;
          currentSection = 'none';
          currentRule.bodyStartLine = lineNum;
        }
        continue;
      }
    }

    if (trimmed === '{' && currentRule && braceDepth === 0) {
      braceDepth = 1;
      currentSection = 'none';
      currentRule.bodyStartLine = lineNum;
      continue;
    }

    if (braceDepth > 0 && currentRule) {
      if (currentSection === 'strings') {
        for (let ci = 0; ci < trimmed.length; ci++) {
          if (inHexString) {
            if (trimmed[ci] === '}') {
              inHexString = false;
            }
          } else if (trimmed[ci] === '{') {
            const rest = trimmed.slice(ci + 1);
            if (/^\s*([0-9a-fA-F?[\]()|~\s-]|$)/.test(rest)) {
              inHexString = true;
            } else {
              braceDepth++;
            }
          } else if (trimmed[ci] === '}') {
            braceDepth--;
          }
        }
      } else {
        for (const ch of trimmed) {
          if (ch === '{') braceDepth++;
          if (ch === '}') braceDepth--;
        }
      }

      if (braceDepth === 0) {
        if (currentSection === 'condition') {
          currentRule.conditionText = conditionLines
            .map((cl) => cl.text)
            .join(' ')
            .trim();
          currentRule.conditionRefs = extractConditionRefs(conditionLines);
          currentRule.conditionLines = conditionLines;
          conditionLines = [];
        }
        currentRule.bodyEndLine = lineNum;
        inHexString = false;
        rules.push(currentRule);
        currentRule = null;
        currentSection = 'none';
        continue;
      }

      if (META_RE.test(trimmed)) {
        if (currentSection === 'condition') {
          currentRule.conditionText = conditionLines
            .map((cl) => cl.text)
            .join(' ')
            .trim();
          currentRule.conditionRefs = extractConditionRefs(conditionLines);
          currentRule.conditionLines = conditionLines;
          conditionLines = [];
        }
        const col = raw.indexOf('meta') + 1;
        if (currentRule.hasMeta) {
          currentRule.duplicateSections.push({
            name: 'meta',
            line: lineNum,
            column: col,
            prevLine: currentRule.metaLine!,
            prevColumn: currentRule.metaColumn!,
          });
        }
        currentSection = 'meta';
        currentRule.hasMeta = true;
        currentRule.metaLine = lineNum;
        currentRule.metaColumn = col;
        continue;
      }

      if (STRINGS_RE.test(trimmed)) {
        if (currentSection === 'condition') {
          currentRule.conditionText = conditionLines
            .map((cl) => cl.text)
            .join(' ')
            .trim();
          currentRule.conditionRefs = extractConditionRefs(conditionLines);
          currentRule.conditionLines = conditionLines;
          conditionLines = [];
        }
        const col = raw.indexOf('strings') + 1;
        if (currentRule.hasStrings) {
          currentRule.duplicateSections.push({
            name: 'strings',
            line: lineNum,
            column: col,
            prevLine: currentRule.stringsLine!,
            prevColumn: currentRule.stringsColumn!,
          });
        }
        currentSection = 'strings';
        currentRule.hasStrings = true;
        currentRule.stringsLine = lineNum;
        currentRule.stringsColumn = col;
        continue;
      }

      if (CONDITION_RE.test(trimmed)) {
        if (currentSection === 'condition') {
          currentRule.conditionText = conditionLines
            .map((cl) => cl.text)
            .join(' ')
            .trim();
          currentRule.conditionRefs = extractConditionRefs(conditionLines);
          currentRule.conditionLines = conditionLines;
          conditionLines = [];
        }
        const col = raw.indexOf('condition') + 1;
        if (currentRule.hasCondition) {
          currentRule.duplicateSections.push({
            name: 'condition',
            line: lineNum,
            column: col,
            prevLine: currentRule.conditionLine!,
            prevColumn: currentRule.conditionColumn!,
          });
        }
        currentSection = 'condition';
        currentRule.hasCondition = true;
        currentRule.conditionColumn = col;
        currentRule.conditionLine = lineNum;
        const afterColon = trimmed.slice(trimmed.indexOf(':') + 1).trim();
        if (afterColon) {
          const colonIdx = raw.indexOf(':');
          conditionLines.push({ text: afterColon, line: lineNum });
          // Adjust column offset: find where the afterColon starts in raw line
          const afterColonStart = raw.indexOf(afterColon, colonIdx + 1);
          if (afterColonStart >= 0) {
            conditionLines[conditionLines.length - 1] = {
              text: raw.slice(afterColonStart),
              line: lineNum,
            };
          }
        }
        continue;
      }

      if (braceDepth === 1) {
        const sectionMatch = trimmed.match(SECTION_RE);
        if (sectionMatch) {
          const name = sectionMatch[1];
          const col = raw.indexOf(name) + 1;
          currentRule.unknownSections.push({ name, line: lineNum, column: col });
        }
      }

      if (currentSection === 'meta') {
        const metaMatch = trimmed.match(META_ENTRY_RE);
        if (metaMatch) {
          const key = metaMatch[1];
          const col = raw.indexOf(key) + 1;
          currentRule.metaKeys.push(key);
          currentRule.metaKeyPositions.push({ key, line: lineNum, column: col });
        }
      }

      if (currentSection === 'strings') {
        const stringMatch = trimmed.match(STRING_DEF_RE);
        if (stringMatch) {
          const id = stringMatch[1];
          const valuePart = stringMatch[2];
          const col = raw.indexOf(id) + 1;
          const strType = detectStringType(valuePart);

          let afterValue = '';
          if (strType === 'text') {
            const closeQuote = valuePart.indexOf('"', 1);
            if (closeQuote >= 0) afterValue = valuePart.slice(closeQuote + 1);
          } else if (strType === 'regex') {
            const lastSlash = valuePart.lastIndexOf('/');
            if (lastSlash > 0) afterValue = valuePart.slice(lastSlash + 1);
          } else if (strType === 'hex') {
            const closeBrace = valuePart.lastIndexOf('}');
            if (closeBrace >= 0) afterValue = valuePart.slice(closeBrace + 1);
          }

          const eqIdx = raw.indexOf('=', col);
          const valueStartCol = eqIdx + 2;
          const modPositions = parseModifiers(afterValue, valueStartCol + (valuePart.length - afterValue.length));

          const def: YaraStringDef = {
            id,
            line: lineNum,
            column: col,
            type: strType,
            modifiers: modPositions.map((mp) => mp.modifier),
            modifierPositions: modPositions,
          };
          currentRule.stringDefs.push(def);
          currentRule.stringIds.push(id);
        }
      }

      if (currentSection === 'condition') {
        conditionLines.push({ text: raw, line: lineNum });
      }
    }
  }

  if (currentRule) {
    if (currentSection === 'condition') {
      currentRule.conditionText = conditionLines
        .map((cl) => cl.text)
        .join(' ')
        .trim();
      currentRule.conditionRefs = extractConditionRefs(conditionLines);
      currentRule.conditionLines = conditionLines;
    }
    errors.push({ line: currentRule.nameLine, message: `Rule '${currentRule.name}' has unclosed braces` });
    rules.push(currentRule);
  }

  return { imports, rules, errors };
}
