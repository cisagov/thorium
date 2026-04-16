import { StreamLanguage, type StringStream, LanguageSupport } from '@codemirror/language';

const KEYWORDS = new Set([
  'rule',
  'private',
  'global',
  'import',
  'include',
  'meta',
  'strings',
  'condition',
  'true',
  'false',
  'not',
  'and',
  'or',
  'at',
  'in',
  'of',
  'them',
  'for',
  'all',
  'any',
  'none',
  'entrypoint',
  'filesize',
  'matches',
  'contains',
  'startswith',
  'endswith',
  'icontains',
  'istartswith',
  'iendswith',
  'iequals',
  'defined',
  'ascii',
  'wide',
  'xor',
  'base64',
  'base64wide',
  'fullword',
  'nocase',
]);

const INT_FUNC = /^u?int(8|16|32)(be)?/;

interface YaraState {
  inBlockComment: boolean;
  inString: boolean;
  inHexString: boolean;
  afterRule: boolean;
}

function tokenize(stream: StringStream, state: YaraState): string | null {
  if (state.inBlockComment) {
    if (stream.match('*/')) {
      state.inBlockComment = false;
    } else {
      stream.next();
    }
    return 'comment';
  }

  if (state.inString) {
    while (!stream.eol()) {
      const ch = stream.next();
      if (ch === '\\') {
        stream.next();
      } else if (ch === '"') {
        state.inString = false;
        return 'string';
      }
    }
    return 'string';
  }

  if (state.inHexString) {
    if (stream.eat('}')) {
      state.inHexString = false;
      return 'string';
    }
    if (stream.match(/^\/\/[^\n]*/)) {
      return 'comment';
    }
    if (stream.match(/^\/\*/)) {
      state.inBlockComment = true;
      return 'comment';
    }
    stream.next();
    return 'string';
  }

  if (stream.eatSpace()) return null;

  if (stream.match('/*')) {
    state.inBlockComment = true;
    return 'comment';
  }

  if (stream.match(/^\/\/[^\n]*/)) {
    return 'comment';
  }

  if (stream.match(/^\/(?:[^/\\\n]|\\.)*\/[is]{0,2}/)) {
    return 'string';
  }

  if (stream.peek() === '"') {
    stream.next();
    state.inString = true;
    while (!stream.eol()) {
      const ch = stream.next();
      if (ch === '\\') {
        stream.next();
      } else if (ch === '"') {
        state.inString = false;
        return 'string';
      }
    }
    return 'string';
  }

  if (stream.peek() === '{') {
    const pos = stream.pos;
    stream.next();
    if (stream.match(/^\s*[0-9a-fA-F?[\]()|~\s-]/, false)) {
      state.inHexString = true;
      return 'string';
    }
    stream.pos = pos;
    stream.next();
    return 'punctuation';
  }

  if (stream.eat('}')) {
    return 'punctuation';
  }

  if (stream.match(/^[$#@!][a-zA-Z_][a-zA-Z0-9_]*\*?/)) {
    return 'variableName.definition';
  }

  if (stream.match(/^0x[0-9a-fA-F]+/)) return 'number';
  if (stream.match(/^0o[0-7]+/)) return 'number';
  if (stream.match(/^[0-9]+\.[0-9]+/)) return 'number';
  if (stream.match(/^[0-9]+(MB|KB)?/)) return 'number';

  if (stream.match(/^[=!<>]=|^<<|^>>|^\.\./)) return 'operator';

  if (stream.match(/^[(),:]/)) return 'punctuation';

  if (stream.match(/^[<>=]/)) return 'operator';

  if (stream.match(INT_FUNC)) return 'keyword';

  if (stream.match(/^[a-zA-Z_][a-zA-Z0-9_]*/)) {
    const word = stream.current();
    if (word === 'rule') {
      state.afterRule = true;
      return 'keyword';
    }
    if (state.afterRule) {
      state.afterRule = false;
      return 'typeName';
    }
    if (word === 'true' || word === 'false') return 'bool';
    if (KEYWORDS.has(word)) return 'keyword';
    return 'variableName';
  }

  stream.next();
  return null;
}

const yaraStreamParser = {
  startState(): YaraState {
    return {
      inBlockComment: false,
      inString: false,
      inHexString: false,
      afterRule: false,
    };
  },

  token: tokenize,

  copyState(state: YaraState): YaraState {
    return { ...state };
  },
};

const yaraLang = StreamLanguage.define(yaraStreamParser);

export function yaraLanguage(): LanguageSupport {
  return new LanguageSupport(yaraLang);
}
