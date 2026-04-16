export const SECTION_KEYWORDS = ['meta', 'strings', 'condition'] as const;

export const CONDITION_KEYWORDS = [
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
] as const;

export const STRING_MODIFIERS = ['ascii', 'wide', 'xor', 'base64', 'base64wide', 'fullword', 'nocase', 'private'] as const;

export const TEXT_STRING_MODIFIERS = new Set(['ascii', 'wide', 'xor', 'base64', 'base64wide', 'fullword', 'nocase', 'private']);

export const REGEX_STRING_MODIFIERS = new Set(['ascii', 'wide', 'nocase', 'fullword', 'private']);

export const HEX_STRING_MODIFIERS = new Set(['private']);

export const INTEGER_FUNCTIONS = [
  'int8',
  'int16',
  'int32',
  'uint8',
  'uint16',
  'uint32',
  'int8be',
  'int16be',
  'int32be',
  'uint8be',
  'uint16be',
  'uint32be',
] as const;

export const KNOWN_MODULES = [
  'pe',
  'elf',
  'cuckoo',
  'magic',
  'hash',
  'math',
  'dotnet',
  'time',
  'string',
  'console',
  'dex',
  'macho',
] as const;

export const COMMON_META_KEYS = ['description', 'author', 'reference', 'date', 'hash', 'version', 'severity', 'tlp', 'license'] as const;

export const RULE_NAME_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
