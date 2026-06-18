/**
 * Type-guard variant of `Array.includes` that narrows the searched string to the array's element type.
 *
 * Lets callers test membership in a `readonly` string-literal tuple and have TypeScript treat the
 * value as that literal union afterwards.
 *
 * @template T - The string-literal element type of the array.
 * @param arr - The array of allowed values.
 * @param val - The string to test for membership.
 * @returns `true` (narrowing `val` to `T`) if `val` is in `arr`, otherwise `false`.
 */
export function includes<T extends string>(arr: readonly T[], val: string): val is T {
  return (arr as readonly string[]).includes(val);
}

export enum Severity {
  Error = 'error',
  Warning = 'warning',
  Info = 'info',
}

export interface Diagnostic {
  line: number;
  column?: number;
  endLine?: number;
  endColumn?: number;
  severity: Severity;
  message: string;
}

export enum FieldValueType {
  String = 'string',
  Number = 'number',
  Boolean = 'boolean',
  Enum = 'enum',
  Object = 'object',
  StringArray = 'string_array',
}

export interface UnitOption {
  label: string;
  suffix: string;
}

export interface FieldSchema {
  type: FieldValueType;
  required?: boolean;
  placeholder?: string;
  description?: string;
  enumValues?: readonly string[];
  fields?: Record<string, FieldSchema>;
  unit?: { options: readonly UnitOption[]; defaultUnit: string };
  transform?: (value: string) => { yaml: string; json: string; valid: boolean; error?: string };
  typeName?: string;
  variants?: Record<string, FieldSchema | null>;
  /**
   * For an Object schema whose shape varies by a discriminator field (e.g. a Volume's
   * `archetype`): `field` names the discriminator and `fieldMap` maps each discriminator
   * value to the sibling object field holding that variant's config (e.g.
   * `{ HostPath: 'host_path', NFS: 'nfs' }`). Lets a preview render only the active config.
   */
  variantField?: { field: string; fieldMap: Record<string, string> };
  /**
   * Marks a `StringArray` field as a list of *stages*, where each stage is a parallel group of
   * strings (serialized as `Vec<Vec<String>>`, e.g. pipeline `order`). Switches the populate
   * widget to the stage editor and the insert builders to grouped output.
   */
  nestedList?: boolean;
  /**
   * Current value of a `nestedList` field, attached per-suggestion so the stage editor can seed
   * itself with the existing stages (letting the user add to an existing pipeline `order` rather
   * than overwriting it). Each inner array is one stage's parallel images.
   */
  currentStages?: readonly (readonly string[])[];
}

export interface Suggestion {
  line: number;
  lineEnd?: number;
  field: string;
  message: string;
  values?: readonly string[];
  isList?: boolean;
  isMapEntry?: boolean;
  isRemoval?: boolean;
  isReplace?: boolean;
  schema?: FieldSchema;
  category?: string;
}

export interface CheckResult {
  diagnostics: Diagnostic[];
  suggestions: Suggestion[];
}

export enum FormatType {
  YAML = 'yaml',
  JSON = 'json',
  YARA = 'yara',
}

export interface RuleChecker {
  format: FormatType;
  check(text: string): CheckResult;
}
