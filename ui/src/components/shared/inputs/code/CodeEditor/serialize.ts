import { stringify as yamlStringify } from 'yaml';
import { FormatType } from '@utilities/rules/types';

export function toText(value: unknown, format: FormatType): string {
  if (value === null || value === undefined) {
    return '';
  }

  switch (format) {
    case FormatType.YAML:
      return yamlStringify(value, { indent: 4, lineWidth: 0 });
    case FormatType.JSON:
      return JSON.stringify(value, null, 2);
    case FormatType.YARA:
      return typeof value === 'string' ? value : '';
    default:
      return JSON.stringify(value);
  }
}
