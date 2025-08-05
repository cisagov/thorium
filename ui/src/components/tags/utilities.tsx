import { Tags } from '@models';

// filter tags to only include specific tags
export const filterIncludedTags = (tags: Tags, includeList: string[]): Tags => {
  const upperIncludeList = includeList.map((tag) => {
    return tag.toUpperCase();
  });
  if (tags) {
    return Object.fromEntries(Object.entries(tags).filter(([k, v]) => upperIncludeList.includes(k.toUpperCase())));
  }
  return {};
};

// return tags without excluded values
export const filterExcludedTags = (tags: Tags, excludeList: string[]): Tags => {
  const upperExcludedList = excludeList.map((tag) => {
    return tag.toUpperCase();
  });
  return Object.fromEntries(Object.entries(tags).filter(([k, v]) => !upperExcludedList.includes(k.toUpperCase())));
};

// Lists of preformatted or categorized tags
export const FileInfoTagKeys = [
  'FileType',
  'FileTypeExtension',
  'Match',
  'FileTypeMatch',
  'Format',
  'FileFormat',
  'Compiler',
  'CompilerVersion',
  'CompilerFlags',
  'FileSize',
  'Arch',
  'Endianness',
  'PEType',
  'MachineType',
  'MIMEType',
  'EntryPoint',
  'linker',
  'packer',
  'type',
  'tool',
  'imphash',
  'detections',
  'Sign tool',
  'SignTool',
];

export const TLPLevels = ['CLEAR', 'GREEN', 'AMBER', 'AMBER+STRICT', 'RED'];

export const DangerTagKeys = [
  'SYMANTECAV',
  'CLAMAV',
  'YARARULEHITS',
  'YARAHIT',
  'SURICATASIGHIT',
  'SURICATAALERT',
  'IDSALERT',
  'PACKED',
  'CVEBINTOOLCVE',
];

// need capitalized file info keys for value checks (all keys cast to uppercase)
export const FormattedFileInfoTagKeys = FileInfoTagKeys.map((tag) => tag.toUpperCase());
