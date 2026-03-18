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
  'Endianess',
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

export const MitreTagKeys = ['ATT&CK', 'MBC'];

// need capitalized file info keys for value checks (all keys cast to uppercase)
export const FormattedFileInfoTagKeys = FileInfoTagKeys.map((tag) => tag.toUpperCase());
