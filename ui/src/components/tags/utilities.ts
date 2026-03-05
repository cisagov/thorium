// project imports
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

export enum TagUpperKeyEnum {
  TLP = 'TLP',
  RESULTS = 'RESULTS',
  ATTACK = 'ATT&CK',
  MBC = 'MBC',
}

export enum TagValueEnum {
  RED = 'RED',
  AMBER = 'AMBER',
  AMBER_STRICT = 'AMBER+STRICT',
  GREEN = 'GREEN',
  WHITE = 'WHITE',
  CLEAR = 'CLEAR',
}

export function getTagColorClass(key: string, value: string): string {
  const upperKey = key.toUpperCase() as TagUpperKeyEnum;
  if (upperKey == TagUpperKeyEnum.TLP) {
    switch (value.toUpperCase() as TagValueEnum) {
      case TagValueEnum.RED:
        return 'tlp-red-btn';
      case TagValueEnum.AMBER:
        return 'tlp-amber-btn';
      case TagValueEnum.AMBER_STRICT:
        return 'tlp-amber-btn';
      case TagValueEnum.GREEN:
        return 'tlp-green-btn';
      case TagValueEnum.WHITE:
        return 'tlp-clear-btn';
      case TagValueEnum.CLEAR:
        return 'tlp-clear-btn';
    }
  } else if (upperKey == TagUpperKeyEnum.RESULTS) {
    return 'general-tag';
  } else if (upperKey == TagUpperKeyEnum.ATTACK) {
    return 'attack-tag';
  } else if (upperKey == TagUpperKeyEnum.MBC) {
    return 'mbc-tag';
  } else if (FormattedFileInfoTagKeys.includes(upperKey)) {
    return 'info-tag';
  } else if (DangerTagKeys.includes(upperKey)) {
    return 'danger-tag';
  }
  return 'other-tag';
}

export function getTagBadgeText(key: string, value: string, condensed: boolean): string {
  const upperTag = key.toUpperCase() as TagUpperKeyEnum;
  switch (upperTag) {
    case TagUpperKeyEnum.TLP:
      if (!condensed) return value.toUpperCase();
      return `TLP: ${value.toUpperCase()}`;
    case TagUpperKeyEnum.ATTACK:
    case TagUpperKeyEnum.MBC:
      return `${value}`;
    case TagUpperKeyEnum.RESULTS:
    default:
      return `${key}: ${value}`;
  }
}
