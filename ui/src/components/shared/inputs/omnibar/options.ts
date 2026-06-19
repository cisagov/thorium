import { TagOptions } from '@models/tags';
import { ClauseCondition } from './ClauseTypes';

export type OmnibarOptionMap = Record<string, OmnibarCategoryOption>;

export type OmnibarCategoryOption = {
  fields: Record<string, OmnibarFieldOption>;
  helpText?: string;
  /// Whether the user may enter arbitrary field keys not present in `fields` (e.g. custom tag keys)
  fieldsCreatable?: boolean;
};

export type OmnibarFieldOption = {
  values: string[];
  conditions: ClauseCondition[];
  creatable: boolean;
  category: string;
  helpText?: string;
};

//list of fields that can have multiple entries
export const multiCategories = new Set(['tag', 'text', 'time']);

export function addGroupOptions(optMap: OmnibarOptionMap, groups: string[]): OmnibarOptionMap {
  return {
    ...optMap,
    group: {
      fields: {
        group: {
          values: groups,
          conditions: [ClauseCondition.Is, ClauseCondition.IsOneOf],
          creatable: false,
          category: 'group',
        },
      },
    },
  };
}

export function addIndexOptions(optMap: OmnibarOptionMap): OmnibarOptionMap {
  return {
    ...optMap,
    index: {
      fields: {
        index: {
          values: ['SampleResults', 'RepoResults', 'SampleTags', 'RepoTags'],
          conditions: [ClauseCondition.Is, ClauseCondition.IsOneOf],
          creatable: false,
          category: 'index',
        },
      },
    },
  };
}

export function addTextOptions(optMap: OmnibarOptionMap): OmnibarOptionMap {
  return {
    ...optMap,
    text: {
      fields: {
        text: {
          values: [],
          conditions: [ClauseCondition.Is],
          creatable: true,
          category: 'text',
          helpText: 'Search for this text',
        },
      },
      helpText: 'Search for this text',
    },
  };
}

export function addTagOptions(optMap: OmnibarOptionMap, tagOpts: TagOptions): OmnibarOptionMap {
  const newMap = {
    ...optMap,
    tag: {
      fields: {},
      helpText: 'Match a tag key/value',
      // tag keys are open-ended: allow filtering on user-entered keys not in the known set
      fieldsCreatable: true,
    },
  };

  Object.keys(tagOpts)
    .sort()
    .forEach((key) => {
      newMap.tag.fields[key] = {
        values: tagOpts[key],
        conditions: [ClauseCondition.Is],
        creatable: true,
        category: 'tag',
      };
    });
  return newMap;
}

export function addHideTagOptions(optMap: OmnibarOptionMap, tagOpts: TagOptions): OmnibarOptionMap {
  return {
    ...optMap,
    'hidden tags': {
      fields: {
        'hidden tags': {
          values: Object.keys(tagOpts).sort(),
          conditions: [ClauseCondition.Are],
          creatable: true,
          category: 'tag',
          helpText: 'Hide tags from results',
        },
      },
      helpText: 'Hide tags from results',
    },
  };
}

export function addLimitOptions(optMap: OmnibarOptionMap): OmnibarOptionMap {
  return {
    ...optMap,
    limit: {
      fields: {
        limit: {
          values: ['10', '25', '50', '100', '500', '1000'],
          conditions: [ClauseCondition.Is],
          creatable: true,
          category: 'limit',
          helpText: 'Limit for api results (default 25)',
        },
      },
      helpText: 'Limit for api results (default 25)',
    },
  };
}

type ExtraOptions = {
  category?: string;
  creatable?: boolean;
  helpText?: string;
};

export function addStringOption(
  optMap: OmnibarOptionMap,
  name: string,
  values: string[],
  conditions: ClauseCondition[],
  { category = 'text', creatable = false, helpText = '' }: ExtraOptions = {},
): OmnibarOptionMap {
  if (values.length == 0) {
    return optMap;
  }

  const fields = {};
  fields[name] = {
    values: values,
    conditions: conditions,
    creatable: creatable,
    category: category,
    helpText: helpText,
  };
  optMap[name] = {
    fields: fields,
  };
  return optMap;
}
