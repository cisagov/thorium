import { TagOptions } from '@models/tags';
import { ClauseCondition } from './ClauseTypes';

// spec: ./SPEC.md

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
export const multiCategories = new Set(['tag', 'text', 'time', 'Show', 'Hide', 'Exclude', 'Include']);

/** The entity-layer action categories (a lexicon for controlling which entity types render/traverse). */
export const ENTITY_LAYER_CATEGORIES = ['Show', 'Hide', 'Exclude', 'Include'] as const;

/**
 * Add the four entity-layer action categories (`Show`/`Hide`/`Exclude`/`Include`) to an option map, each
 * offering the given entity-type values (raw enum values so clause values round-trip to policy keys).
 * These replace per-kind visibility toggles: Show=render+explore, Hide=pass-through, Exclude=prune,
 * Include=whitelist.
 *
 * @param optMap - The option map to extend.
 * @param kinds - Entity-type enum values present in the tree.
 * @returns The extended option map (unchanged when there are no kinds).
 */
export function addEntityLayerOptions(optMap: OmnibarOptionMap, kinds: string[]): OmnibarOptionMap {
  if (kinds.length === 0) return optMap;
  const next = { ...optMap };
  for (const category of ENTITY_LAYER_CATEGORIES) {
    next[category] = {
      fields: {
        [category]: {
          values: kinds,
          conditions: [ClauseCondition.Is, ClauseCondition.IsOneOf],
          creatable: false,
          category,
          helpText: `${category} these entity types`,
        },
      },
      helpText: `${category} entity types`,
    };
  }
  return next;
}

/**
 * Add a traversal-`depth` category (replaces the API `limit` field for the graph browser). The value is the
 * current depth being searched/filtered; raising it additively grows the shared graph.
 *
 * @param optMap - The option map to extend.
 * @param maxDepth - The largest selectable depth.
 * @returns The extended option map.
 */
export function addDepthOptions(optMap: OmnibarOptionMap, maxDepth: number = 10): OmnibarOptionMap {
  return {
    ...optMap,
    depth: {
      fields: {
        depth: {
          values: Array.from({ length: maxDepth }, (_, i) => String(i + 1)),
          conditions: [ClauseCondition.Is],
          creatable: true,
          category: 'depth',
          helpText: 'Traversal depth to search/filter',
        },
      },
      helpText: 'Traversal depth',
    },
  };
}

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
        // IsOneOf lets is-one-of tag clauses (merged same-key dashboard filters) render/edit in the omnibar
        values: tagOpts[key],
        conditions: [ClauseCondition.Is, ClauseCondition.IsOneOf],
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
