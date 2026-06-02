/// A single tag's values and the groups who can see them
export type Tag = {
  [value: string]: string[];
};

/// A map of tags for a specific sample or repo
export type Tags = {
  [key: string]: {
    [value: string]: string[];
  };
};

/// Tags in request format (create/patch)
export type RequestTags = {
  [key: string]: string[];
};

/// A single tag key/value entry used in the TagSelect component
export type TagEntry = {
  /// The tag key
  key: string;
  /// The tag value
  value: string;
};

/// Tags in tree format (aggregate key/values)
export type TreeTags = {
  /// The tags in aggregate
  tags: { [key: string]: string[] };
};

/// The types of tags
export enum TagTypes {
  /// File tags
  Files = 'Files',
  /// Repo tags
  Repos = 'Repos',
}

/// The counts for a specific tag key
export type TagKeyCounts = {
  /// The total number of items with this tag key
  total: number;
  /// The number of times each value for this tag was counted
  values: { [key: string]: number };
};

/// The result of counting tags
export type TagCounts = {
  /// The id for this cursor if it can be continued
  cursor?: string;
  /// The total number of items that were counted
  total: number;
  /// The specific counts for each Tag
  tags: {
    [key: string]: TagKeyCounts;
  };
};

/// Available tag keys and their values for autocomplete
export type TagOptions = Record<string, string[]>;

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
