export type Tag = {
  [value: string]: string[]; // tag values and groups who can see them
};

export type Tags = {
  [key: string]: {
    [key: string]: string[];
  };
};

// create/patch tags structure
export type RequestTags = {
  [key: string]: string[];
};

// used in TagSelect component
// instead of tag key and list of values each key/value pair has its own entry
export type TagEntry = {
  key: string;
  value: string;
};

export type TreeTags = {
  tags: { [key: string]: string[] };
};

export enum TagTypes {
  Files = 'Files',
  Repos = 'Repos',
}

export type TagKeyCounts = {
  /// The total number of items with this tag key
  total: number;
  /// The number of times each value for this tag was counted
  values: { [key: string]: number };
};

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

export type TagOptions = {
  [key: string]: string[];
};
