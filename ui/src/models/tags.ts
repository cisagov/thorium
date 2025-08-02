export type Tag = {
  [value: string]: string[]; // tag values and groups who can see them
};

export type Tags = {
  [key: string]: {
    [key: string]: string[];
  };
};

export type FilterTags = {
  [key: string]: string[];
};

// create tags structure, same as Filter Tags for browsing
export type CreateTags = FilterTags;

export enum TagTypes {
  Files = 'Files',
  Repos = 'Repos',
}
