import { Entities } from './entities/entities';
import { FilterTags } from './tags';

export enum FilterTypes {
  Groups = 'Groups',
  Tags = 'Tags',
  TagsCaseInsensitive = 'Case Insensitive',
  Limit = 'Limit',
  Start = 'Start',
  End = 'End',
}

export interface Filters {
  limit?: number;
  groups?: Array<string>;
  tags?: FilterTags;
  start?: string | null;
  end?: string | null;
  tags_case_insensitive?: boolean;
  hideTags?: string[];
  kinds?: Entities[];
  cursor?: string;
}

export enum Index {
  Tags = 'thorium_sample_tags',
  SampleResults = 'thorium_sample_results',
}

export enum ElasticIndex {
  SampleResults = 'SampleResults',
  RepoResults = 'RepoResults',
  SampleTags = 'SampleTags',
  RepoTags = 'RepoTags',
}

// used in client for building request params
export type SearchFilters = Omit<Filters, 'tags'> & {
  query?: string;
  indexes?: ElasticIndex[];
};
