// project imports
import { Entities } from './entities/entities';
import { RequestTags } from './tags';

/// The types of filters available for browsing
export enum FilterTypes {
  Groups = 'Groups',
  Tags = 'Tags',
  TagsCaseInsensitive = 'Case Insensitive',
  Limit = 'Limit',
  Start = 'Start',
  End = 'End',
}

/// Filters used for listing entities and files
export interface Filters {
  /// The max number of objects to retrieve on a single page
  limit?: number;
  /// The groups to limit our search to
  groups?: Array<string>;
  /// The tags to filter on
  tags?: RequestTags;
  /// The latest date to start listing from
  start?: string | null;
  /// The oldest date to stop listing from
  end?: string | null;
  /// Whether matching on tags should be case-insensitive
  tags_case_insensitive?: boolean;
  /// Tag keys to hide from results
  hideTags?: string[];
  /// The entity kinds to filter on
  kinds?: Entities[];
  /// The cursor to use to continue this search
  cursor?: string;
}

/// The Elasticsearch indexes available for search
export enum Index {
  Tags = 'thorium_sample_tags',
  SampleResults = 'thorium_sample_results',
}

/// The search index types
export enum ElasticIndex {
  SampleResults = 'SampleResults',
  RepoResults = 'RepoResults',
  SampleTags = 'SampleTags',
  RepoTags = 'RepoTags',
}

/// A document returned from Elasticsearch
export type ElasticDoc = {
  /// The id for this document
  id: string;
  /// The index this doc came from
  index: string;
  /// The relevance score for this doc
  score?: number;
  /// The actual document in elastic
  source?: Record<string, unknown>;
  /// Highlighted matches in elastic
  highlight?: Record<string, unknown>;
  /// The sort values for this document
  sort: number[];
};

/// Filters used in the search API client for building request params
export type SearchFilters = Omit<Filters, 'tags'> & {
  /// The search query string
  query?: string;
  /// The indexes to search
  indexes?: ElasticIndex[];
};
