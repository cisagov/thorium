import { Tags } from './tags';

export type RepoCheckout = {
  /// Checkout a commit in detached mode
  Commit?: string;
  /// Checkout a branch
  Branch?: string;
  /// Checkout a tag
  Tag: string;
};

export type RepoScheme = {
  /// Use https when pulling this repo
  Https?: 'Https';
  /// Use http when pulling this repo
  Http?: 'Http';
  /// Use https when pulling this repo with auth info
  HttpsAuthed?: { username: string; password: string };
  /// Use http when pulling this repo with auth info
  HttpAuthed?: { username: string; password: string };
  /// An authenticated scheme was used but not for this user
  ScrubbedAuth?: 'ScrubbedAuth';
};

export type RepoSubmissionChunk = {
  /// The group this repo submission is visible by
  groups: string[];
  /// The unique id for this repo submission
  id: string;
  /// The user that added this repo to Thorium
  creator: string;
  /// When this repo was added to Thorium
  uploaded: string;
  // The scheme to use when cloning this repo
  scheme: RepoScheme;
  /// The earliest commit ever seen in this repo
  earliest?: string;
};

export type Repo = {
  /// Where this repo comes from (e.g. github.com)
  provider: string;
  /// The user that created this repo in the provider
  user: string;
  /// The name of this repo
  name: string;
  /// The url for this repo
  url: string;
  /// The tags for this repo
  tags: Tags;
  /// The default checkout behavior for this repo
  default_checkout?: RepoCheckout;
  /// The submissions for this repo
  submissions: RepoSubmissionChunk[];
  /// The earliest commit ever seen in this repo
  earliest?: string;
};
