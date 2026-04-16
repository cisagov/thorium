// Types generated from api/src/models/reactions.rs, api/src/models/jobs.rs,
// and api/src/models/git/repos.rs

// -- Job argument types (from api/src/models/jobs.rs) --

/** Options to apply to a generic job */
export interface GenericJobOpts {
  /** Whether to always override all positional args in the original image */
  override_positionals: boolean;
  /** Whether to always override all keyword args in the original image */
  override_kwargs: boolean;
  /** The cmd to override the original cmd from the image with in its entirety */
  override_cmd?: string[];
}

/** Keyword arguments mapping keys to arrays of string values */
export type GenericJobKwargs = Record<string, string[]>;

/** Arguments for a generic job */
export interface GenericJobArgs {
  /** The positional arguments to overlay onto the original cmd */
  positionals: string[];
  /** The keyword arguments to overlay onto the original cmd */
  kwargs: GenericJobKwargs;
  /** The switch arguments to overlay onto the original cmd */
  switches: string[];
  /** The options to apply to this generic job */
  opts: GenericJobOpts;
}

/** The arguments for all images in a reaction */
export type ReactionArgs = Record<string, GenericJobArgs>;

// -- Repo dependency types (from api/src/models/git/repos.rs, commits.rs) --

/** The kind of commitish reference */
export enum CommitishKinds {
  /** A commit */
  Commit = 'Commit',
  /** A branch */
  Branch = 'Branch',
  /** A tag */
  Tag = 'Tag',
}

/** A repo dependency request for a reaction */
export interface RepoDependencyRequest {
  /** The url to the repo to download */
  url: string;
  /** The branch, commit, or tag to checkout */
  commitish?: string;
  /** The kind of commitish to use for checkout */
  kind?: CommitishKinds;
}

/** A specific repo/commit to download before executing a job */
export interface RepoDependency {
  /** The url to the repo to download */
  url: string;
  /** The commitish to checkout */
  commitish?: string;
  /** The kind of commitish to use for checkout */
  kind?: CommitishKinds;
}

// -- Reaction types (from api/src/models/reactions.rs) --

/** The current status of a reaction */
export enum ReactionStatus {
  /** This reaction is created, but is not yet running */
  Created = 'Created',
  /** At least one stage of this reaction has started */
  Started = 'Started',
  /** This reaction has completed */
  Completed = 'Completed',
  /** This reaction has failed due to an error */
  Failed = 'Failed',
}

/** A response containing the reaction id */
export interface ReactionIdResponse {
  /** The uuidv4 of a reaction */
  id: string;
}

/** The response from creating reactions in bulk */
export interface BulkReactionResponse {
  /** Any errors that occurred while creating reactions */
  errors: Record<number, string>;
  /** The successfully created reactions */
  created: string[];
}

/** A response for handling the reaction command */
export interface HandleReactionResponse {
  /** The status of the executed command */
  status: ReactionStatus;
}

/** A request to create a new reaction */
export interface ReactionRequest {
  /** The group the reaction is in */
  group: string;
  /** The pipeline this reaction is built around */
  pipeline: string;
  /** The args to overlay on top of the args for images in this reaction */
  args: ReactionArgs;
  /** The number of seconds we have to meet this reaction's SLA */
  sla?: number;
  /** The tags this reaction can be listed under */
  tags: string[];
  /** The parent reaction to set if this is a sub reaction */
  parent?: string;
  /** A list of sample sha256s to download before executing this reaction's jobs */
  samples: string[];
  /** A map of ephemeral buffers to download before executing this reaction's jobs */
  buffers: Record<string, string>;
  /** Any repos to download before executing this reaction's jobs */
  repos: RepoDependencyRequest[];
  /** This reaction's depth in triggers if this reaction was caused by a trigger */
  trigger_depth?: number;
  /** Any initial cache for this reaction */
  cache: ReactionCache;
}

/** A reaction built around a pipeline, used to track jobs across a single run */
export interface Reaction {
  /** The uuidv4 that identifies this reaction */
  id: string;
  /** The group this reaction is in */
  group: string;
  /** The creator of this reaction */
  creator: string;
  /** The pipeline this reaction is built around */
  pipeline: string;
  /** The current status of this reaction */
  status: ReactionStatus;
  /** The current stage of this reaction */
  current_stage: number;
  /** The current stage's progress */
  current_stage_progress: number;
  /** The current stage's length */
  current_stage_length: number;
  /** The args for this reaction (passed to all jobs) */
  args: ReactionArgs;
  /** The timestamp this reaction's SLA expires at */
  sla: string;
  /** The uuidv4s of the jobs in this pipeline */
  jobs: string[];
  /** The tags this reaction can be listed under */
  tags: string[];
  /** The parent reaction for this reaction if it's a sub reaction */
  parent?: string;
  /** The number of subreactions for this reaction */
  sub_reactions: number;
  /** The number of completed subreactions for this reaction */
  completed_sub_reactions: number;
  /** The job id for any generators this reaction currently has active */
  generators: string[];
  /** A list of sample sha256s to download before executing this reaction's jobs */
  samples: string[];
  /** A list of ephemeral files to download */
  ephemeral: string[];
  /** A list of ephemeral files from any parent reactions and what parent reaction it's tied to */
  parent_ephemeral: Record<string, string>;
  /** A list of repos to download before executing this reaction's jobs */
  repos: RepoDependency[];
  /** This reaction's depth in triggers if this reaction was caused by a trigger */
  trigger_depth?: number;
  /** Whether this reaction has any cache data set */
  has_cache: boolean;
}

/** Cached data for a reaction */
export interface ReactionCache {
  /** A generic key/value cache of info across this reaction */
  generic: Record<string, string>;
  /** Files in this reaction cache */
  files: string[];
}

/** A non-file update to a reaction cache */
export interface ReactionCacheUpdate {
  /** A generic key/value cache of info across this reaction */
  generic: Record<string, string>;
  /** The cache keys to remove */
  remove_generic: string[];
}

/**
 * A client-side reaction selection used when submitting pipelines for a file upload.
 * This is the shape built by `buildReactionsList` in the upload flow and sent to
 * `createReaction` after adding `samples`.
 */
export interface ReactionSelection {
  /** The pipeline to run */
  pipeline: string;
  /** The group the pipeline belongs to */
  group: string;
  /** Arguments to pass to images in the pipeline */
  args: ReactionArgs;
  /** The number of seconds for SLA */
  sla: number;
  /** Optional tags for the reaction */
  tags?: Record<string, string[]>;
}
