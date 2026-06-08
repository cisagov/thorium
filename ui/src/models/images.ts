// project imports
import { SemVer } from './semver';
import { Volume } from './volumes';
import { OutputDisplayType, OutputCollection } from './results';

export type ImageVersion = {
  SemVer?: SemVer;
  Custom?: string;
};

export type ImageScaler = 'K8s' | 'BareMetal' | 'Windows' | 'Kvm' | 'External';

type ImageLifetime = {
  counter: string;
  amount: number;
};

export type Image = {
  /// The group this image is in
  group: string;
  /// The name of this image
  name: string;
  /// The creator of this image
  creator: string;
  /// The version of this image
  version?: ImageVersion;
  /// What scaler is responsible for scaling this image
  scaler: ImageScaler;
  /// The image to use (url or tag)
  image?: string;
  /// The lifetime of a pod
  lifetime?: ImageLifetime;
  /// The timeout for individual jobs
  timeout?: number;
  /// The resources to required to spawn this image
  resources: Resources;
  /// The limit to use for how many workers of this image type can be spawned
  spawn_limit: SpawnLimits;
  /// The environment variables to set
  env: Record<string, string | null>;
  /// How long this image takes to execute on average in seconds (defaults to
  /// 10 minutes on image creation).
  runtime: number;
  /// Any volumes to bind in to this container
  volumes: Volume[];
  /// The arguments to add to this images jobs
  args: ImageArgs;
  /// The path to the modifier folders for this image
  modifiers?: string;
  /// The image description
  description?: string;
  /// The security context for this image
  security_context: SecurityContext;
  /// The pipelines that are using this image
  used_by: string[];
  /// Whether the agent should stream stdout/stderr back to Thorium
  collect_logs: boolean;
  /// Whether this is a generator or not
  generator: boolean;
  /// How to handle dependencies for this image
  dependencies: Dependencies;
  /// The type of display class to use in the UI for this images output
  display_type: OutputDisplayType;
  /// The settings for collecting results from this image
  output_collection: OutputCollection;
  /// Any regex filters to match on when uploading children
  ///
  /// If no filters are given, all children will be uploaded. Regular expressions
  /// must conform to standards according to the
  /// [regex crate](https://docs.rs/regex/latest/regex/) or an error will be
  /// returned on image creation/update.
  child_filters: ChildFilters;
  /// The settings to use when cleaning up canceled jobs
  clean_up?: any; //TODO:
  /// The settings to use for Kvm jobs
  kvm?: any; //TODO:
  /// A list of reasons an image is banned mapped by ban UUID;
  /// if the list has any bans, the image cannot be spawned
  bans: Record<string, ImageBan>;
  /// A set of the names of network policies to apply to the image when it's spawned
  ///
  /// Only applies when scaled with K8's currently
  network_policies: Set<string>;
};

export type Resources = {
  /// The total amount of cpu in millicpu
  cpu: number;
  /// The total amount of ram in mebibytes
  memory: number;
  /// The total amount of ephemeral storage in mebibytes
  ephemeral_storage: number;
  /// The number of available worker slots if its applicable
  worker_slots: number;
  /// The total number of Nvidia GPUs
  nvidia_gpu: number;
  /// The total number of AMD GPUs
  amd_gpu: number;
  /// The amount of resources to allow an image to burst with
  burstable: BurstableResources;
};

export type BurstableResources = {
  /// The total amount of cpu in millicpu
  cpu: number;
  /// The total amount of ram in mebibytes
  memory: number;
};

export type SpawnLimits = { Basic: number } | 'Unlimited';

type ImageArgs = {
  /// The entrypoint to force all jobs to use
  entrypoint?: string[];
  /// The command to force all jobs to use
  command?: string[];
  /// What kwarg to pass the current reaction id in with
  reaction?: string;
  /// What kwarg to pass the repo url in as
  repo?: string;
  /// What kwarg to pass the repo commit in with
  commit?: string;
  /// What kwarg pass the result location as
  output: ArgStrategy;
  /// What kwarg pass the result files location as
  output_files: ArgStrategy;
};

//FIXME: not sure if this matches. check
export type ArgStrategy = 'None' | 'Append' | { Kwarg: string };

type SecurityContext = {
  /// The user to run as
  user?: number;
  /// The group to use
  group: number;
  /// Allow users to escalate their privileges
  allow_privilege_escalation: boolean;
};

type Dependencies = {
  /// The settings  the agent should use when passing donwloaded samples to tools
  samples: SampleDependencySettings;
  /// The settings the agent should use when passing donwloaded ephemeral files to tools
  ephemeral: EphemeralDependencySettings;
  /// The settings the agent should use when passing prior results to tools
  results: ResultDependencySettings;
  /// The settings the agent should use when passing prior repos to tools
  repos: RepoDependencySettings;
  /// The settings the agent should use when passing tags to tools
  tags: TagDependencySettings;
  /// The settings the agent should use when passing children files from past tools
  children: ChildrenDependencySettings;
  /// The settings to use when getting cache info
  cache: CacheDependencySettings;
};

type SampleDependencySettings = {
  /// Where the agent should store downloaded files
  location: string;
  /// The kwarg to pass these samples in with if one is set (otherwise use positional args)
  kwarg?: string;
  /// The strategy the agent should use when passing samples downloaded to jobs
  strategy: DependencyPassStrategy;
  /// The strategy to when naming any downloaded files
  naming: FileNamingStrategy;
};

//FIXME: check
type DependencyPassStrategy = 'Paths' | 'Names' | 'Directory' | 'Disabled';

type FileNamingStrategy = 'Sha256' | 'MostRecent';

type EphemeralDependencySettings = {
  /// Where the agent should store downloaded ephemeral files
  location: string;
  /// The kwarg to pass these files in with if one is set (otherwise use positional args)
  kwarg?: string;
  /// The strategy the agent should use when passing dependencies downloaded to jobs
  strategy: DependencyPassStrategy;
  /// Any files to limit this image to downloading
  names: string[];
};

type ResultDependencySettings = {
  /// The prior images to collect results from
  images: string[];
  /// Where the agent should store downloaded prior result files
  location: string;
  /// The kwarg to pass these files in with if one is set (otherwise use positional args)
  kwarg: KwargDependency;
  /// The strategy the agent should use when passing dependencies downloaded to jobs
  strategy: DependencyPassStrategy;
  /// Any files to limit this image to downloading
  names: string[];
};

type KwargDependency = { List: string } | { Map: Record<string, string> } | 'None';

type RepoDependencySettings = {
  /// Where the agent should store downloaded repos
  location: string;
  /// The kwarg to pass these repos in with if one is set (otherwise use positional args)
  kwarg?: string;
  /// The strategy the agent should use when passing repos downloaded to jobs
  strategy: DependencyPassStrategy;
};

type TagDependencySettings = {
  /// Whether this job wants tags to be downloaded or not
  enabled: boolean;
  /// Where the agent should store downloaded tags
  location: string;
  /// The kwarg to pass these tags in with if one is set (otherwise use positional args)
  kwarg?: string;
  /// The strategy the agent should use when passing tags downloaded to jobs
  strategy: DependencyPassStrategy;
};

type ChildrenDependencySettings = {
  //// Whether children dependencies should be enabled or not
  enabled: boolean;
  /// The prior images to restrict children collection too
  images: string[];
  /// Where the agent should store downloaded childrens
  location: string;
  /// The kwarg to pass these childrens in with if one is set (otherwise use positional args)
  kwarg?: string;
  /// The strategy the agent should use when passing childrens downloaded to jobs
  strategy: DependencyPassStrategy;
};

type CacheDependencySettings = {
  /// The location to write our generic cache too
  location: string;
  /// The settings to use for the generic cache
  generic: GenericCacheDependencySettings;
  /// Whether to use our parents cache if we have one or not
  use_parent_cache: boolean;
  /// Whether cache is enabled for this image
  enabled: boolean;
};

type GenericCacheDependencySettings = {
  /// The kwarg to pass this cache in with if one is set (otherwise use positional args)
  kwarg?: string;
  /// The strategy the agent should use when passing the downloaded cache to jobs
  strategy: DependencyPassStrategy;
};

type ChildFilters = {
  /// Any filters to apply to the MIME type
  mime: string[];
  /// Any filters to apply to the file name (including the extension)
  file_name: string[];
  /// Any filters to apply to the file extension, not including the dot
  /// (e.g. "txt", "so", "exe", etc.)
  file_extension: string[];
  /// Submit children that do *not* match any of the given filters rather
  /// than ones that do match
  submit_non_matches: boolean;
};

type ImageBan = {
  /// The unique id for this ban
  id: string;
  /// The time in UTC that the ban was made
  time_banned: any;
  /// The kind of ban this is
  ban_kind: any;
};
