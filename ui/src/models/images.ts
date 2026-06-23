import { SemVer } from './semver';
import { OutputCollection, OutputDisplayType } from './results';
import { Volume } from './volumes';

export type ImageVersion = {
  SemVer?: SemVer;
  Custom?: string;
};

export enum ImageScaler {
  K8s = 'K8s',
  BareMetal = 'BareMetal',
  Windows = 'Windows',
  Kvm = 'Kvm',
  External = 'External',
}

export enum ArgStrategy {
  None = 'None',
  Append = 'Append',
}

export type ArgStrategyKwarg = {
  Kwarg: string;
};

export type ArgStrategyValue = ArgStrategy | ArgStrategyKwarg;

export type SpawnLimitsBasic = {
  Basic: number;
};

export type SpawnLimitsValue = 'Unlimited' | SpawnLimitsBasic;

export enum DependencyPassStrategy {
  Paths = 'Paths',
  Names = 'Names',
  Directory = 'Directory',
  Disabled = 'Disabled',
}

export enum FileNamingStrategy {
  Sha256 = 'Sha256',
  MostRecent = 'MostRecent',
}

export enum OutputHandler {
  Files = 'Files',
}

export enum AutoTagLogic {
  Exists = 'Exists',
}

export type AutoTagLogicValue =
  | AutoTagLogic
  | { Equal: unknown }
  | { Not: unknown }
  | { Greater: unknown }
  | { GreaterOrEqual: unknown }
  | { LesserOrEqual: unknown }
  | { Lesser: unknown }
  | { In: unknown[] }
  | { NotIn: unknown[] };

/// The requested burstable resources to spawn the container with
export type BurstableResourcesRequest = {
  /// Cpu cores in millicpus
  cpu: number;
  /// Ram in mebibytes
  memory: number;
};

/// The requested resources to spawn the container with
export type ResourcesRequest = {
  /// Cpu cores in millicpus
  cpu: number;
  /// Ram in mebibytes
  memory: number;
  /// Ephemeral storage in mebibytes
  ephemeral_storage?: number;
  /// The number of available worker slots
  worker_slots?: number;
  /// The total number of Nvidia GPUs
  nvidia_gpu?: number;
  /// The total number of AMD GPUs
  amd_gpu?: number;
  /// The amount of resources to allow an image to burst with
  burstable?: BurstableResourcesRequest;
};

/// The resources available on a node or required for an image
export type Resources = {
  /// The total amount of cpu in millicpu
  cpu: number;
  /// The total amount of ram in mebibytes
  memory: number;
  /// The total amount of ephemeral storage in mebibytes
  ephemeral_storage: number;
  /// The number of available worker slots
  worker_slots: number;
  /// The total number of Nvidia GPUs
  nvidia_gpu: number;
  /// The total number of AMD GPUs
  amd_gpu: number;
  /// The amount of resources to allow an image to burst with
  burstable: BurstableResourcesRequest;
};

export type ImageLifetime = {
  counter: string;
  amount: number;
};

export type SecurityContext = {
  user?: number;
  group?: number;
  allow_privilege_escalation?: boolean;
};

/// The arguments to add to this images jobs
export type ImageArgs = {
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
  /// How to pass the result files location in
  output?: ArgStrategyValue;
  /// How to pass the result location in
  output_files?: ArgStrategyValue;
};

export type SampleDependencySettings = {
  location?: string;
  kwarg?: string;
  strategy?: DependencyPassStrategy;
  naming?: FileNamingStrategy;
};

export type RepoDependencySettings = {
  location?: string;
  kwarg?: string;
  strategy?: DependencyPassStrategy;
};

export type TagDependencySettings = {
  enabled?: boolean;
  location?: string;
  kwarg?: string;
  strategy?: DependencyPassStrategy;
};

export type ChildrenDependencySettings = {
  enabled?: boolean;
  images?: string[];
  location?: string;
  kwarg?: string;
  strategy?: DependencyPassStrategy;
};

export type EphemeralDependencySettings = {
  location?: string;
  kwarg?: string;
  strategy?: DependencyPassStrategy;
  names?: string[];
};

export type KwargDependencyList = {
  List: string;
};

export type KwargDependencyMap = {
  Map: { [image: string]: string };
};

export type KwargDependencyValue = 'None' | KwargDependencyList | KwargDependencyMap;

export type ResultDependencySettings = {
  images?: string[];
  location?: string;
  kwarg?: KwargDependencyValue;
  strategy?: DependencyPassStrategy;
  names?: string[];
};

export type GenericCacheDependencySettings = {
  kwarg?: string;
  strategy?: DependencyPassStrategy;
};

export type CacheDependencySettings = {
  location?: string;
  generic?: GenericCacheDependencySettings;
  use_parent_cache?: boolean;
  enabled?: boolean;
};

export type Dependencies = {
  samples?: SampleDependencySettings;
  ephemeral?: EphemeralDependencySettings;
  results?: ResultDependencySettings;
  repos?: RepoDependencySettings;
  tags?: TagDependencySettings;
  children?: ChildrenDependencySettings;
  cache?: CacheDependencySettings;
};

export type AutoTag = {
  logic: AutoTagLogicValue;
  key?: string;
};

export type FilesHandler = {
  results?: string;
  result_files?: string;
  // Deliberate divergence from the Rust `FilesHandler` (api/src/models/results.rs), which has no
  // `entities` field. Reserved for a planned future feature; keep until the backend adds it.
  entities?: string;
  tags?: string;
  names?: string[];
};

// Defined here rather than results.ts so the `OutputHandler` enum value is available without a
// runtime circular import (results.ts only imports image types, never values).
/// A blank OutputCollection with all fields defaulted, for initializing image forms.
export const BlankOutputCollection: OutputCollection = {
  handler: OutputHandler.Files,
  files: {},
  as_filesystem: false,
  children: '',
  auto_tag: {},
  groups: [],
};

export type ChildFilters = {
  mime?: string[];
  file_name?: string[];
  file_extension?: string[];
  submit_non_matches?: boolean;
};

export type Cleanup = {
  job_id: ArgStrategyValue;
  results: ArgStrategyValue;
  result_files_dir: ArgStrategyValue;
  script: string;
};

export type Kvm = {
  xml: string;
  qcow2: string;
};

/// List of image names with a cursor (from api/src/models/images.rs)
export type ImageList = {
  cursor?: number;
  names: string[];
};

/// List of image details with a cursor (from api/src/models/images.rs)
export type ImageDetailsList = {
  cursor?: number;
  details: Image[];
};

export type ImageRequest = {
  group: string;
  name: string;
  version?: ImageVersion;
  scaler?: ImageScaler;
  image?: string;
  lifetime?: ImageLifetime;
  timeout?: number;
  resources?: ResourcesRequest;
  spawn_limit?: SpawnLimitsValue;
  volumes?: Volume[];
  env?: { [key: string]: string | null };
  args?: ImageArgs;
  modifiers?: string;
  description?: string;
  security_context?: SecurityContext;
  collect_logs?: boolean;
  generator?: boolean;
  dependencies?: Dependencies;
  display_type?: OutputDisplayType;
  output_collection?: OutputCollection;
  child_filters?: ChildFilters;
  clean_up?: Cleanup;
  kvm?: Kvm;
  network_policies?: string[];
};

export type ImageBanKind =
  { Generic: { msg: string } } | { InvalidImageUrl: { url: string } } | { InvalidHostPath: { volume_name: string; host_path: string } };

export type ImageBan = {
  id: string;
  time_banned: string;
  ban_kind: ImageBanKind;
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
  /// The resources required to spawn this image
  resources: Resources;
  /// The limit to use for how many workers of this image type can be spawned
  spawn_limit: SpawnLimitsValue;
  /// The environment variables to set
  env: { [key: string]: string | null };
  /// How long this image takes to execute on average in seconds
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
  child_filters: ChildFilters;
  /// The settings to use when cleaning up canceled jobs
  clean_up?: Cleanup;
  /// The settings to use for Kvm jobs
  kvm?: Kvm;
  /// A list of reasons an image is banned mapped by ban UUID
  bans: { [uuid: string]: ImageBan };
  /// The names of network policies to apply to the image when spawned
  network_policies: string[];
};
