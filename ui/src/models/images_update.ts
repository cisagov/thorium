import {
  ArgStrategy,
  AutoTagLogic,
  DependencyPassStrategy,
  FileNamingStrategy,
  ImageBan,
  ImageLifetime,
  ImageScaler,
  ImageVersion,
  OutputHandler,
} from './images';
import { OutputDisplayType } from './results';
import { Volume } from './volumes';

/**
 * An update for an image in Thorium
 */
export interface ImageUpdate {
  /** The image version to update */
  version?: ImageVersion | null;
  /** Whether the scaler should spawn containers for this image */
  external?: boolean | null;
  /** The image to use, either URL or tag */
  image?: string | null;
  /** What scaler is responsible for scaling this image */
  scaler?: ImageScaler | null;
  /** The lifetime of a pod */
  lifetime?: ImageLifetime | null;
  /** The timeout for individual jobs */
  timeout?: number | null; // Rust u64
  /** The resources to require for this image */
  resources?: ResourcesUpdate | null;
  /** The limit to use for how many workers of this image type can be spawned */
  spawn_limit?: SpawnLimits | null;
  /** The volumes to add */
  add_volumes?: Volume[];
  /** The names of the volumes to remove */
  remove_volumes?: string[];
  /** Environment args to add */
  add_env?: Record<string, string | null>;
  /** Environment args to remove */
  remove_env?: string[];
  /** Whether to clear the version or not */
  clear_version?: boolean;
  /** Whether to clear the image or not */
  clear_image?: boolean;
  /** Whether to clear the lifetime or not */
  clear_lifetime?: boolean;
  /** Whether to clear the description or not */
  clear_description?: boolean;
  /** The arguments to add to this image's jobs */
  args?: ImageArgsUpdate | null;
  /** The path to the modifier folders for this image */
  modifiers?: string | null;
  /** The image description */
  description?: string | null;
  /** The updates to the security context for this image */
  security_context?: SecurityContextUpdate | null;
  /** Whether the agent should stream stdout/stderr back to Thorium */
  collect_logs?: boolean | null;
  /** Whether this is a generator or not */
  generator?: boolean | null;
  /** Updates the dependency settings for this image */
  dependencies?: DependenciesUpdate;
  /** The type of display class to use in the UI for this image's output */
  display_type?: OutputDisplayType | null;
  /** The settings for collecting results from this image */
  output_collection?: OutputCollectionUpdate | null;
  /** An update to the image's child filters */
  child_filters?: ChildFiltersUpdate | null;
  /** The settings to use when cleaning up canceled jobs */
  clean_up?: CleanupUpdate;
  /** The settings to use for KVM jobs */
  kvm?: KvmUpdate;
  /** An update to the ban list containing a list of bans to add or remove */
  bans?: ImageBanUpdate;
  /** An update to the network policies to apply to the image */
  network_policies?: ImageNetworkPolicyUpdate;
}

export interface ResourcesUpdate {
  /** CPU cores in millicpus */
  cpu?: number | null;
  /** RAM in mebibytes */
  memory?: number | null;
  /** Ephemeral storage in mebibytes */
  ephemeral_storage?: number | null;
  /** The total number of Nvidia GPUs */
  nvidia_gpu?: number | null;
  /** The total number of AMD GPUs */
  amd_gpu?: number | null;
  /** The amount of resources to allow an image to burst with */
  burstable?: BurstableResourcesUpdate;
}

export interface BurstableResourcesUpdate {
  cpu?: number | null;
  memory?: number | null;
}

/** Limits for how many workers of an image type can be spawned */
export type SpawnLimits =
  /** Limit the amount of spawned workers for this image using a basic limit */
  | { Basic: number }
  /** No limit on the number of workers that can be spawned */
  | 'Unlimited';

export interface ImageArgsUpdate {
  /** The entrypoint to force all jobs to use */
  entrypoint?: string[] | null;
  /** Clear this images entrypoint */
  clear_entrypoint?: boolean;
  /** The command to force all jobs to use */
  command?: string[] | null;
  /** Clear this images command */
  clear_command?: boolean;
  /** What kwarg to pass the current reaction id in with */
  reaction?: string | null;
  /** Clear the reaction kwarg */
  clear_reaction?: boolean;
  /** What kwarg to pass the repo url in as */
  repo?: string | null;
  /** Clear the reaction kwarg */
  clear_repo?: boolean;
  /** What kwarg to pass the repo commit in with */
  commit?: string | null;
  /** Clear the reaction kwarg */
  clear_commit?: boolean;
  /** Update how to pass the the result files location in */
  output?: ArgStrategy | null;
  /** Update how to pass the the result location in */
  output_files?: ArgStrategy | null;
}

export interface SecurityContextUpdate {
  /** The user to run as */
  user?: number | null;
  /** The group to use */
  group?: number | null;
  /** Allow users to escalate their privileges */
  allow_privilege_escalation?: boolean | null;
  /** Clear the user id field */
  clear_user?: boolean;
  /** Clear the group id field */
  clear_group?: boolean;
}

export interface DependenciesUpdate {
  /** The strategy the agent should use when passing donwloaded samples to tools */
  samples?: SampleDependencySettingsUpdate;
  /** The strategy the agent should use when passing downloaded ephemeral files to tools */
  ephemeral?: EphemeralDependencySettingsUpdate;
  /** The strategy the agent should use when passing in prior results */
  results?: ResultDependencySettingsUpdate;
  /** The strategy the agent should use when passing donwloaded repos to tools */
  repos?: RepoDependencySettingsUpdate;
  /** The strategy the agent should use when passing donwloaded tags to tools */
  tags?: TagDependencySettingsUpdate;
  /** The settings the agent should use when passing children files from past tools */
  children?: ChildrenDependencySettingsUpdate;
  /** The settings the agent should use when reconstructing filesystems dumped by past tools */
  filesystems?: FileSystemDependencySettingsUpdate;
  /** The updated settings to use for a reactions cache */
  cache?: CacheDependencySettingsUpdate;
}

export interface SampleDependencySettingsUpdate {
  /** Where the agent should store downloaded dependencies */
  location?: string | null;
  /** The kwarg to pass these samples in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Whether to clear the kwarg setting or not */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing downloaded dependencies to jobs */
  strategy?: DependencyPassStrategy | null;
  /** The strategy to when naming any downloaded files */
  naming?: FileNamingStrategy | null;
}

export interface EphemeralDependencySettingsUpdate {
  /** Where the agent should store downloaded files */
  location?: string | null;
  /** The kwarg to pass these samples in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Whether to clear the kwarg setting or not */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing samples downloaded to jobs */
  strategy?: DependencyPassStrategy | null;
  /** Any names to add to the list of dependencies to restrict this image too */
  add_names?: string[];
  /** The names to remove from the list of dependencies to restrict this image too */
  remove_names?: string[];
}

export interface ResultDependencySettingsUpdate {
  /** The prior images to pass results from */
  add_images?: string[];
  /** The images to stop passing results from */
  remove_images?: string[];
  /** Where the agent should store downloaded prior result files */
  location?: string | null;
  /** The kwarg to pass these files in with if one is set, otherwise use positional args */
  kwarg?: KwargDependency | null;
  /** The strategy the agent should use when passing dependencies downloaded to jobs */
  strategy?: DependencyPassStrategy | null;
  /** Any files to limit this image to downloading */
  add_names?: string[];
  /** The file names to remove form our download list */
  remove_names?: string[];
}

export interface RepoDependencySettingsUpdate {
  /** Where the agent should store downloaded dependencies */
  location?: string | null;
  /** The kwarg to pass these samples in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Whether to clear the kwarg setting or not */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing downloaded dependencies to jobs */
  strategy?: DependencyPassStrategy | null;
}

export interface TagDependencySettingsUpdate {
  /** Whether tag dependencies should be enabled or not */
  enabled?: boolean | null;
  /** Where the agent should store downloaded dependencies */
  location?: string | null;
  /** The kwarg to pass these samples in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Whether to clear the kwarg setting or not */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing downloaded dependencies to jobs */
  strategy?: DependencyPassStrategy | null;
}

export interface ChildrenDependencySettingsUpdate {
  /** Whether tag dependencies should be enabled or not */
  enabled?: boolean | null;
  /** The prior images to pass results from */
  add_images?: string[];
  /** The images to stop passing results from */
  remove_images?: string[];
  /** Where the agent should store downloaded dependencies */
  location?: string | null;
  /** The kwarg to pass these samples in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Whether to clear the kwarg setting or not */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing downloaded dependencies to jobs */
  strategy?: DependencyPassStrategy | null;
}

export interface FileSystemDependencySettingsUpdate {
  /** Whether filesystem dependencies should be enabled or not */
  enabled?: boolean | null;
  /** The prior images to reconstruct filesystems from */
  add_images?: string[];
  /** The images to stop reconstructing filesystems from */
  remove_images?: string[];
  /** Where the agent should reconstruct downloaded filesystems */
  location?: string | null;
  /** The kwarg to pass these filesystems in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Whether to clear the kwarg setting or not */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing reconstructed filesystems to jobs */
  strategy?: DependencyPassStrategy | null;
}

export interface CacheDependencySettingsUpdate {
  /** The location to write our generic cache too */
  location?: string | null;
  /** The settings to use for the generic cache */
  generic?: GenericCacheDependencySettingsUpdate;
  /** Whether to use our parents cache if we have one or not */
  use_parent_cache?: boolean | null;
  /** Whether cache is enabled for this image */
  enabled?: boolean | null;
}

export interface GenericCacheDependencySettingsUpdate {
  /** The kwarg to pass this cache in with if one is set, otherwise use positional args */
  kwarg?: string | null;
  /** Clear the kwarg for our generic cache */
  clear_kwarg?: boolean;
  /** The strategy the agent should use when passing the downloaded cache to jobs */
  strategy?: DependencyPassStrategy | null;
}

export interface OutputCollectionUpdate {
  /** The handler used to collect output */
  handler?: OutputHandler | null;
  /** The file Handler settings */
  files?: FilesHandlerUpdate;
  /** Update settings for automatically extracting a tag from results */
  auto_tag?: Record<string, AutoTagUpdate>;
  /** Where to look for child files to ingest */
  children?: string | null;
  /** Whether to collect any children as a filesystem */
  as_filesystem?: boolean | null;
  /** The groups we should restrict our results uploads too */
  groups?: string[];
  /** Whether to clear the files handler settings */
  clear_files?: boolean;
  /** Whether to clear the results groups restrictions or not */
  clear_groups?: boolean;
}

export interface FilesHandlerUpdate {
  /** The location to look for small renderable results at on disk */
  results?: string | null;
  /** The location to look for files that should be uploaded as result files */
  result_files?: string | null;
  /** The location to load tags to set from */
  tags?: string | null;
  /** Any new file names to restrict our handler to */
  add_names?: string[];
  /** Any file names to remove from the list of file names to restrict our handler to */
  remove_names?: string[];
  /** Whether to clear the list of files names to restrict our handler to */
  clear_names?: boolean;
}

export interface AutoTagUpdate {
  /** The logic to use when deciding whether to apply this tag */
  logic?: AutoTagLogic | null;
  /** What to rename this tags key too */
  key?: string | null;
  /** Whether to clear the key value or not */
  clear_key?: boolean;
  /** Whether to delete this tag key or not */
  delete?: boolean;
}

export interface ChildFiltersUpdate {
  /** The mime filters to add */
  add_mime?: string[];
  /** The mime filters to remove */
  remove_mime?: string[];
  /** The file name filters to add */
  add_file_name?: string[];
  /** The file name filters to remove */
  remove_file_name?: string[];
  /** The file extension filters to add */
  add_file_extension?: string[];
  /** The file extension filters to remove */
  remove_file_extension?: string[];
  submit_non_matches?: boolean | null;
}

export interface CleanupUpdate {
  /** How to pass in the id of the cancelled job */
  job_id?: ArgStrategy | null;
  /** How to pass in this images result file path */
  results?: ArgStrategy | null;
  /** How to pass in the output dir for this tools result files */
  result_files_dir?: ArgStrategy | null;
  /** The clean up script to call */
  script?: string | null;
  /** Whether to clear our clean up settings */
  clear: boolean;
}

export interface KvmUpdate {
  /** The path to the golden XML file to use */
  xml?: string | null;
  /** The path to the golden qcow2 image to use */
  qcow2?: string | null;
}

export interface ImageBanUpdate {
  /** The list of bans to be added */
  bans_added: ImageBan[];
  /** The list of bans to be removed */
  bans_removed: string[];
}

export interface ImageNetworkPolicyUpdate {
  /** The list of policies to be added */
  policies_added: string[];
  /** The list of policies to be removed */
  policies_removed: string[];
}

export type KwargDependency =
  /** Pass in all results with a single kwarg key */
  | { List: string }
  /** Pass in all results with unique kwarg keys, image name to key */
  | { Map: Record<string, string> }
  /** Pass in all results with positional args */
  | 'None';
