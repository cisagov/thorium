import { FieldValueType, type FieldSchema } from '../../types';

export const REQUIRED_IMAGE_FIELDS = ['group', 'name'] as const;

export const KNOWN_IMAGE_FIELDS = [
  'group',
  'name',
  'version',
  'scaler',
  'image',
  'lifetime',
  'timeout',
  'resources',
  'spawn_limit',
  'volumes',
  'env',
  'args',
  'modifiers',
  'description',
  'security_context',
  'collect_logs',
  'generator',
  'dependencies',
  'display_type',
  'output_collection',
  'child_filters',
  'clean_up',
  'kvm',
  'network_policies',
] as const;

export const IMAGE_SCALER_VALUES = ['K8s', 'BareMetal', 'Windows', 'Kvm', 'External'] as const;

export const ARG_STRATEGY_VALUES = ['None', 'Append', 'Kwarg'] as const;

export const SPAWN_LIMIT_VALUES = ['Unlimited', 'Basic'] as const;

export const ARG_STRATEGY_SCHEMA: FieldSchema = {
  type: FieldValueType.Enum,
  typeName: 'ArgStrategy',
  enumValues: ARG_STRATEGY_VALUES,
  variants: {
    None: null,
    Append: null,
    Kwarg: { type: FieldValueType.String, placeholder: 'kwarg-name' },
  },
};

export const DEPENDENCY_PASS_STRATEGY_VALUES = ['Paths', 'Names', 'Directory', 'Disabled'] as const;

export const FILE_NAMING_STRATEGY_VALUES = ['Sha256', 'MostRecent'] as const;

export const KWARG_DEPENDENCY_VALUES = ['None', 'List', 'Map'] as const;

// Mirrors the Rust `KwargDependency` enum: None (positional), List(String) (single kwarg for all),
// or Map(<image> -> <kwarg>) (per-image kwargs). Used for dependencies.results.kwarg.
export const KWARG_DEPENDENCY_SCHEMA: FieldSchema = {
  type: FieldValueType.Enum,
  typeName: 'KwargDependency',
  enumValues: KWARG_DEPENDENCY_VALUES,
  variants: {
    None: null,
    List: { type: FieldValueType.String, placeholder: '--results' },
    // Object without `fields` => a free key/value map (image name -> kwarg)
    Map: { type: FieldValueType.Object, placeholder: 'image-name' },
  },
};

export const OUTPUT_DISPLAY_TYPE_VALUES = [
  'JSON',
  'String',
  'Table',
  'Image',
  'Custom',
  'Disassembly',
  'HTML',
  'Markdown',
  'Hidden',
  'XML',
] as const;

export const OUTPUT_HANDLER_VALUES = ['Files'] as const;

export const VOLUME_TYPE_VALUES = ['HostPath', 'ConfigMap', 'Secret', 'NFS'] as const;

export const HOST_PATH_TYPE_VALUES = [
  'DirectoryOrCreate',
  'Directory',
  'FileOrCreate',
  'File',
  'Socket',
  'CharDevice',
  'BlockDevice',
] as const;

export const LIFETIME_COUNTER_VALUES = ['jobs', 'time'] as const;

export const KNOWN_RESOURCES_FIELDS = ['cpu', 'memory', 'ephemeral_storage', 'worker_slots', 'nvidia_gpu', 'amd_gpu', 'burstable'] as const;

export const KNOWN_ARGS_FIELDS = ['entrypoint', 'command', 'reaction', 'repo', 'commit', 'output', 'output_files'] as const;

export const KNOWN_DEPENDENCIES_FIELDS = ['samples', 'ephemeral', 'results', 'repos', 'tags', 'children', 'cache'] as const;

export const KNOWN_SAMPLE_DEP_FIELDS = ['location', 'kwarg', 'strategy', 'naming'] as const;

export const KNOWN_REPO_DEP_FIELDS = ['location', 'kwarg', 'strategy'] as const;

export const KNOWN_TAG_DEP_FIELDS = ['enabled', 'location', 'kwarg', 'strategy'] as const;

export const KNOWN_CHILDREN_DEP_FIELDS = ['enabled', 'images', 'location', 'kwarg', 'strategy'] as const;

export const KNOWN_EPHEMERAL_DEP_FIELDS = ['location', 'kwarg', 'strategy', 'names'] as const;

export const KNOWN_RESULT_DEP_FIELDS = ['images', 'location', 'kwarg', 'strategy', 'names'] as const;

export const KNOWN_CACHE_DEP_FIELDS = ['location', 'generic', 'use_parent_cache', 'enabled'] as const;

export const KNOWN_OUTPUT_COLLECTION_FIELDS = ['handler', 'files', 'as_filesystem', 'children', 'auto_tag', 'groups'] as const;

export const KNOWN_FILES_HANDLER_FIELDS = ['results', 'result_files', 'entities', 'tags', 'names'] as const;

export const KNOWN_CHILD_FILTERS_FIELDS = ['mime', 'file_name', 'file_extension', 'submit_non_matches'] as const;

export const KNOWN_CLEANUP_FIELDS = ['job_id', 'results', 'result_files_dir', 'script'] as const;

export const KNOWN_SECURITY_CONTEXT_FIELDS = ['user', 'group', 'allow_privilege_escalation'] as const;

export const KNOWN_LIFETIME_FIELDS = ['counter', 'amount'] as const;

export const KNOWN_VOLUME_FIELDS = [
  'name',
  'archetype',
  'mount_path',
  'sub_path',
  'read_only',
  'kustomize',
  'host_path',
  'config_map',
  'secret',
  'nfs',
] as const;

export const KNOWN_KVM_FIELDS = ['xml', 'qcow2'] as const;

export const VOLUME_ENTRY_SCHEMA: FieldSchema = {
  type: FieldValueType.Object,
  typeName: 'Volume',
  // archetype selects which nested config object applies (one of host_path/config_map/secret/nfs)
  variantField: { field: 'archetype', fieldMap: { HostPath: 'host_path', ConfigMap: 'config_map', Secret: 'secret', NFS: 'nfs' } },
  fields: {
    name: { type: FieldValueType.String, required: true, placeholder: 'volume-name' },
    archetype: { type: FieldValueType.Enum, required: true, enumValues: VOLUME_TYPE_VALUES },
    mount_path: { type: FieldValueType.String, required: true, placeholder: '/mnt/data' },
    sub_path: { type: FieldValueType.String, placeholder: 'sub/path' },
    read_only: { type: FieldValueType.Boolean },
    kustomize: { type: FieldValueType.Boolean },
    host_path: {
      type: FieldValueType.Object,
      typeName: 'HostPath',
      fields: {
        path: { type: FieldValueType.String, required: true, placeholder: '/host/path' },
        path_type: { type: FieldValueType.Enum, enumValues: HOST_PATH_TYPE_VALUES },
      },
    },
    config_map: {
      type: FieldValueType.Object,
      typeName: 'ConfigMap',
      fields: {
        default_mode: { type: FieldValueType.Number, placeholder: '420' },
        optional: { type: FieldValueType.Boolean },
      },
    },
    secret: {
      type: FieldValueType.Object,
      typeName: 'Secret',
      fields: {
        default_mode: { type: FieldValueType.Number, placeholder: '420' },
        optional: { type: FieldValueType.Boolean },
      },
    },
    nfs: {
      type: FieldValueType.Object,
      typeName: 'NFS',
      fields: {
        path: { type: FieldValueType.String, required: true, placeholder: '/nfs/share' },
        server: { type: FieldValueType.String, required: true, placeholder: 'nfs-server' },
      },
    },
  },
};

export const KNOWN_BURSTABLE_FIELDS = ['cpu', 'memory'] as const;

export const KNOWN_GENERIC_CACHE_FIELDS = ['kwarg', 'strategy'] as const;

export const AUTO_TAG_LOGIC_VALUES = [
  'Exists',
  'Equal',
  'Not',
  'Greater',
  'GreaterOrEqual',
  'LesserOrEqual',
  'Lesser',
  'In',
  'NotIn',
] as const;
export const KNOWN_AUTO_TAG_FIELDS = ['logic', 'key'] as const;

// Mirrors the Rust `AutoTagLogic` enum. `Exists` is a bare unit variant; the comparison variants
// carry a single JSON value, and In/NotIn carry a list of values. Used for output_collection.auto_tag.*.logic.
export const AUTO_TAG_LOGIC_SCHEMA: FieldSchema = {
  type: FieldValueType.Enum,
  typeName: 'AutoTagLogic',
  required: true,
  enumValues: AUTO_TAG_LOGIC_VALUES,
  variants: {
    Exists: null,
    Equal: { type: FieldValueType.String, placeholder: 'value' },
    Not: { type: FieldValueType.String, placeholder: 'value' },
    Greater: { type: FieldValueType.String, placeholder: 'value' },
    GreaterOrEqual: { type: FieldValueType.String, placeholder: 'value' },
    LesserOrEqual: { type: FieldValueType.String, placeholder: 'value' },
    Lesser: { type: FieldValueType.String, placeholder: 'value' },
    In: { type: FieldValueType.StringArray, placeholder: 'value' },
    NotIn: { type: FieldValueType.StringArray, placeholder: 'value' },
  },
};

const SEMVER_RE = /^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$/;

export function transformVersion(value: string): { yaml: string; json: string; valid: boolean; error?: string } {
  if (!value.trim()) return { yaml: "''", json: '""', valid: false, error: 'Version is required' };
  const m = value.trim().match(SEMVER_RE);
  if (m) {
    const [, major, minor, patch] = m;
    const pre = m[4] ?? '';
    const build = m[5] ?? '';
    const indent = '  ';
    const yaml = [
      '',
      `${indent}SemVer:`,
      `${indent}${indent}major: ${major}`,
      `${indent}${indent}minor: ${minor}`,
      `${indent}${indent}patch: ${patch}`,
      `${indent}${indent}pre: '${pre}'`,
      `${indent}${indent}build: '${build}'`,
    ].join('\n');
    const json = `{ "SemVer": { "major": ${major}, "minor": ${minor}, "patch": ${patch}, "pre": "${pre}", "build": "${build}" } }`;
    return { yaml, json, valid: true };
  }
  return { yaml: `\n  Custom: '${value.trim()}'`, json: `{ "Custom": "${value.trim()}" }`, valid: true };
}

export const IMAGE_SECTION_ORDER = [
  'Image',
  'Resources',
  'Arguments',
  'Output Collection',
  'Dependencies',
  'Environment',
  'Volumes',
  'Network Policies',
  'Security Context',
  'Child Filters',
  'Clean Up',
  'KVM',
  'Invalid Fields',
  'Unknown Fields',
] as const;

const IMAGE_CATEGORY_MAP: Record<string, string> = {
  resources: 'Resources',
  args: 'Arguments',
  output_collection: 'Output Collection',
  dependencies: 'Dependencies',
  env: 'Environment',
  volumes: 'Volumes',
  network_policies: 'Network Policies',
  security_context: 'Security Context',
  child_filters: 'Child Filters',
  clean_up: 'Clean Up',
  kvm: 'KVM',
};

export function imageFieldCategory(field: string): string {
  const root = field.split('.')[0];
  return IMAGE_CATEGORY_MAP[root] ?? 'Image';
}

export const IMAGE_FIELD_SCHEMAS: Record<string, FieldSchema> = {
  group: { type: FieldValueType.String, required: true, placeholder: 'group-name', description: 'Group this image belongs to' },
  name: { type: FieldValueType.String, required: true, placeholder: 'image-name', description: 'Name of this image' },
  version: {
    type: FieldValueType.String,
    placeholder: '1.0.0 or custom-tag',
    transform: transformVersion,
    description: 'Semantic version or custom tag',
  },
  image: { type: FieldValueType.String, placeholder: 'registry/image:tag', description: 'Container image URL or tag' },
  description: { type: FieldValueType.String, placeholder: 'Image description', description: 'Human-readable description' },
  modifiers: { type: FieldValueType.String, placeholder: 'modifier string', description: 'Path to modifier folders' },
  scaler: { type: FieldValueType.Enum, enumValues: IMAGE_SCALER_VALUES, description: 'Backend that scales this image' },
  display_type: { type: FieldValueType.Enum, enumValues: OUTPUT_DISPLAY_TYPE_VALUES, description: 'How results are displayed in the UI' },
  timeout: { type: FieldValueType.Number, placeholder: '300', description: 'Timeout in seconds for individual jobs' },
  spawn_limit: {
    type: FieldValueType.Enum,
    typeName: 'SpawnLimits',
    enumValues: SPAWN_LIMIT_VALUES,
    description: 'Limit workers spawned across clusters',
    variants: {
      Unlimited: null,
      Basic: { type: FieldValueType.Number, placeholder: '5' },
    },
  },
  collect_logs: { type: FieldValueType.Boolean, description: 'Stream stdout/stderr back to Thorium' },
  generator: { type: FieldValueType.Boolean, description: 'Whether this image generates child submissions' },
  kvm: {
    type: FieldValueType.Object,
    typeName: 'Kvm',
    description: 'KVM virtual machine configuration',
    fields: {
      xml: { type: FieldValueType.String, required: true, placeholder: '/path/to/vm.xml', description: 'Path to the VM XML definition' },
      qcow2: { type: FieldValueType.String, required: true, placeholder: '/path/to/disk.qcow2', description: 'Path to the disk image' },
    },
  },
  lifetime: {
    type: FieldValueType.Object,
    typeName: 'ImageLifetime',
    description: 'Pod lifetime limit by job count or time',
    fields: {
      counter: {
        type: FieldValueType.Enum,
        required: true,
        enumValues: LIFETIME_COUNTER_VALUES,
        description: 'Lifetime counter type (jobs or time)',
      },
      amount: {
        type: FieldValueType.Number,
        required: true,
        placeholder: '32',
        description: 'Number of jobs or seconds before pod terminates',
      },
    },
  },
  resources: {
    type: FieldValueType.Object,
    typeName: 'Resources',
    description: 'CPU, memory, and GPU resources required',
    fields: {
      cpu: { type: FieldValueType.Number, placeholder: '1000', description: 'CPU in millicpu' },
      memory: { type: FieldValueType.Number, placeholder: '4096', description: 'RAM in mebibytes' },
      ephemeral_storage: { type: FieldValueType.Number, placeholder: '0', description: 'Ephemeral storage in mebibytes' },
      worker_slots: { type: FieldValueType.Number, placeholder: '1', description: 'Concurrent job slots consumed' },
      nvidia_gpu: { type: FieldValueType.Number, placeholder: '0', description: 'Nvidia GPUs required' },
      amd_gpu: { type: FieldValueType.Number, placeholder: '0', description: 'AMD GPUs required' },
      burstable: {
        type: FieldValueType.Object,
        typeName: 'BurstableResources',
        description: 'Extra resources for burstable workloads',
        fields: {
          cpu: { type: FieldValueType.Number, placeholder: '1000', description: 'Burstable CPU in millicpu' },
          memory: { type: FieldValueType.Number, placeholder: '4096', description: 'Burstable RAM in mebibytes' },
        },
      },
    },
  },
  args: {
    type: FieldValueType.Object,
    typeName: 'ImageArgs',
    description: 'Arguments and kwargs passed to jobs',
    fields: {
      entrypoint: { type: FieldValueType.StringArray, placeholder: '/entrypoint.sh', description: 'Container entrypoint override' },
      command: { type: FieldValueType.StringArray, placeholder: 'run', description: 'Container command override' },
      reaction: { type: FieldValueType.String, placeholder: 'reaction-kwarg', description: 'Kwarg for the current reaction ID' },
      repo: { type: FieldValueType.String, placeholder: 'repo-kwarg', description: 'Kwarg for the repo URL' },
      commit: { type: FieldValueType.String, placeholder: 'commit-kwarg', description: 'Kwarg for the repo commit hash' },
      output: { ...ARG_STRATEGY_SCHEMA, description: 'How to pass the result output path' },
      output_files: { ...ARG_STRATEGY_SCHEMA, description: 'How to pass the result files path' },
    },
  },
  dependencies: {
    type: FieldValueType.Object,
    typeName: 'Dependencies',
    description: 'How dependencies are downloaded and passed to jobs',
    fields: {
      samples: {
        type: FieldValueType.Object,
        typeName: 'SampleDependencySettings',
        description: 'Settings for downloading sample files',
        fields: {
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/samples', description: 'Download path on the worker' },
          kwarg: { type: FieldValueType.String, placeholder: 'samples', description: 'Kwarg to pass sample paths with' },
          strategy: {
            type: FieldValueType.Enum,
            enumValues: DEPENDENCY_PASS_STRATEGY_VALUES,
            description: 'How to pass downloaded samples to jobs',
          },
          naming: { type: FieldValueType.Enum, enumValues: FILE_NAMING_STRATEGY_VALUES, description: 'File naming strategy for downloads' },
        },
      },
      ephemeral: {
        type: FieldValueType.Object,
        typeName: 'EphemeralDependencySettings',
        description: 'Settings for ephemeral file dependencies',
        fields: {
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/ephemeral', description: 'Download path on the worker' },
          kwarg: { type: FieldValueType.String, placeholder: 'ephemeral', description: 'Kwarg to pass ephemeral paths with' },
          strategy: {
            type: FieldValueType.Enum,
            enumValues: DEPENDENCY_PASS_STRATEGY_VALUES,
            description: 'How to pass ephemeral files to jobs',
          },
          names: { type: FieldValueType.StringArray, placeholder: 'file-name', description: 'Specific file names to download' },
        },
      },
      results: {
        type: FieldValueType.Object,
        typeName: 'ResultDependencySettings',
        description: 'Settings for prior result dependencies',
        fields: {
          images: { type: FieldValueType.StringArray, placeholder: 'image-name', description: 'Images to pull results from' },
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/results', description: 'Download path on the worker' },
          kwarg: {
            ...KWARG_DEPENDENCY_SCHEMA,
            description: 'How result paths are passed to jobs (positional, single kwarg, or per-image kwargs)',
          },
          strategy: { type: FieldValueType.Enum, enumValues: DEPENDENCY_PASS_STRATEGY_VALUES, description: 'How to pass results to jobs' },
          names: { type: FieldValueType.StringArray, placeholder: 'result-name', description: 'Specific result names to download' },
        },
      },
      repos: {
        type: FieldValueType.Object,
        typeName: 'RepoDependencySettings',
        description: 'Settings for repo dependencies',
        fields: {
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/repos', description: 'Download path on the worker' },
          kwarg: { type: FieldValueType.String, placeholder: 'repos', description: 'Kwarg to pass repo paths with' },
          strategy: { type: FieldValueType.Enum, enumValues: DEPENDENCY_PASS_STRATEGY_VALUES, description: 'How to pass repos to jobs' },
        },
      },
      tags: {
        type: FieldValueType.Object,
        typeName: 'TagDependencySettings',
        description: 'Settings for prior tag dependencies',
        fields: {
          enabled: { type: FieldValueType.Boolean, description: 'Enable tag downloading' },
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/tags', description: 'Download path on the worker' },
          kwarg: { type: FieldValueType.String, placeholder: 'tags', description: 'Kwarg to pass tag paths with' },
          strategy: { type: FieldValueType.Enum, enumValues: DEPENDENCY_PASS_STRATEGY_VALUES, description: 'How to pass tags to jobs' },
        },
      },
      children: {
        type: FieldValueType.Object,
        typeName: 'ChildrenDependencySettings',
        description: 'Settings for prior children dependencies',
        fields: {
          enabled: { type: FieldValueType.Boolean, description: 'Enable children downloading' },
          images: { type: FieldValueType.StringArray, placeholder: 'image-name', description: 'Restrict to children from these images' },
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/children', description: 'Download path on the worker' },
          kwarg: { type: FieldValueType.String, placeholder: 'children', description: 'Kwarg to pass children paths with' },
          strategy: { type: FieldValueType.Enum, enumValues: DEPENDENCY_PASS_STRATEGY_VALUES, description: 'How to pass children to jobs' },
        },
      },
      cache: {
        type: FieldValueType.Object,
        typeName: 'CacheDependencySettings',
        description: 'Settings for cache dependencies',
        fields: {
          location: { type: FieldValueType.String, placeholder: '/tmp/thorium/cache', description: 'Cache storage path on the worker' },
          generic: {
            type: FieldValueType.Object,
            typeName: 'GenericCacheDependencySettings',
            description: 'Generic cache kwarg settings',
            fields: {
              kwarg: { type: FieldValueType.String, placeholder: 'cache-kwarg', description: 'Kwarg to pass cache path with' },
              strategy: {
                type: FieldValueType.Enum,
                enumValues: DEPENDENCY_PASS_STRATEGY_VALUES,
                description: 'How to pass cache to jobs',
              },
            },
          },
          use_parent_cache: { type: FieldValueType.Boolean, description: "Use parent job's cache" },
          enabled: { type: FieldValueType.Boolean, description: 'Enable cache dependencies' },
        },
      },
    },
  },
  security_context: {
    type: FieldValueType.Object,
    typeName: 'SecurityContext',
    description: 'User/group and privilege settings for the container',
    fields: {
      user: { type: FieldValueType.Number, placeholder: '1000', description: 'Unix user ID to run as' },
      group: { type: FieldValueType.Number, placeholder: '1000', description: 'Unix group ID to run as' },
      allow_privilege_escalation: { type: FieldValueType.Boolean, description: 'Allow privilege escalation in the container' },
    },
  },
  output_collection: {
    type: FieldValueType.Object,
    typeName: 'OutputCollection',
    description: 'Settings for collecting results from jobs',
    fields: {
      handler: {
        type: FieldValueType.Enum,
        required: true,
        enumValues: OUTPUT_HANDLER_VALUES,
        description: 'Output collection handler type',
      },
      files: {
        type: FieldValueType.Object,
        typeName: 'FilesHandler',
        description: 'File handler result paths',
        fields: {
          results: { type: FieldValueType.String, placeholder: '/tmp/thorium/results', description: 'Path for result data' },
          result_files: { type: FieldValueType.String, placeholder: '/tmp/thorium/result-files', description: 'Path for result files' },
          entities: { type: FieldValueType.String, placeholder: '/tmp/thorium/entities.json', description: 'Path for entity JSON output' },
          tags: { type: FieldValueType.String, placeholder: '/tmp/thorium/tags', description: 'Path for tag output' },
          names: { type: FieldValueType.StringArray, placeholder: 'file-name', description: 'Result file names to collect' },
        },
      },
      as_filesystem: { type: FieldValueType.Boolean, description: 'Treat output as a filesystem' },
      children: { type: FieldValueType.String, placeholder: '/tmp/thorium/children', description: 'Path for child submissions' },
      auto_tag: {
        type: FieldValueType.Object,
        typeName: 'AutoTag',
        placeholder: 'tag-name',
        description: 'Automatic tagging rules based on results',
        fields: {
          logic: { ...AUTO_TAG_LOGIC_SCHEMA, description: 'Comparison logic for the auto tag rule' },
          key: { type: FieldValueType.String, placeholder: 'result-key', description: 'Result key to evaluate' },
        },
      },
      groups: { type: FieldValueType.StringArray, placeholder: 'group-name', description: 'Groups for output collection' },
    },
  },
  child_filters: {
    type: FieldValueType.Object,
    typeName: 'ChildFilters',
    description: 'Regex filters for child file submissions',
    fields: {
      mime: { type: FieldValueType.StringArray, placeholder: 'application/pdf', description: 'Filter by MIME type patterns' },
      file_name: { type: FieldValueType.StringArray, placeholder: '*.exe', description: 'Filter by file name patterns' },
      file_extension: { type: FieldValueType.StringArray, placeholder: '.dll', description: 'Filter by file extensions' },
      submit_non_matches: { type: FieldValueType.Boolean, description: 'Submit files that do not match filters' },
    },
  },
  clean_up: {
    type: FieldValueType.Object,
    typeName: 'Cleanup',
    description: 'Script and settings for cleaning up canceled jobs',
    fields: {
      job_id: { ...ARG_STRATEGY_SCHEMA, description: 'How to pass the job ID to the cleanup script' },
      results: { ...ARG_STRATEGY_SCHEMA, description: 'How to pass the results path' },
      result_files_dir: { ...ARG_STRATEGY_SCHEMA, description: 'How to pass the result files directory' },
      script: { type: FieldValueType.String, required: true, placeholder: '/cleanup.sh', description: 'Path to the cleanup script' },
    },
  },
  env: { type: FieldValueType.Object, placeholder: 'KEY: value', description: 'Environment variables for the container' },
  volumes: { type: FieldValueType.StringArray, placeholder: 'volume entry', description: 'Volumes to bind into the container' },
  network_policies: {
    type: FieldValueType.StringArray,
    placeholder: 'policy-name',
    description: 'Network policies to apply when spawned (K8s only)',
  },
};
