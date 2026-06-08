export type Volume = {
  /// The name of the volume in k8s
  name: string;
  /// The type of volume this is
  archetype: VolumeTypes;
  /// Where this should be mounted at in the pod
  mount_path: string;
  /// A sub path for mounting specific files
  sub_path?: string;
  /// Whether this volume should be readonly or not
  read_only: boolean;
  /// whether to use the most recent config created by kustomize
  kustomize: boolean;
  // Specific options for all the different types of volumes
  /// Host path settings
  host_path?: HostPath;
  // Config map settings
  config_map?: ConfigMap;
  /// Secret settings
  secret?: Secret;
  /// NFS settings
  nfs?: NFS;
};

export type VolumeTypes = 'host_path' | 'config_map' | 'secret' | 'nfs';
export type HostPathTypes = 'DirectoryOrCreate' | 'Directory' | 'FileOrCreate' | 'File' | 'Socket' | 'CharDevice' | 'BlockDevice';

type HostPath = {
  path: string;
  path_type?: HostPathTypes;
};

type ConfigMap = {
  /// The mode bits to set on files in this volume
  default_mode?: number;
  /// Whether this configmap is optional or not
  optional?: boolean;
};

type Secret = {
  /// The mode bits to set on files in this volume
  default_mode?: number;
  /// Whether this secret is optional or not
  optional?: boolean;
};

type NFS = {
  /// The path that is exported by the NFS server
  path: string;
  /// The host/ip:port of the NFS server
  server: string;
};
