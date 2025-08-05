# Volumes in Images

Volumes are how you tell Thorium that pods based on this image need to have a 
volume bound in when being spawned. Currently Thorium supports three volume types

- HostPath
- ConfigMap
- Secret
- NFS

Thorium will not create any volumes that do not exist. This means the user must
create any volumes they expect to use in Thorium. 

When adding a volume to an image you will add a json blob that looks something like
this to to the volumes array:

```json
{
  "name": "SecretVolume",
  "archetype": "Secret",
  "mount_path": "/mnt/spooky",
  "sub_path": "spooky.yml",
  "read_only": true,
  "kustomize": true,
}
```

| key | definition |
| --- | ---------- |
| name | The name of the volume in k8s/Thorium |
| archetype | The type of volume this is |
| mount_path | Where to mount this volume at in the pod |
| sub_path | A specific file to mount from the volume (optional) |
| read_only | If this volume should be read only (optional, defaults to false) |
| kustomize | If kustomize support should be enabled (optional, defaults to false) |

### Kustomize Support
---

The following types of volumes can be created by kustomize and be used in Thorium:

- ConfigMap
- Secret

In order to support volumes created by kustomize Thorium tries to list the volumes
in the target namespace. It will then find the most recently created volume that
starts with the same name as the your target volume. This does mean that it is
possible the wrong volume will be bound in if you have a name collision. So if we
want to bind in a volume created by kustomize named "kustomize-volume". The scaler
will look list all the volumes and then find the most recently created one that
begins with "kustomize-volume".

### Specific Settings
---

Some volume types also require specific settings to also be set. Currently this is just
NFS but specific settings can also be set for ConfigMap and Secrets. Setting the
incorrect specific settings will just be ignored.

#### NFS specific Settings

```json
{
  "name": "NFS-vol",
  "archetype": "NFS",
  "mount_path": "/mnt/nfs",
  "read_only": true,
  "nfs": {
    "path": "/nfs",
    "server": "nfs-host"
  }
}
```

| key | definition |
| --- | ---------- |
| path | The path to bind in from the NFS share |
| server | The hostname/ip of the server that is hosting the NFS share |

Kustomize support is not available for NFS.

#### HostPath Specific Settings

```json
{
  "name": "HostPath-vol",
  "archetype": "HostPath",
  "mount_path": "/mnt/path/in/pod",
  "kustomize": false,
  "host_path": {
    "path": "/mnt/path/in/host",
    "path_type": "DirectoryOrCreate"
  }
}
```

| key | definition |
| --- | ---------- |
| path | The path to bind in from the host |
| path_type | The type of host path volume to use (optional) |

#### ConfigMap Specific Settings

setting ``read_only`` to true will override the default mode settings in the
specific settings.

```json
{
  "name": "ConfigMap-vol",
  "archetype": "ConfigMap",
  "mount_path": "/mnt/configmap",
  "kustomize": false,
  "config_map": {
    "default_mode": 0777,
    "optional": true
  }
}
```

| key | definition |
| --- | ---------- |
| default_mode | The permissions for all files in this volume (optional) |
| optional | Whether this volume is required to spawn this pod (optional) |

#### Secret Specific Settings

setting ``read_only`` to true will override the default mode settings in the
specific settings.

```json
{
  "name": "Secret-vol",
  "archetype": "Secret",
  "mount_path": "/mnt/secret",
  "kustomize": false,
  "secret": {
    "default_mode": 0400,
    "optional": true
  }
}
```

| key | definition |
| --- | ---------- |
| default_mode | The permissions for all files in this volume (optional) |
| optional | Whether this volume is required to spawn this pod (optional) |


