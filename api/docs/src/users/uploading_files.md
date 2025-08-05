# Uploading Files

Now that you have access to Thorium, you may want to upload some files and run analysis tools on them. You can do that
in either the Web UI or through Thorctl. When uploading a small number of files, the Web UI is usually preferable, while
Thorctl is helpful in uploading many files or when a browser is not accessible.

When uploading files there are several options you may set that are described below. `Groups` is the only required
field. If you are not yet a member of any groups then follow the steps in the
[Adding/Editing Groups](../getting_started/adding_editing_groups.md) section and come back afterward.

| Field | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Groups | Limits who can see this file | One or more group names | yes |
| Description | A short text explanation of the sample and/or its source | Any valid UTF-8 formatted text | no |
| Tags | Key/value pairs to help locate and categorize files | Any key/value pair; both key and value are required | no |
| Origins | Specifies where a file came from | Downloaded, Transformed, Unpacked, Wire, Incident, or Memory Dump | no |

It is recommended that you provide origin information for any file(s) you upload whenever possible. A key feature of Thorium
is its ability to store origin information in a structured format and automatically translate that information into metadata tags.
Tags allow you to filter the files that you browse through when looking for a file. As a result, if you don't provide any origin
information, it may be difficult to locate your files at a later date.

### File Origins

File `Origins` are the single most important piece of information in describing, locating, and understanding relationships
between files. Described below are all the options for file origins and their respective subfields.

#### Downloaded

The "Downloaded" Origin specifies that the file was downloaded from a specific URL.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| URL | The URL the file was downloaded from | A valid URL | yes |
| Site Name | The name of the website the file was downloaded from | Any UTF-8 formatted text | no |

#### Transformed

The "Transformed" Origin specifies that the file is a result of transforming another file, whether by a tool or some other means.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Parent | The SHA256 of the original file that was transformed to produce this file | A valid SHA256 of an existing file in Thorium<sup>1</sup> | yes |
| Tool | The tool that was used to produce this transformed file | Any UTF-8 formatted text | no |
| Flags | The tool command-line flags that were used to transform this sample | One or more hyphenated alphanumeric flags<sup>2</sup> | no |

1. Your account must have access to the parent file in order to specify it in a file's origin
2. Example: `--flag1, --flag2, --flag3, -f`

#### Unpacked

The "Unpacked" Origin specifies that the file was unpacked from some other file, whether by a tool or some other means.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Parent | The SHA256 of the original file that this file was unpacked from | A valid SHA256 of an existing file in Thorium<sup>1</sup> | yes |
| Tool | The tool that was used to unpack this file | Any UTF-8 formatted text | no |
| Flags | The tool command-line flags that were used to unpack this sample | One or more hyphenated alphanumeric flags<sup>2</sup> | no |

1. Your account must have access to the parent file in order to specify it in a file's origin
2. Example: `--flag1, --flag2, --flag3, -f`

#### Wire

The "Wire" Origin specifies that a file was captured/sniffed "on the wire" en route to a destination.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Sniffer | The sniffer<sup>1</sup> used to capture this file | Any UTF-8 formatted text | yes |
| Source | The source IP/hostname this file came from when it was sniffed | Any UTF-8 formatted text | no |
| Destination | The destination IP/hostname where this file was headed to when it was sniffed | Any UTF-8 formatted text | no |

1. Example: `wireshark`

#### Incident

The "Incident" Origin specifies that the file originated from a specific security incident.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Incident ID | The name or ID identifying the incident from which the file originated | Any UTF-8 formatted text | yes |
| Cover Term | An optional term for the organization where an incident occurred | Any UTF-8 formatted text | no |
| Mission Team | The name of the mission team that handled the incident | Any UTF-8 formatted text | no |
| Network | The name of the network where the incident occurred | Any UTF-8 formatted text | no |
| Machine | The IP or hostname of the machine where the incident occurred | Any UTF-8 formatted text | no |
| Location | The physical/geographical location where the incident occurred | Any UTF-8 formatted text | no |

#### Memory Dump

The "Memory Dump" Origin specifies that the file originated from a memory dump.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Memory Type | The type of memory dump this file originated from | Any UTF-8 formatted text | yes |
| Parent | The SHA256 of the memory dump file in Thorium from which this file originates | A valid SHA256 of an existing file in Thorium<sup>1</sup> | no |
| Reconstructed | The characteristics that were reconstructed in this memory dump | One or more UTF-8 formatted strings | no |
| Base Address | The virtual address where the memory dump starts | An alphanumeric memory address | no |

1. Your account must have access to the parent file in order to specify it in a file's origin

#### Carved

The "Carved" Origin specifies that a file was "carved out" of another file (e.g. archive, memory dump, packet capture, etc.).
Unlike "Unpacked," "Carved" describes a sample that is a simple, discrete piece of another file. It's extraction can be easily
replicated without any dynamic unpacking process.

| Subfield | Description | Format/Accepted Values | Required |
| ----- | ----------- | ----------- | ----- |
| Parent | The SHA256 of the original file that was carved to produce this file | A valid SHA256 of an existing file in Thorium<sup>1</sup> | yes |
| Tool | The tool that was used to produce this transformed file | Any UTF-8 formatted text | no |
| Carved Origin | The type of file this sample was carved from (and other related metadata) | See below Carved origin subtypes | no |

1. Your account must have access to the parent file in order to specify it in a file's origin

Carved origins may also have an optional subtype defining what type of file the sample was originally carved from. The Carved
subtypes are described below:

##### PCAP

The "Carved PCAP" Origin specifies that a file was "carved out" of a network/packet capture.

| Subfield | Description | Format/Accepted Values | Required |
| -------- | ----------- | ---------------------- | -------- |
| Source IP | The source IP address this file came from | Any valid IPv4/IPv6 | no |
| Destination IP | The destination IP address this file was going to | Any valid IPv4/IPv6 | no |
| Source Port | The source port this file was sent from | Any valid port (16-bit unsigned integer) | no |
| Destination Port | The destination port this file was going to | Any valid port (16-bit unsigned integer) | no |
| Protocol | The protocol by which this file was sent | "UDP"/"Udp"/"udp" or "TCP"/"Tcp"/"tcp" | no |
| URL | The URL this file was sent from or to if it was sent using HTTP | Any UTF-8 formatted text | no |

##### Unknown

The "Carved Unknown" Origin specifies that a file was "carved out" of an unknown or unspecified file type.

This origin has no other subfields except for the ones from its parent "Carved" origin.

## Web UI
---
You can upload files in the Web UI by following the steps shown in the following video:

<video autoplay loop controls>
  <source src="../static_resources/upload/upload-file.mp4", type="video/mp4">
</video>

### Run Pipelines

You can choose to immediately run one or more pipelines on your uploaded file by selecting them in the `Run Pipelines` submenu.
You can also run pipelines on the file later from the file's page in the Web UI or using Thorctl (see
[Spawning Reactions](./spawning_reactions.md) for more info on running pipelines on files).

## Thorctl
---
It is best to use Thorctl when you have a large number of files that you want to upload. Thorctl will eagerly upload
multiple files in parallel by default, and specifying a directory to upload will recursively upload every file within
the directory tree. To upload a file or a folder of files, you can use the following command (using `--file-groups`/`-G`
go specify the groups to upload to):

```bash
thorctl files upload --file-groups <group> <files/or/folders>
```

If you have multiple files or folders to upload (e.g. `./hello.txt`, `/bin/ls`, and `~/Documents`), you can upload them all
in one command like so:

```bash
thorctl files upload -G example-group ./hello.txt /bin/ls ~/Documents
```

### Uploading to Multiple Groups

You can upload to more than one group by placing commas between each group:

```bash
thorctl files upload -G <group1>,<group2>,<group3> <file/or/folder>
```

Or by adding multiple `-G` or `--file-groups` flags:

```bash
thorctl files upload -G <group1> -G <group2> -G <group3> <file/or/folder>
```

### Uploading with Tags

You can also upload a file with specific tags with the `--file-tags` or `-T` flag:

```bash
thorctl files upload --file-groups <group> --file-tags Dataset=Examples --file-tags Corn=good <file/or/folder>
```

Because tags can contain any symbol (including commas), you must specify each tag with its own `-file-tags` or `-T` flag rather
than delimiting them with commas.

### Filtering Which Files to Upload

There may be cases where you want to upload only certain files within a folder. Thorctl provides the ability to either
inclusively or exclusively filter with regular expressions using the `--filter` and `--skip` flags, respectively.
For example, to upload only files with the `.exe` extension within a folder, you could run the following command:

```bash
thorctl files upload --file-groups example-group --filter .*\.exe ./my-folder
```

Or to upload everything within a folder except for files starting with `temp-`, you could run this command:

```bash
thorctl files upload --file-groups example-group --skip temp-.* ./my-folder
```

Supply multiple filters by specifying filter flags multiple times:

```bash
thorctl files upload --file-groups example-group --filter .*\.exe --filter .*evil.* --skip temp-.* ./my-folder
```

The filter and skip regular expressions must adhere to the format used by the Rust
[regex crate](https://docs.rs/regex/latest/regex/#syntax). Fortunately, this format is very similar to
most other popular regex types and should be relatively familiar. A helpful site to build and test your
regular expressions can be found here: [https://rustexp.lpil.uk](https://rustexp.lpil.uk/)

#### Hidden Directories

Additionally, if you want to include hidden sub-directories/files in a target directory, use the `--include-hidden` flag:

```bash
thorctl files upload -G example-group ./files --include-hidden
```

### Folder Tags

Thorctl also has a feature to use file subdirectories as tag values with customizable tag keys using the `--folder-tags` option.
For example, say you're uploading a directory `bin` with the following structure:

```
cool_binaries
├── file1
└── dumped
    ├── file2
    ├── file3
    ├── pe
        └── file4
    └── elf
        └── file5
```

The `cool_binaries` directory contains five total files spread across three subdirectories. Each tag we provide with `--folder-tags`
corresponds to a directory from top to bottom (including the root `cool_binaries` directory). So for example, if you run:

```bash
thorctl files upload -G example-group ./bin --folder-tags alpha --folder-tags beta --folder-tags gamma
```

The key `alpha` would correspond to the `bin` directory, `beta` to `dumped`, and `gamma` to `pe` and `elf`. So all
files in the `cool_binaries` directory **including files in subdirectories** would get the tag `alpha=cool_binaries`, all files in the
`dumped` directory would get the tag `beta=dumped`, and so on. Below is a summary of the files and the tags they
would have after running the above command:

| File | Tags |
| ---- | ---- |
| file1 | `alpha=cool_binaries` |
| file2 | `alpha=cool_binaries`, `beta=dumped` |
| file3 | `alpha=cool_binaries`, `beta=dumped` |
| file4 | `alpha=cool_binaries`, `beta=dumped`, `gamma=pe` |
| file5 | `alpha=cool_binaries`, `beta=dumped`, `gamma=elf` |

A few things to note:

- Tags correspond to subdirectory *levels*, not individual subdirectories, meaning files in subdirectories on the same
level will get the same tag key (like `pe` and `elf` above).
- You don't have to provide the same number of tags as subdirectory levels. Any files in subdirectories deeper than the
number of folder tags will receive all of their parents' tags until the provided tags are exhausted (e.g. a file in a
child directory of `elf` called `x86` would get tags for `cool_binaries`, `dumped` and `elf` but not for `x86`).

### Adjust Number of Parallel Uploads

By default, Thorctl can perform a maximum of 10 actions in parallel at any given time. In the case of file uploads, that means
a maximum of 10 files can be uploaded concurrently. You can adjust the number of parallel actions Thorctl will attempt to conduct
using the `-w` flag:

```bash
thorctl -w 20 files upload --file-groups <group> <file/or/folders>
```
