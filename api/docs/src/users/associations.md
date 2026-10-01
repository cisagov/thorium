# Associating Data

Associations are relationships between files, repositories, and entities in Thorium. They let you record that a
file is firmware for a device, that a repo was developed by a vendor, or that two files are simply related. Thorium
uses associations to build relationship trees between your data.

Every association has:

- a **kind** describing the relationship (e.g. `FileFor`, `DevelopedBy`, `AssociatedWith`)
- a **source** and a **target**, each of which can be a file (by SHA256), a repo (by URL), or an entity (by ID)
- a **direction**, which is either from the source to the target or bidirectional
- the **groups** it is visible in

Associations are stored on both sides, so listing the associations for either the source or the target will show
it. When listing, the direction is shown relative to the object you are listing:

| Direction | Meaning |
|-----------|---------|
| `To` | The association points from the listed object to the other object |
| `From` | The association points from the other object to the listed object |
| `Bidirectional` | The association points both ways |

The available association kinds are: `FileFor`, `DocumentationFor`, `FirmwareFor`, `AssociatedWith`,
`DevelopedBy`, `ContainsCVE`, `ContainsCWE`, `BasedIn`, `EmployedBy`, `ParentCompanyOf`, `UsedBy`, `UsedIn`,
`PerformedBy`, `FileSystemIn`, `FolderIn`, `FileIn`, `ProcessTreeIn`, `ChildProcess`, `HasNetworkConnection`,
`SectionIn`, `ImportIn`, `FlagFor`, `CreatedBy`, `SigmaRuleHit`, and `InvolvedIn`.

## Creating Associations (Thorctl)
---

Use `thorctl associations create` (or `thorctl assoc create`) with exactly one source (`--file`, `--repo`, or
`--entity`), one or more targets (`--target-files`/`-F`, `--target-repos`/`-R`, or `--target-entities`/`-E`), and an
association kind.

```bash
$ thorctl associations create --file 619dddbd681b593d8c6f66de50dd41e422bfac7a83feab6c3f0944de7c73a4d3 \
    --kind FirmwareFor --target-entities 0f4e4f3c-5d5e-4b8a-9a43-2a6f1c8e9b11
```

Targets can be repeated or comma separated, and a single request can mix files, repos, and entities:

```bash
$ thorctl assoc create --repo github.com/curl/curl --kind AssociatedWith \
    -F 619dddbd681b593d8c6f66de50dd41e422bfac7a83feab6c3f0944de7c73a4d3 \
    -R github.com/curl/curl-www
```

Add `--bidirectional` (`-b`) to make the associations point both ways. By default associations are created in the
groups shared by the source and each target. Use `--groups` (`-g`) to create them in specific groups instead. You
must be able to edit those groups, and they must allow associations.

## Listing Associations (Thorctl)
---

Use `thorctl associations list` with a single source to see its associations:

```bash
$ thorctl associations list --file 619dddbd681b593d8c6f66de50dd41e422bfac7a83feab6c3f0944de7c73a4d3
```

By default the first 50 associations are shown in a table. Use `--limit` (`-l`) or `--no-limit` to change how many
are listed, `--kind` (`-k`) to only show certain kinds, and `--groups` (`-g`) to only list associations from certain
groups. Use `--json`, `--condensed` (`-c`), or `--output <PATH>` (`-o`) to get JSON instead of a table.

## Deleting Associations (Thorctl)
---

Use `thorctl associations delete` with the same source, targets, and kind that you would use to create the
associations:

```bash
$ thorctl associations delete --file 619dddbd681b593d8c6f66de50dd41e422bfac7a83feab6c3f0944de7c73a4d3 \
    --kind FirmwareFor --target-entities 0f4e4f3c-5d5e-4b8a-9a43-2a6f1c8e9b11
```

Associations are matched no matter which side created them, so you can delete an association starting from either
its source or its target. Both sides of a matched association are removed. To only delete associations in a certain
direction relative to the source, use `--direction` (`-d`) with `to`, `from`, or `bidirectional`.

You will be asked to confirm before anything is deleted. Use `--skip-confirm` (`-y`) to skip the prompt, which is
required when running without a terminal.

Only the user who submitted an association, a manager or owner of the group it is in, or an admin can delete it. If
any matching association can't be deleted by you, nothing is deleted. Use `--groups` (`-g`) to only delete
associations from the groups you manage.

## API Routes
---

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/associations/` | Create associations |
| `DELETE` | `/api/associations/` | Delete associations matching a kind, source, and targets |
| `GET` | `/api/files/associations/<SHA256>` | List the associations for a file |
| `GET` | `/api/repos/associations/<URL>` | List the associations for a repo |
| `GET` | `/api/entities/associations/<ID>` | List the associations for an entity |
