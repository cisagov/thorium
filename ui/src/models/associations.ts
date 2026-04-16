/// The different possible associations
export enum AssociationKind {
  /// This file is associated with something else
  FileFor = 'FileFor',
  /// This is documentation for something else
  DocumentationFor = 'documentationFor',
  /// This file or repo is or contains firmware for a device
  FirmwareFor = 'FirmwareFor',
  /// This file/repo/entity is associated with something else
  AssociatedWith = 'AssociatedWith',
  /// This was developed or created by
  DevelopedBy = 'DevelopedBy',
  /// This contains a CVE
  ContainsCVE = 'ContainsCVE',
  /// This contains a CWE
  ContainsCWE = 'ContainsCWE',
  /// This is based in specific countries
  BasedIn = 'BasedIn',
  /// This person was or is employed by
  EmployedBy = 'EmployedBy',
  /// This is the parent company of another company
  ParentCompanyOf = 'ParentCompanyOf',
  /// This is used by a specific person or group
  UsedBy = 'UsedBy',
  /// This was used in a specific campaign or engagement
  UsedIn = 'UsedIn',
  /// This campaign was performed by
  PerformedBy = 'PerformedBy',
  /// This filesystem was extracted/carved from
  FileSystemIn = 'FileSystemIn',
  /// This is a folder within a filesystem or another folder
  FolderIn = 'FolderIn',
  /// This is a file in a folder in a filesytem
  FileIn = 'FileIn',
  /// A Process tree in or from something
  ProcessTreeIn = 'ProcessTreeIn',
  /// A Process in a process tree or a child process
  ChildProcess = 'ChildProcess',
}

export type AssociationTarget = {
  /// This association is associated with another entity
  Entity?: { id: string; name: string };
  /// This association is associated with a file
  File?: string;
  /// This association is associated with a repo
  Repo?: string;
};

export type Association = {
  /// The kind of association this is
  kind: AssociationKind;
  /// The other data this directional association is with
  other: AssociationTarget;
  /// The creator of this association
  submitter?: string;
  /// The groups for this association
  groups: string[];
  /// When this association was created
  created: string;
  /// Whether this direction is to our source object or away from it
  to_source: boolean;
};

export type AssociationCreate = {
  /// The kind of association to make
  kind: AssociationKind;
  /// The piece of data this association starts with
  source: AssociationTarget;
  /// The piece of data this association starts with
  targets: AssociationTarget[];
  /// The groups for this association
  groups: string[];
};

export const BlankAssociationCreate: AssociationCreate = {
  kind: AssociationKind.AssociatedWith,
  source: {},
  targets: [],
  groups: [],
};
