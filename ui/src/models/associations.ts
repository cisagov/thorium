export enum AssociationKind {
  /// This file or repo is or contains firmware for a device
  FirmwareFor = 'FirmwareFor',
  // This file or repo is a file related to a device
  FileFor = 'File For',
  // Manual or other docs for hardware or software
  DocumentationFor = 'Documentation For',
  // Company owns company that makes/sells hardware or software
  ParenCompanyOf = 'Parent Company Of',
  /// This was developed by
  DevelopedBy = 'Developed By',
  /// This contains a CVE
  ContainsCVE = 'Contains CVE',
  /// This contains a CWE
  ContainsCWE = 'Contains CWE',
  /// This is based in specific countries
  BasedIn = 'Based In',
  /// This is used by a specific person or group
  UsedBy = 'Used By',
  /// This was used in a specific campaign or engagement
  UsedIn = 'Used In',
  /// This campaign was performed by
  PerformedBy = 'Performed By',
  AssociatedWith = 'Associated With',
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
